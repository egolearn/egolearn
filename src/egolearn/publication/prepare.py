"""Prepare a portable publication bundle from local checkpoints and evaluations."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from egolearn.publication.schema import (
    DEFAULT_COLUMNS,
    digest,
    encoded,
    identifier,
    model_directory,
    portable,
    read_json,
    run_id,
    validate_run,
)
from egolearn.utils.input_provenance import INPUT_PATH_KEYS
from egolearn.utils.model_artifact import export_sha256, file_digest

# Paths describe local storage, not evaluation protocols. Hashes identify inputs.
DIRECTORY_INPUTS = {
    "data_root",
    "root_dir",
    "image_root",
    "image_dir",
    "pos_dir",
    "neg_dir",
    "train_dir",
    "val_dir",
    "cache_dir",
}
INPUT_PATHS = INPUT_PATH_KEYS | DIRECTORY_INPUTS | {"vocabulary"}
FILE_INPUTS = INPUT_PATHS - DIRECTORY_INPUTS - {"checkpoint"}
OUTPUT_PATHS = {"output_json", "output_dir", "feature_cache"}


def import_evaluations(
    paths: list[Path],
    *,
    checkpoint: dict,
    columns: list[dict],
    input_hashes: list[dict],
    allow_legacy: bool = False,
) -> tuple[list[dict], bool]:
    evaluations = []
    legacy = False
    for path in paths:
        document = read_json(path)
        version = document.get("schema_version")
        if version not in (1, 2):
            raise ValueError(f"unsupported evaluation schema in {path.name}")
        if document.get("status") != "complete":
            if not (version == 1 and allow_legacy and "status" not in document):
                raise ValueError(f"evaluation is not complete: {path.name}")
            legacy = True
        if document.get("failures") or document.get("failed_benchmark"):
            raise ValueError(f"evaluation has failures: {path.name}")
        origin = document.get("checkpoint") or {}
        valid_digests = {
            checkpoint["sha256"],
            export_sha256(checkpoint["weights_sha256"], checkpoint["metadata_sha256"]),
        }
        if origin.get("sha256") not in valid_digests:
            raise ValueError(f"evaluation checkpoint does not match the export: {path.name}")
        if origin.get("step") != checkpoint["step"]:
            raise ValueError(f"evaluation checkpoint step differs: {path.name}")
        source_hash = file_digest(path)
        for column in columns:
            name, metric = column["benchmark"], column["metric"]
            scores = document.get("results", {}).get(name, {})
            if metric not in scores:
                continue
            count_keys = {
                "vtwt": ("vtwt_n", "vtwt_skipped"),
                "baby_winoground": ("bw_n", "bw_skipped"),
            }.get(name)
            if count_keys:
                evaluated_key, skipped_key = count_keys
                evaluated, skipped = scores.get(evaluated_key), scores.get(skipped_key)
                if (
                    type(evaluated) not in (int, float)
                    or not math.isfinite(evaluated)
                    or evaluated <= 0
                    or evaluated % 1 != 0
                    or type(skipped) not in (int, float)
                    or skipped != 0
                ):
                    raise ValueError(
                        f"incomplete {name} evaluation in {path.name}: "
                        f"require a positive integer {evaluated_key} and zero {skipped_key}"
                    )
            original_settings = document["configs"][name]
            settings = {
                k: v for k, v in original_settings.items() if k not in INPUT_PATHS | OUTPUT_PATHS
            }
            portable(settings)
            pins: dict[str, str] = {}
            for pin in input_hashes:
                if pin["benchmark"] == name:
                    if pin["kind"] in pins and pins[pin["kind"]] != pin["sha256"]:
                        raise ValueError("conflicting recorded input hashes")
                    pins[pin["kind"]] = pin["sha256"]
            missing = {k for k in FILE_INPUTS if original_settings.get(k) and k not in pins}
            if missing:
                raise ValueError(f"missing recorded input hashes for {name}: {sorted(missing)}")
            # Preserve class ordering and sample counts without publishing class/sample lists.
            metadata = document.get("metadata", {}).get(name, {})
            if metadata:
                pins["evaluation_metadata"] = digest(metadata)
            if not pins:
                raise ValueError(f"evaluation has no recorded input identity: {name}")
            revision = document.get("code_revision", {})
            code_commit = revision.get("commit")
            if code_commit in {None, "unavailable"} or revision.get("dirty"):
                code_commit = None
            protocol = {
                "version": document["versions"][name],
                "settings": settings,
                "input_sha256": pins,
                "input_scope": "Recorded input files and metadata; image content is not verified.",
                "evaluation_seed": document["seed"],
                "code_commit": code_commit,
            }
            evaluations.append(
                {
                    "column": column,
                    "value": scores[metric],
                    "protocol": protocol,
                    "protocol_sha256": digest(protocol),
                    "source_sha256": source_hash,
                }
            )
    return evaluations, legacy


def prepare(manifest: Path, destination: Path, *, allow_legacy: bool = False) -> Path:
    """Export models and validate their evaluation evidence in a new directory."""
    from omegaconf import DictConfig, OmegaConf

    from egolearn.evaluation import load_model
    from egolearn.export import export_checkpoint

    if destination.exists():
        raise FileExistsError(f"publication destination exists: {destination}")
    plan = read_json(manifest)
    if plan.get("schema_version") != 1 or not plan.get("runs"):
        raise ValueError("expected a version 1 publication manifest with runs")
    columns = plan.get("columns", DEFAULT_COLUMNS)
    condition = plan["condition"]
    identifier(condition["id"])
    reference = plan.get("reference", False)
    if type(reference) is not bool:
        raise ValueError("reference must be a boolean")
    if reference and (plan["expected_seeds"] != [None] or len(plan["runs"]) != 1):
        raise ValueError("publish each unknown-seed reference as one condition with [null]")
    unavailable = plan.get("not_applicable", {})
    if not isinstance(unavailable, dict):
        raise ValueError("not_applicable must map column IDs to reasons")
    by_id = {column["id"]: column for column in columns}
    if unavailable.keys() - by_id.keys():
        raise ValueError("not_applicable names an unknown column")
    not_applicable = [
        {"column": by_id[name], "reason": reason} for name, reason in sorted(unavailable.items())
    ]
    version = 2 if reference or not_applicable else 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publication-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        runs: list[dict[str, Any]] = []
        for spec in plan["runs"]:
            seed = spec["training_seed"]
            if reference:
                if seed is not None or spec.get("training_config") is not None:
                    raise ValueError(
                        "unknown-seed references require null seed and training_config"
                    )
                training_config_sha256 = None
            else:
                if type(seed) is not int:
                    raise ValueError("training seed must be an integer")
                training_config = OmegaConf.load(spec["training_config"])
                if (
                    not isinstance(training_config, DictConfig)
                    or type(training_config.get("seed")) is not int
                    or training_config.get("seed") != seed
                ):
                    raise ValueError("training seed differs from the saved training configuration")
                training_config_sha256 = file_digest(Path(spec["training_config"]))
            export = stage / "models" / model_directory(spec)
            export_checkpoint(Path(spec["checkpoint"]), export)
            metadata = read_json(export / "model.json")
            # A strict model reconstruction catches missing imports and incompatible state dicts.
            restored = load_model(export, device="cpu")
            del restored
            source = metadata["provenance"]
            checkpoint = {
                "sha256": source["sha256"],
                "step": source["step"],
                "weights_sha256": metadata["weights_sha256"],
                "metadata_sha256": file_digest(export / "model.json"),
                "repo_id": spec["repo_id"],
                "revision": None,
                "private": True,
            }
            portable(metadata)
            pins = read_json(Path(spec["input_hashes"])) if spec.get("input_hashes") else []
            evaluations, legacy = import_evaluations(
                [Path(p) for p in spec["evaluations"]],
                checkpoint=checkpoint,
                columns=columns,
                input_hashes=pins,
                allow_legacy=allow_legacy,
            )
            training = source.get("training_provenance")
            revision = training if isinstance(training, dict) else {}
            if revision.get("seed", seed) != seed:
                raise ValueError("checkpoint training seed differs from the publication manifest")
            commit = revision.get("commit")
            if commit in {None, "unavailable"} or revision.get("dirty"):
                commit = None
            run = {
                "schema_version": version,
                "id": "",
                "condition": condition,
                "training_seed": seed,
                "expected_seeds": [None] if reference else sorted(plan["expected_seeds"]),
                "checkpoint": checkpoint,
                "provenance": {
                    "training_config_sha256": training_config_sha256,
                    "training_code_commit": commit,
                    "legacy_evaluation": legacy,
                },
                "evaluations": evaluations,
            }
            if version == 2:
                run["provenance"]["reference"] = reference
                run["not_applicable"] = not_applicable
            run["id"] = run_id(run)
            validate_run(run)
            runs.append(run)
        if len({r["training_seed"] for r in runs}) != len(runs):
            raise ValueError("duplicate training seed in publication manifest")
        (stage / "bundle.json").write_bytes(encoded({"schema_version": 1, "runs": runs}))
        stage.rename(destination)
    return destination
