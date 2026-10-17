"""Identify configured inputs without inventing revisions for local datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egolearn.hub import read_checksums, resolve_with_metadata, sha256_file, verify_checksums
from egolearn.utils.model_artifact import METADATA, WEIGHTS, export_sha256, resolve_model_files

# These are input arguments used by the shipped streams, tokenizers and probes.
# Do not guess from arbitrary strings or inspect output/config/cache directories.
_PATH_KEYS = {
    "checkpoint",
    "root_dir",
    "data_root",
    "image_root",
    "image_dir",
    "pos_dir",
    "neg_dir",
    "manifest",
    "manifest_json",
    "abstract_2d_manifest",
    "rendered_3d_manifest",
    "vocab_json",
    "transcript_json",
    "split_file",
    "csv_path",
    "storyboard_json",
    "metadata_csv",
    "trials_json",
}
_INPUT_SECTIONS = {"stream", "tokenizer", "probes", "benchmarks", "init_from"}


def input_identity(
    source: str | Path, *, repo_type: str = "dataset", verify_directory: bool = False
) -> tuple[str, dict]:
    """Observe a file or inventory; full collection verification is opt-in.

    Hub paths are resolved once and the returned local path is used by the run.
    A file hash identifies that file only, never all the images named inside it.
    An unverified inventory hash never claims to identify the directory contents.
    """
    source = str(source)
    identity: dict[str, Any] = {"sha256": "unavailable", "revision": "unavailable"}
    path = Path(source).expanduser()
    if repo_type == "model" and source.startswith("hf://") and path.suffix == ".safetensors":
        # Fetch both exported files at the immutable revision before replacing
        # the Hub reference with a local path in the execution config.
        path, _, metadata = resolve_model_files(source)
        identity.update(metadata)
    elif source.startswith("hf://"):
        path, metadata = resolve_with_metadata(source, repo_type=repo_type)
        identity.update(metadata)
        # `revision` in Hub metadata is the requested tag; commit is immutable.
    identity["name"] = path.name
    try:
        if repo_type == "model" and (
            path.suffix == ".safetensors" or (path.is_dir() and (path / WEIGHTS).is_file())
        ):
            weights = path / WEIGHTS if path.is_dir() else path
            weights_sha = sha256_file(weights)
            metadata_sha = sha256_file(weights.with_name(METADATA))
            identity.update(
                kind="model_export",
                sha256=export_sha256(weights_sha, metadata_sha),
                weights_sha256=weights_sha,
                metadata_sha256=metadata_sha,
            )
        elif path.is_file():
            identity.update(kind="file", sha256=sha256_file(path))
        elif path.is_dir():
            identity["kind"] = "directory"
            inventory = path / "checksums.sha256"
            if inventory.is_file():
                read_checksums(path)
                identity["inventory_sha256"] = sha256_file(inventory)
                if not verify_directory:
                    identity["reason"] = "inventory_not_verified"
                elif verify_checksums(path):
                    identity["reason"] = "checksum_verification_failed"
                else:
                    identity.update(
                        sha256=identity["inventory_sha256"], checksum_scope="checksums.sha256"
                    )
            else:
                identity["reason"] = "no_checksum_inventory"
        else:
            identity["reason"] = "missing"
    except OSError as error:
        identity["reason"] = type(error).__name__
    except (ValueError, RuntimeError):
        identity["reason"] = "invalid_checksum_inventory"
    return str(path), identity


def prepare_inputs(
    config: dict, *, descriptive_sections: frozenset[str] = frozenset()
) -> dict[str, dict]:
    """Capture known inputs and freeze Hub resolution in this run's config copy.

    Local paths retain their spelling; only Hub references are replaced. Unknown
    component input conventions remain in the config and have no inferred digest.
    """
    inputs: dict[str, dict] = {}
    observed: dict[tuple[str, str], tuple[str, dict]] = {}

    def capture(parent, key, name: str, *, model: bool = False) -> None:
        source = parent[key]
        if not isinstance(source, (str, Path)) or not source or source == "auto":
            return
        if config.get("dry_run") or any(
            name.startswith(prefix + ".") for prefix in descriptive_sections
        ):
            inputs[name] = {
                "sha256": "unavailable",
                "revision": "unavailable",
                "reason": "dry_run" if config.get("dry_run") else "preconstructed_benchmark",
            }
            return
        kind = "model" if model else "dataset"
        cache_key = (str(source), kind)
        if cache_key not in observed:
            observed[cache_key] = input_identity(source, repo_type=kind)
        path, identity = observed[cache_key]
        inputs[name] = dict(identity)
        if str(source).startswith("hf://"):
            parent[key] = path

    def visit(node, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                name = f"{prefix}.{key}"
                if key in _PATH_KEYS:
                    capture(node, key, name, model=key == "checkpoint")
                elif isinstance(value, (dict, list)):
                    visit(value, name)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{prefix}.{index}")

    for section in sorted(_INPUT_SECTIONS):
        visit(config.get(section), section)
    for key in ("checkpoint", "resume", "model_config"):
        if key in config:
            capture(config, key, key, model=True)
    return inputs
