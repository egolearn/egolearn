"""Validate publication records and aggregate matching training seeds."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from copy import deepcopy
from typing import Any

SCHEMA_VERSIONS = (1, 2)
METRICS = [
    ("labeled_s_4afc", "four_afc_acc", "Labeled-S 4AFC", "Four-alternative image-word matching."),
    ("babyview_4afc", "four_afc_acc", "BabyView 4AFC", "Four-alternative image-word matching."),
    (
        "konkle_4afc",
        "four_afc_acc",
        "Konkle 4AFC",
        "Four-alternative word-object matching on Konkle object photographs.",
    ),
    ("vtwt", "vtwt_acc", "VTWT", "Visual word-text matching accuracy."),
    ("baby_winoground", "bw_group_acc", "Baby Winoground", "Group accuracy."),
    (
        "shape_bias",
        "shape_bias",
        "Shape preference",
        "Shape preference, not classification accuracy.",
    ),
    (
        "labeled_s_linear",
        "top1",
        "Labeled-S linear",
        "Frozen-feature linear decoding, top-1 accuracy.",
    ),
    (
        "babyview_linear",
        "top1",
        "BabyView linear",
        "Frozen-feature linear decoding, top-1 accuracy.",
    ),
    (
        "imagenet_linear",
        "top1",
        "ImageNet linear",
        "Frozen-feature linear decoding, top-1 accuracy.",
    ),
]
DEFAULT_COLUMNS = [
    {
        "id": name,
        "benchmark": name,
        "metric": metric,
        "label": label,
        "description": description,
        "unit": "fraction",
        "direction": "max",
    }
    for name, metric, label, description in METRICS
]


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path: Any) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def _keys(value: Any, keys: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"expected exactly these fields: {keys}")


def sha256(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("expected a SHA-256 digest")


def identifier(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value):
        raise ValueError("invalid publication identifier")


def repo_id(value: Any) -> None:
    if not isinstance(value, str) or len(value.split("/")) != 2:
        raise ValueError("expected an org/repository identifier")
    for part in value.split("/"):
        identifier(part)


def portable(value: Any) -> None:
    """Reject paths, credentials, nonfinite numbers and oversized metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            portable(key)
            portable(child)
    elif isinstance(value, list):
        for child in value:
            portable(child)
    elif isinstance(value, str):
        if len(value) > 1000 or re.search(
            r"(^[/~]|[A-Za-z]:\\|hf_[A-Za-z0-9]{15,}|[\r\n\x00])", value
        ):
            raise ValueError("publication metadata contains a path, credential or multiline text")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("publication numbers must be finite")
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise ValueError("unsupported publication metadata")


def run_id(run: dict[str, Any]) -> str:
    return digest(
        {
            "condition": run["condition"],
            "seed": run["training_seed"],
            "checkpoint": run["checkpoint"]["sha256"],
        }
    )


def model_directory(run: dict[str, Any]) -> str:
    """Return the bundle directory for a seeded run or an imported reference."""
    return "reference" if run["training_seed"] is None else str(run["training_seed"])


def validate_column(column: dict[str, Any]) -> None:
    _keys(column, "id benchmark metric label description unit direction")
    if any(not isinstance(value, str) or not value.strip() for value in column.values()):
        raise ValueError("column fields must be nonempty strings")
    for key in ("id", "benchmark", "metric"):
        identifier(column[key])
    if column["unit"] not in {"fraction", "number"} or column["direction"] not in {"min", "max"}:
        raise ValueError("unsupported metric unit or direction")


def validate_run(run: dict[str, Any], *, published: bool = False) -> None:
    version = run.get("schema_version")
    _keys(
        run,
        "schema_version id condition training_seed expected_seeds "
        "checkpoint provenance evaluations" + (" not_applicable" if version == 2 else ""),
    )
    portable(run)
    if type(version) is not int or version not in SCHEMA_VERSIONS or run["id"] != run_id(run):
        raise ValueError("unsupported schema or incorrect run identity")
    condition = run["condition"]
    _keys(condition, "id label dataset regime architecture description")
    identifier(condition["id"])
    if any(not isinstance(v, str) or not v for v in condition.values()):
        raise ValueError("condition fields must be nonempty strings")
    provenance = run["provenance"]
    _keys(
        provenance,
        "training_config_sha256 training_code_commit legacy_evaluation"
        + (" reference" if version == 2 else ""),
    )
    if version == 2 and type(provenance["reference"]) is not bool:
        raise ValueError("reference must be a boolean")
    seeds = run["expected_seeds"]
    reference = version == 2 and provenance["reference"]
    if reference:
        if seeds != [None] or run["training_seed"] is not None:
            raise ValueError("an unknown-seed reference requires training_seed=null and [null]")
    elif (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int for s in seeds)
        or seeds != sorted(set(seeds))
        or type(run["training_seed"]) is not int
        or run["training_seed"] not in seeds
    ):
        raise ValueError("expected_seeds must be sorted unique integers including training_seed")
    checkpoint = run["checkpoint"]
    _keys(checkpoint, "sha256 step weights_sha256 metadata_sha256 repo_id revision private")
    for key in ("sha256", "weights_sha256", "metadata_sha256"):
        sha256(checkpoint[key])
    if checkpoint["step"] is not None and type(checkpoint["step"]) is not int:
        raise ValueError("checkpoint step must be an integer or null")
    revision = checkpoint["revision"]
    if checkpoint["repo_id"] is None:
        if revision is not None or checkpoint["private"] is not None:
            raise ValueError("local checkpoints must not claim Hub revision or visibility")
    else:
        repo_id(checkpoint["repo_id"])
        if type(checkpoint["private"]) is not bool:
            raise ValueError("checkpoint visibility must be explicit")
        if (published or revision is not None) and (
            not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision)
        ):
            raise ValueError("hosted checkpoints require an immutable Hub commit")
    if reference:
        if provenance["training_config_sha256"] is not None:
            raise ValueError("unknown-seed references must not claim a training config")
    else:
        sha256(provenance["training_config_sha256"])
    if type(provenance["legacy_evaluation"]) is not bool:
        raise ValueError("legacy_evaluation must be a boolean")
    commit = provenance["training_code_commit"]
    if commit is not None and (
        not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit)
    ):
        raise ValueError("invalid training code commit")
    if not isinstance(run["evaluations"], list) or not run["evaluations"]:
        raise ValueError("a published run needs evaluations")
    seen = set()
    evaluated_metrics = set()
    for evaluation in run["evaluations"]:
        _keys(evaluation, "column value protocol protocol_sha256 source_sha256")
        column = evaluation["column"]
        validate_column(column)
        if column["id"] in seen:
            raise ValueError("duplicate metric in one run")
        seen.add(column["id"])
        evaluated_metrics.add((column["benchmark"], column["metric"]))
        value = evaluation["value"]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("metrics must be finite numbers")
        if column["unit"] == "fraction" and not 0 <= value <= 1:
            raise ValueError("fraction metrics must be between zero and one")
        sha256(evaluation["source_sha256"])
        protocol = evaluation["protocol"]
        _keys(protocol, "version settings input_sha256 input_scope evaluation_seed code_commit")
        if not isinstance(protocol["version"], str) or not protocol["version"]:
            raise ValueError("benchmark version is required")
        if not isinstance(protocol["input_scope"], str) or not protocol["input_scope"]:
            raise ValueError("protocol input scope is required")
        if not isinstance(protocol["settings"], dict) or not isinstance(
            protocol["input_sha256"], dict
        ):
            raise ValueError("protocol settings and inputs must be objects")
        for checksum in protocol["input_sha256"].values():
            sha256(checksum)
        if type(protocol["evaluation_seed"]) is not int:
            raise ValueError("evaluation seed is required")
        commit = protocol["code_commit"]
        if commit is not None and (
            not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit)
        ):
            raise ValueError("invalid evaluation code commit")
        if evaluation["protocol_sha256"] != digest(protocol):
            raise ValueError("protocol identity mismatch")
    unavailable = run.get("not_applicable", [])
    if not isinstance(unavailable, list):
        raise ValueError("not_applicable must be a list")
    for entry in unavailable:
        _keys(entry, "column reason")
        validate_column(entry["column"])
        name = entry["column"]["id"]
        metric = (entry["column"]["benchmark"], entry["column"]["metric"])
        if name in seen or metric in evaluated_metrics:
            raise ValueError("a metric cannot be evaluated and not applicable, or declared twice")
        seen.add(name)
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError("not_applicable needs a nonempty reason")


def merge_runs(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for run in [*existing, *incoming]:
        validate_run(run, published=True)
        previous = by_id.get(run["id"])
        if previous is not None and previous != run:
            same_content = deepcopy(run)
            for key in ("repo_id", "revision", "private"):
                same_content["checkpoint"][key] = previous["checkpoint"][key]
            if same_content == previous:
                old_repo = previous["checkpoint"]["repo_id"]
                new_repo = run["checkpoint"]["repo_id"]
                if old_repo is None and new_repo is not None:
                    # Adding a verified model location does not change evaluation evidence.
                    by_id[run["id"]] = run
                    continue
                if new_repo is None or old_repo == new_repo:
                    # Keep the original pin after card edits or results-only retries.
                    continue
            raise ValueError(
                f"conflicting results for run {run['id']}; records are immutable. "
                "Reuse the original prepared bundle to attach checkpoint weights."
            )
        by_id[run["id"]] = run
    return sorted(by_id.values(), key=lambda run: run["id"])


def build_table(runs: list[dict], *, results_repo: str, updated_at: str) -> dict:
    """Compute sample SD only over complete, compatible, distinct training seeds."""
    repo_id(results_repo)
    groups = defaultdict(list)
    columns: dict[str, dict] = {}
    checkpoints: set[str] = set()
    for run in runs:
        validate_run(run, published=True)
        checksum = run["checkpoint"]["sha256"]
        if checksum in checkpoints:
            raise ValueError("a checkpoint cannot count as more than one training run")
        checkpoints.add(checksum)
        groups[run["condition"]["id"]].append(run)
        for evaluation in [*run["evaluations"], *run.get("not_applicable", [])]:
            column = evaluation["column"]
            if column["id"] in columns and columns[column["id"]] != column:
                raise ValueError("conflicting column definitions")
            columns[column["id"]] = column
    rows = []
    for condition_id, members in sorted(groups.items()):
        first = members[0]
        expected = first["expected_seeds"]
        if any(
            r["condition"] != first["condition"] or r["expected_seeds"] != expected for r in members
        ):
            raise ValueError(f"conflicting condition or seed set for {condition_id}")
        if len({r["training_seed"] for r in members}) != len(members):
            raise ValueError("multiple checkpoints for the same condition and training seed")
        unavailable = {
            entry["column"]["id"]: entry["reason"] for entry in first.get("not_applicable", [])
        }
        if any(
            {entry["column"]["id"]: entry["reason"] for entry in r.get("not_applicable", [])}
            != unavailable
            for r in members
        ):
            raise ValueError(f"conflicting benchmark applicability for {condition_id}")
        cells = {}
        for column_id in columns:
            if column_id in unavailable:
                cells[column_id] = {
                    "mean": None,
                    "sd": None,
                    "n": 0,
                    "expected_n": len(expected),
                    "status": "not_applicable",
                    "protocols": [],
                    "reason": unavailable[column_id],
                }
                continue
            values = [
                e for r in members for e in r["evaluations"] if e["column"]["id"] == column_id
            ]
            protocols = {e["protocol_sha256"] for e in values}
            complete = len(values) == len(expected) and len(protocols) == 1
            numbers = [e["value"] for e in values]
            cells[column_id] = {
                "mean": statistics.mean(numbers) if complete else None,
                "sd": statistics.stdev(numbers) if complete and len(numbers) > 1 else None,
                "n": len(numbers),
                "expected_n": len(expected),
                "status": "complete"
                if complete
                else "incompatible"
                if len(protocols) > 1
                else "incomplete",
                "protocols": [
                    {
                        "sha256": p,
                        **next(e["protocol"] for e in values if e["protocol_sha256"] == p),
                    }
                    for p in sorted(protocols)
                ],
            }
        rows.append(
            {
                "id": condition_id,
                "condition": first["condition"],
                "expected_seeds": expected,
                "cells": cells,
                "runs": [
                    {
                        "id": r["id"],
                        "seed": r["training_seed"],
                        "checkpoint": r["checkpoint"],
                        "provenance": r["provenance"],
                    }
                    for r in sorted(members, key=lambda r: r["training_seed"])
                ],
            }
        )
    order = {column["id"]: i for i, column in enumerate(DEFAULT_COLUMNS)}
    return {
        "schema_version": max((r["schema_version"] for r in runs), default=1),
        "updated_at": updated_at,
        "results_repo": results_repo,
        "columns": sorted(columns.values(), key=lambda c: (order.get(c["id"], 1000), c["id"])),
        "rows": rows,
    }
