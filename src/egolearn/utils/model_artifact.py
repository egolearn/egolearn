"""Validate portable model exports without importing torch or executing model code."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

WEIGHTS = "model.safetensors"
METADATA = "model.json"


def file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def inspect_export(weights: Path, metadata: Path | None = None) -> dict[str, Any]:
    """Validate the JSON schema, weight digest, and safetensors header."""
    metadata = metadata if metadata is not None else weights.with_name(METADATA)
    document = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ValueError("expected a model.json export with format_version: 1")
    config = document.get("model_config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("model.json requires a model construction config")
    if not isinstance(document.get("export_contract"), dict):
        raise ValueError("model.json requires an export contract")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("model.json requires source provenance")
    filename = provenance.get("filename")
    if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
        raise ValueError("export provenance requires a source filename without directories")
    digest = provenance.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("export provenance requires the source SHA-256")
    if "step" not in provenance or (
        provenance["step"] is not None and type(provenance["step"]) is not int
    ):
        raise ValueError("export provenance requires an integer step or null")
    revision = provenance.get("code_revision")
    if not isinstance(revision, dict) or not isinstance(revision.get("commit"), str):
        raise ValueError("export provenance requires the export code revision")
    if document.get("weights_sha256") != file_digest(weights):
        raise ValueError("model.safetensors SHA-256 does not match model.json")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ValueError("model exports require safetensors; install the train extra") from error
    # Header validation does not materialize tensors or import a model framework.
    with safe_open(weights, framework="numpy") as tensors:
        if not tensors.keys():
            raise ValueError("model.safetensors contains no tensors")
    return document


def validate_export(root: Path, *, publishing: bool = False) -> dict[str, Any]:
    """Require the exported pair; model publishing accepts no additional payloads."""
    if publishing:
        payloads = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
            and path.relative_to(root).as_posix() not in {"README.md", "checksums.sha256"}
        }
        if payloads != {WEIGHTS, METADATA}:
            raise ValueError(
                "model uploads require only model.safetensors and model.json from egl export; "
                ".pth files and other payloads are refused"
            )
    return inspect_export(root / WEIGHTS, root / METADATA)
