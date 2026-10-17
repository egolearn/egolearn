"""Shared checkpoint reading and portable source identification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import torch

from egolearn.utils.model_artifact import (
    WEIGHTS,
    export_sha256,
    file_digest,
    inspect_export,
    resolve_model_files,
)


def read_checkpoint(
    checkpoint: str | Path, *, trusted_snapshot: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read an exported model or a legacy checkpoint; see SECURITY.md.

    Pickles use restricted unpickling unless snapshot trust is explicitly granted.
    There is no unsafe retry, including for invalid exported files.
    """
    if not isinstance(trusted_snapshot, bool):
        raise TypeError("trusted_snapshot must be a bool; only True enables unrestricted loading")
    path, metadata, hub = resolve_model_files(checkpoint)
    if metadata is not None:
        from safetensors import SafetensorError
        from safetensors.torch import load_file

        weights = path / WEIGHTS if path.is_dir() else path
        document = inspect_export(weights, metadata)
        try:
            tensors = load_file(weights, device="cpu")
        except SafetensorError as error:
            raise ValueError(f"invalid safetensors weights: {error}") from error
        state = {
            "model": tensors,
            "model_config": document["model_config"],
            "export_contract": document["export_contract"],
            "step": document["provenance"]["step"],
            "run_provenance": document["provenance"].get("training_provenance"),
        }
        # The artifact identity covers construction metadata as well as tensor bytes.
        metadata_sha = file_digest(metadata)
        digest = export_sha256(document["weights_sha256"], metadata_sha)
        return state, {
            **hub,
            "filename": weights.name,
            "sha256": digest,
            "step": state["step"],
            "weights_sha256": document["weights_sha256"],
            "metadata_sha256": metadata_sha,
            "source": document["provenance"],
            "training_provenance": state["run_provenance"] or "unavailable",
        }
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        try:
            state = torch.load(file, map_location="cpu", weights_only=not trusted_snapshot)
        except EOFError as error:
            raise EOFError("checkpoint is empty or truncated") from error
        except UnpicklingError as error:
            if trusted_snapshot:
                raise
            inner = error.__context__ if isinstance(error.__context__, UnpicklingError) else error
            reason = str(inner).split("WeightsUnpickler error:", 1)[-1].strip()
            reason = reason.splitlines()[0].rstrip(".") if reason else "unpickling failed"
            raise UnpicklingError(
                f"restricted checkpoint loading failed: {reason}. If this is a full snapshot "
                "whose producer and contents you trust, use trusted_snapshot=True "
                "(Hydra: trusted_snapshot=true or "
                "+init_from.trusted_snapshot=true; egl export: --trusted-snapshot)"
            ) from error
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError("expected an EgoLearn checkpoint or snapshot containing a 'model' mapping")
    step = state.get("step")
    if step is None:
        step = ((state.get("context") or {}).get("clock") or {}).get("step")
    return state, {
        **hub,
        "filename": path.name,
        "sha256": digest,
        "step": step,
        "training_provenance": state.get("run_provenance") or "unavailable",
    }
