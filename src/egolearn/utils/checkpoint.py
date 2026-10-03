"""Shared checkpoint reading and portable source identification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def read_checkpoint(
    checkpoint: str | Path, *, trusted_snapshot: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read model state and provenance; see SECURITY.md for the trust rule.

    Ordinary checkpoints use restricted unpickling. Only explicitly trusted
    snapshots may contain arbitrary Python objects; there is no unsafe retry.
    """
    path = Path(checkpoint).expanduser()
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        state = torch.load(file, map_location="cpu", weights_only=not trusted_snapshot)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError("expected an EgoLearn checkpoint or snapshot containing a 'model' mapping")
    step = state.get("step")
    if step is None:
        step = ((state.get("context") or {}).get("clock") or {}).get("step")
    return state, {"filename": path.name, "sha256": digest, "step": step}
