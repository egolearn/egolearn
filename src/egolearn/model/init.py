"""Initialize part of a model from a trained checkpoint before training.

Staged training (pretrain a vision encoder, then fine-tune it with a new
head and a new modality) needs a way to start a run from another run's
weights *by name*, without resuming its optimizer, clocks, or memories.
`load_pretrained_weights` copies the tensors under the requested name
prefixes from a checkpoint or snapshot (`state["model"]`, the format every
CheckpointHook/SnapshotHook file shares) into a freshly built model.

Config surface (root `init_from:` key, applied in pipeline.build right after
the model is built and before the optimizer sees it):

    init_from:
      checkpoint: runs/stage1/final.pth
      include: ["encoders.vision."]     # name prefixes; default: everything
      exclude: []                        # prefixes to skip
      strict: true                       # see below

Strict mode (the default) refuses a partial encoder: every model tensor under
an included prefix must come from the checkpoint with the same shape, and
every included prefix must match something -- a stale prefix or a changed
architecture fails at build time instead of silently training from random
init.  Buffers (BatchNorm running statistics) are copied along with
parameters.  Loading consumes no RNG, so a run's determinism contract is
unchanged; the copied weights simply replace the seed-derived initialization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn


def _selected(name: str, include: Sequence[str] | None, exclude: Sequence[str]) -> bool:
    if include is not None and not any(name.startswith(p) for p in include):
        return False
    return not any(name.startswith(p) for p in exclude)


def load_pretrained_weights(
    model: nn.Module,
    checkpoint: str,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    strict: bool = True,
) -> dict:
    """Copy `checkpoint`'s model tensors under `include` prefixes into `model`.

    Returns a provenance summary: path, sha256, the source run's step, the
    prefixes, and how many tensors were copied (and skipped, when not strict).
    """
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"init_from.checkpoint not found: {path}")
    include = list(include) if include is not None else None
    exclude = list(exclude or [])
    if include is not None and not include:
        raise ValueError("init_from.include is empty; omit it to copy every tensor")

    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        # snapshots carry refs and RNG states next to the tensors; like
        # utils.snapshot and evaluation.checkpoint, only load trusted files
        state = torch.load(file, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"{path} is not an EgoLearn checkpoint/snapshot (no 'model' key)")
    source: dict[str, torch.Tensor] = state["model"]
    target = model.state_dict()

    if include is not None:
        for prefix in include:
            if not any(k.startswith(prefix) for k in source):
                raise ValueError(
                    f"init_from.include prefix {prefix!r} matches nothing in {path.name} "
                    f"(checkpoint tensors look like {next(iter(source), '?')!r})"
                )
            if not any(k.startswith(prefix) for k in target):
                raise ValueError(
                    f"init_from.include prefix {prefix!r} matches no tensor of this model "
                    f"(model tensors look like {next(iter(target), '?')!r})"
                )

    copied: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in source.items():
        if not _selected(name, include, exclude):
            continue
        if name not in target:
            if strict:
                raise ValueError(
                    f"init_from: checkpoint tensor {name!r} has no counterpart in the model "
                    "(different architecture or naming? narrow `include`, or strict: false)"
                )
            skipped.append(name)
            continue
        if tuple(tensor.shape) != tuple(target[name].shape):
            if strict:
                raise ValueError(
                    f"init_from: shape mismatch for {name!r}: checkpoint "
                    f"{tuple(tensor.shape)} vs model {tuple(target[name].shape)}"
                )
            skipped.append(name)
            continue
        copied[name] = tensor
    if strict:
        # every model tensor under an included prefix must have been supplied
        missing = [
            name for name in target if _selected(name, include, exclude) and name not in copied
        ]
        if missing:
            raise ValueError(
                f"init_from: {len(missing)} model tensor(s) under the included prefixes are "
                f"absent from {path.name}, e.g. {missing[:3]} -- a partially initialized "
                "component would silently train from random weights"
            )
    if not copied:
        raise ValueError(f"init_from: nothing selected from {path.name}")

    result = model.load_state_dict(copied, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    step = state.get("step", (state.get("context") or {}).get("clock", {}).get("step"))
    return {
        "checkpoint": str(path),
        "sha256": digest,
        "step": step,
        "include": include,
        "exclude": exclude,
        "strict": bool(strict),
        "tensors": len(copied),
        "skipped": skipped,
    }
