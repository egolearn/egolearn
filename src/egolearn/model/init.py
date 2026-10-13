"""Initialize parameters and buffers selected by include/exclude name prefixes.

Strict loading requires complete, shape-compatible selections. Loading consumes
no RNG; snapshot loading requires explicit trust (see SECURITY.md). On a fresh V-JEPA run, the
step-0 sync replaces any loaded EMA target with the initialized context encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from egolearn.hub import resolve_with_metadata
from egolearn.utils.checkpoint import read_checkpoint


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
    trusted_snapshot: bool = False,
) -> dict:
    """Copy `checkpoint`'s model tensors under `include` prefixes into `model`.

    Returns a provenance summary: file name, sha256, the source run's step, the
    prefixes, and how many tensors were copied (and skipped, when not strict).
    Hub references also retain the original reference and resolved commit.
    """
    path, provenance = resolve_with_metadata(checkpoint, repo_type="model")
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"init_from.checkpoint not found: {path}")
    include = (
        [include] if isinstance(include, str) else list(include) if include is not None else None
    )
    exclude = [exclude] if isinstance(exclude, str) else list(exclude or [])
    if include is not None and not include:
        raise ValueError("init_from.include is empty; omit it to copy every tensor")

    state, metadata = read_checkpoint(path, trusted_snapshot=trusted_snapshot)
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
    return {
        **provenance,
        "checkpoint": metadata["filename"],
        "sha256": metadata["sha256"],
        "step": metadata["step"],
        "include": include,
        "exclude": exclude,
        "strict": bool(strict),
        "tensors": len(copied),
        "skipped": skipped,
    }
