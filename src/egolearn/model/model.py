"""EgoModel: trunks (backbone encoders), named taps, and heads.

A tap is a named feature output of a trunk (e.g. vision.pool2048, the pooled
ResNeXt-50 features); a head is a small module (projector, classifier)
attached to a tap by name.  Losses and probes may only touch the model
through taps and heads, never through internals.  A checkpoint stores
(trunks, heads, tap names, preprocessing spec) -- every checkpoint records
the tap and head names, so a trained encoder can be reloaded elsewhere
(probes, a VLM) by name.

Sharing a trunk between heads needs no extra machinery: several heads simply
reference the same tap (BabyCL: proj128, cls and embed512 all hang off
vision.pool2048).

Note (from design review): slots -- the named groups of items inside one
batch -- are deliberately NEVER fused into a single trunk forward, because
that changes BatchNorm statistics and so changes model behavior.  What IS
shared is the per-slot, per-view trunk forward (`precompute_feats`): the
original trainers run the backbone once per crop and feed every head from
that result, so recomputing it per loss would double memory, compute and
BatchNorm updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EgoModel(nn.Module):
    def __init__(
        self, vision: nn.Module, text: nn.Module | None = None, heads: dict[str, dict] | None = None
    ):
        """heads: name -> {"tap": tap_name, "module": nn.Module}"""
        super().__init__()
        self.vision = vision
        self.text = text
        self._head_tap: dict[str, str] = {}
        mods = {}
        for name, spec in (heads or {}).items():
            self._head_tap[name] = spec["tap"]
            mods[name] = spec["module"]
        self.heads = nn.ModuleDict(mods)

    # ---- taps ---------------------------------------------------------------
    def taps(self) -> list[str]:
        out = [str(self.vision.tap)]
        if self.text is not None:
            out.append(str(self.text.tap))
        return out

    def vision_tap(self, x: torch.Tensor, tap: str | None = None) -> torch.Tensor:
        if tap is not None and tap != self.vision.tap:
            raise KeyError(f"model has no vision tap {tap!r} (has {self.vision.tap!r})")
        return self.vision(x)

    def text_tap(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.text is None:
            raise RuntimeError("model has no text trunk")
        return self.text(ids, lengths)

    def precompute_feats(self, batch: dict, device) -> None:
        """Run the vision trunk once per slot and view, storing the result in
        `slot["feats"]` as [B, V, D]; losses read these instead of re-running
        the trunk.  Each forward sees exactly one slot's one view, so the
        BatchNorm batch composition matches the original per-crop trainers.
        Clip-unit slots ([B, V, T, ...]) are skipped -- no video trunk yet."""
        for slot in batch.values():
            views = slot.get("views")
            if views is None or views.dim() != 5:
                continue
            v = views.to(device, non_blocking=True)
            feats = [self.vision_tap(v[:, i].contiguous()) for i in range(v.size(1))]
            slot["feats"] = torch.stack(feats, dim=1)

    # ---- heads ----------------------------------------------------------------
    def head_names(self) -> list[str]:
        return sorted(self.heads.keys())

    def head_tap(self, name: str) -> str:
        return self._head_tap[name]

    def apply_head(self, name: str, feats: torch.Tensor) -> torch.Tensor:
        return self.heads[name](feats)

    # ---- export contract: what a checkpoint records so encoders reload by name -
    def export_contract(self) -> dict:
        return {
            "taps": self.taps(),
            "heads": {n: self._head_tap[n] for n in self.heads},
            "preprocessing": "imagenet_norm",  # views normalize with ImageNet stats
        }
