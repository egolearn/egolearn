"""EgoModel: trunk registry + named taps + heads.

Losses and probes may only touch the model through taps and heads -- this is the
export contract's foundation: a checkpoint is (trunks, heads, tap names,
preprocessing spec), so any trained encoder can be dropped into probes or a
VLM by tap name.

Physically shared trunks come for free: several heads referencing the same
tap (BabyCL: proj128, cls and embed512 all hang off vision.pool2048).

Note (design review): slots are deliberately NEVER fused into one forward --
that changes BatchNorm statistics and is not semantically neutral.  What IS
shared is the per-slot, per-view trunk forward (`precompute_feats`): the
original trainers run the backbone once per crop and feed every head from
it, so recomputing it per loss would double memory, compute and BN updates.
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
        """One trunk forward per slot and view, written to `slot["feats"]`
        as [B, V, D]; losses consume these instead of re-running the trunk.
        Each forward sees exactly one slot's one view, so BatchNorm batch
        composition matches the original per-crop trainers. Clip-unit slots
        ([B, V, T, ...]) are skipped -- no video trunk yet."""
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

    # ---- export contract ------------------------------------------------------
    def export_contract(self) -> dict:
        return {
            "taps": self.taps(),
            "heads": {n: self._head_tap[n] for n in self.heads},
            "preprocessing": "imagenet_norm",  # views normalize with ImageNet stats
        }
