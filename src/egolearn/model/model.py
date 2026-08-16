"""EgoModel: trunk registry + named taps + heads.

Losses and probes may only touch the model through taps and heads -- this is the
export contract's foundation: a checkpoint is (trunks, heads, tap names,
preprocessing spec), so any trained encoder can be dropped into probes or a
VLM by tap name.

Physically shared trunks come for free: several heads referencing the same
tap (BabyCL: proj128, cls and embed512 all hang off vision.pool2048).

Note (design review): there is deliberately NO cross-loss forward
de-duplication here.  Fusing slots into one forward changes BatchNorm
statistics and is not semantically neutral; losses each call the taps
they need.
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
