"""Symmetric image-text InfoNCE (the textbook CLIP loss).

Core loss library.  Faithful to multimodel.MultiModalModel.
calculate_contrastive_loss (CVCL/BabyCL): L2-normalized embeddings, fixed
temperature, symmetric CE.

`use_sample_weights` reproduces BabyCL's clip-importance weighting: if the
slot carries per-sample weights (weight.sample flowing through
memory/composer), the loss becomes sum(w_i * l_i) / sum(w_i).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES

from .base import Loss


@LOSSES.register("clip_infonce")
class ClipInfoNCE(Loss):
    requires_text = True

    def __init__(
        self,
        slot: str,
        vision_head: str,
        temperature: float = 0.07,
        use_sample_weights: bool = False,
        name: str = "cross_modal",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.slot = slot
        self.vision_head = vision_head
        self.temperature = float(temperature)
        self.use_sample_weights = bool(use_sample_weights)
        self.requires_slots = [slot]
        self.requires_heads = [vision_head]

    def compute(self, batch, model, ctx) -> dict[str, torch.Tensor]:
        slot = self._slot(batch, self.slot)
        for need in ("text_ids", "text_len"):
            if need not in slot:
                raise KeyError(
                    f"clip_infonce: slot {self.slot!r} has no {need} -- set text: true in the "
                    "recipe (and configure a tokenizer)"
                )

        x = slot["views"][:, 0].to(ctx.device, non_blocking=True)
        img = model.apply_head(self.vision_head, model.vision_tap(x))
        txt = model.text_tap(slot["text_ids"].to(ctx.device), slot["text_len"].to(ctx.device))
        img = F.normalize(img, dim=1)
        txt = F.normalize(txt, dim=1)

        logits = img @ txt.t() / self.temperature
        target = torch.arange(logits.size(0), device=ctx.device)

        weights: torch.Tensor | None = slot.get("weights")
        if self.use_sample_weights and weights is not None:
            w = weights.to(ctx.device)
            per = (
                F.cross_entropy(logits, target, reduction="none")
                + F.cross_entropy(logits.t(), target, reduction="none")
            ) / 2
            loss = (w * per).sum() / w.sum().clamp_min(1e-8)
        else:
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
        return {"clip_infonce": loss}
