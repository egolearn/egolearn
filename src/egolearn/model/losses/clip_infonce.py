"""Symmetric image-text InfoNCE (the textbook CLIP loss).

Part of the core loss library.  Faithful to multimodel.MultiModalModel.
calculate_contrastive_loss (CVCL/BabyCL): L2-normalized embeddings, fixed
temperature, symmetric cross-entropy in both directions (image-to-text and
text-to-image).

`use_sample_weights` reproduces BabyCL's clip-importance weighting: if the
slot carries per-sample weights (the weight.sample field, carried through
memory and the batch composer), the loss becomes sum(w_i * l_i) / sum(w_i).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES
from egolearn.core import keys as K

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
        for need in (K.SLOT_TEXT_IDS, K.SLOT_TEXT_LEN):
            if need not in slot:
                raise KeyError(
                    f"clip_infonce: slot {self.slot!r} has no {need} -- set text: true in the "
                    "recipe (and configure a tokenizer)"
                )

        if K.SLOT_FEATS in slot:
            img = model.apply_head(self.vision_head, slot[K.SLOT_FEATS][:, 0])
        else:
            x = slot[K.SLOT_VIEWS][:, 0].to(ctx.device, non_blocking=True)
            img = model.apply_head(self.vision_head, model.encode("image", x).pooled)
        txt = model.encode(
            "text", slot[K.SLOT_TEXT_IDS].to(ctx.device), slot[K.SLOT_TEXT_LEN].to(ctx.device)
        ).pooled
        img = F.normalize(img, dim=1)
        txt = F.normalize(txt, dim=1)

        logits = img @ txt.t() / self.temperature
        target = torch.arange(logits.size(0), device=ctx.device)

        weights: torch.Tensor | None = slot.get(K.SLOT_WEIGHTS)
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
