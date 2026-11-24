"""Supervised contrastive learning over two views and global segment labels.

Each view is an anchor. All other views with the same label are positives,
including the second view of the same frame. The denominator includes every
non-self view in the batch. Log probabilities are averaged over positives,
then anchors; labels need not be contiguous or bounded by a class count.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES
from egolearn.core import keys as K
from egolearn.model.losses.base import Loss


def _supcon(
    z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if z1.ndim != 2 or z1.shape != z2.shape or z1.size(0) == 0:
        raise ValueError("supcon needs matching nonempty [B, D] projections")
    if labels.ndim != 1 or labels.size(0) != z1.size(0):
        raise ValueError("supcon needs one label per frame")
    # Casting alone would still let an outer autocast lower the matmul precision.
    with torch.autocast(device_type=z1.device.type, enabled=False):
        z = torch.cat([z1, z2], dim=0)
        if z.dtype in (torch.float16, torch.bfloat16):
            z = z.float()
        z = F.normalize(z, dim=1)
        labels = labels.to(z.device).repeat(2)
        self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        positives = labels[:, None].eq(labels[None, :]) & ~self_mask
        logits = (z @ z.t()) / temperature
        log_probs = F.log_softmax(logits.masked_fill(self_mask, -torch.inf), dim=1)
        # The paired view guarantees at least one positive for each anchor.
        return -(log_probs.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)).mean()


@LOSSES.register("supcon")
class SupervisedContrastive(Loss):
    """Contrast frames using labels supplied by the batch's selected slots."""

    def __init__(
        self,
        slots: list[str],
        head: str,
        temperature: float = 0.2,
        name: str = "temporal",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        if not slots:
            raise ValueError("supcon needs at least one slot")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("supcon temperature must be positive and finite")
        self.slots = list(slots)
        self.head = head
        self.temperature = float(temperature)
        self.requires_slots = self.slots
        self.requires_heads = [head]

    def compute(self, batch, model, ctx) -> dict[str, torch.Tensor]:
        slots = [self._slot(batch, s) for s in self.slots]
        for name, slot in zip(self.slots, slots, strict=True):
            if K.SLOT_LABELS not in slot:
                raise KeyError(f"supcon: slot {name!r} carries no labels; set labels_from")
            values = slot[K.SLOT_FEATS if K.SLOT_FEATS in slot else K.SLOT_VIEWS]
            if values.size(1) < 2:
                raise ValueError(f"supcon needs >=2 views in slot {name!r}")
            if slot[K.SLOT_LABELS].shape != (values.size(0),):
                raise ValueError(f"supcon needs one label per frame in slot {name!r}")
        first, second, labels = [], [], []
        for slot in slots:
            if K.SLOT_FEATS in slot:
                feats = slot[K.SLOT_FEATS].to(ctx.device, non_blocking=True)
                f1, f2 = feats[:, 0], feats[:, 1]
            else:
                # Match precompute_feats: one encoder call per slot and view.
                # Combining raw slots would change BatchNorm statistics.
                views = slot[K.SLOT_VIEWS].to(ctx.device, non_blocking=True)
                modality = slot.get(K.SLOT_MODALITY, "image")
                f1 = model.encode(modality, views[:, 0].contiguous()).pooled
                f2 = model.encode(modality, views[:, 1].contiguous()).pooled
            first.append(f1)
            second.append(f2)
            labels.append(slot[K.SLOT_LABELS])
        z1 = model.apply_head(self.head, torch.cat(first, dim=0))
        z2 = model.apply_head(self.head, torch.cat(second, dim=0))
        targets = torch.cat(labels, dim=0).to(ctx.device)
        return {"supcon": _supcon(z1, z2, targets, self.temperature)}
