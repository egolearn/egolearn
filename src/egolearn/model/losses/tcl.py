"""Temporal classification loss (TCL): classify each frame by its segment
label.

Part of the core loss library.  Faithful to
osiris_model.TemporalClassification: cross-entropy of a classifier head
against segment labels on BOTH augmented views, summed and divided by 2B;
`filter_unseen` zeroes the cross-entropy weight of classes beyond the
largest label seen so far.

The `slots` argument (slots are the named groups of items in the batch)
expresses what the original code did with tensor slicing: Memory Storyboard
applies TCL to the replay portion only (slots: [replay]); BabyCL applies it
to the whole visual batch with batch-local labels (slots: [segments], labels
derived by the composer).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES
from egolearn.core import keys as K

from .base import Loss


@LOSSES.register("tcl")
class TemporalClassification(Loss):
    def __init__(
        self,
        slots: list[str],
        head: str,
        num_classes: int,
        filter_unseen: bool = True,
        name: str = "temporal",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.slots = list(slots)
        self.head = head
        self.num_classes = int(num_classes)
        self.filter_unseen = bool(filter_unseen)
        self.max_label_seen = 0
        self.requires_slots = self.slots
        self.requires_heads = [head]

    def compute(self, batch, model, ctx) -> dict[str, torch.Tensor]:
        views_list, labels_list = [], []
        for s in self.slots:
            slot = self._slot(batch, s)
            if "labels" not in slot:
                raise KeyError(
                    f"tcl: slot {s!r} carries no labels -- set labels_from "
                    "(LabelStore) or batch_label in the recipe"
                )
            views_list.append(slot[K.SLOT_VIEWS])
            labels_list.append(slot[K.SLOT_LABELS])
        labels = torch.cat(labels_list, dim=0).to(ctx.device)
        if int(labels.max()) >= self.num_classes:
            raise ValueError(
                f"tcl: label {int(labels.max())} >= num_classes "
                f"{self.num_classes} (grow num_classes or check the label store)"
            )

        self.max_label_seen = max(self.max_label_seen, int(labels.max()))
        ce_weight = None
        if self.filter_unseen:
            ce_weight = torch.ones(self.num_classes, device=ctx.device)
            ce_weight[self.max_label_seen + 1 :] = 0.0

        feats = views = None
        if all(K.SLOT_FEATS in self._slot(batch, s) for s in self.slots):
            # encoder features precomputed by the driver (one forward per slot/view)
            feats = torch.cat([self._slot(batch, s)[K.SLOT_FEATS] for s in self.slots], dim=0)
            n_views = min(2, feats.size(1))
        else:
            # only materialize the raw pixel tensor when it will be consumed
            views = torch.cat(views_list, dim=0)  # [B, V, C, H, W]
            n_views = min(2, views.size(1))
        total: torch.Tensor | float = 0.0
        count = 0
        for v in range(n_views):
            if feats is not None:
                f = feats[:, v]
            else:
                assert views is not None
                x = views[:, v].to(ctx.device, non_blocking=True)
                f = model.encode("image", x).pooled
            logits = model.apply_head(self.head, f)
            total = total + F.cross_entropy(logits, labels, weight=ce_weight, reduction="sum")
            count += labels.size(0)
        assert isinstance(total, torch.Tensor)  # n_views >= 1
        return {"tcl": total / max(1, count)}
