"""NT-Xent (SimCLR) contrastive loss over two augmented views of each image.

Part of the core loss library: a textbook, method-agnostic loss, faithful to
osiris_model.NT_Xent (the implementation both founding repos share).
Methods differ in HOW they combine losses, and that difference lives in
config, not code.

`slots` may list several slots (named groups of items in the batch); their
contents are CONCATENATED before the loss is computed.  Memory Storyboard
computes NT-Xent over current and replay items jointly ([curr, replay]);
BabyCL uses a single segment slot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES

from .base import Loss


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    b = z1.size(0)
    n = 2 * b
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t()

    pos = torch.cat([torch.diag(sim, b), torch.diag(sim, -b)], dim=0).reshape(n, 1)
    mask = torch.ones((n, n), dtype=torch.bool, device=z.device)
    mask.fill_diagonal_(False)
    mask[torch.arange(b), torch.arange(b) + b] = False
    mask[torch.arange(b) + b, torch.arange(b)] = False
    neg = sim[mask].reshape(n, -1)

    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(n, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


@LOSSES.register("ntxent")
class NTXent(Loss):
    def __init__(
        self,
        slots: list[str],
        head: str,
        temperature: float = 0.1,
        name: str = "instance",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.slots = list(slots)
        self.head = head
        self.temperature = float(temperature)
        self.requires_slots = self.slots
        self.requires_heads = [head]

    def compute(self, batch, model, ctx) -> dict[str, torch.Tensor]:
        if all("feats" in self._slot(batch, s) for s in self.slots):
            # trunk features precomputed by the driver (one forward per slot/view)
            f1s, f2s = [], []
            for s in self.slots:
                feats = self._slot(batch, s)["feats"]  # [B, V, D]
                if feats.size(1) < 2:
                    raise ValueError(f"ntxent needs >=2 views in slot {s!r}")
                f1s.append(feats[:, 0])
                f2s.append(feats[:, 1])
            z1 = model.apply_head(self.head, torch.cat(f1s, dim=0))
            z2 = model.apply_head(self.head, torch.cat(f2s, dim=0))
            return {"ntxent": _nt_xent(z1, z2, self.temperature)}
        v1s, v2s = [], []
        for s in self.slots:
            views = self._slot(batch, s)["views"]  # [B, V, C, H, W]
            if views.size(1) < 2:
                raise ValueError(f"ntxent needs >=2 views in slot {s!r}")
            v1s.append(views[:, 0])
            v2s.append(views[:, 1])
        x1 = torch.cat(v1s, dim=0).to(ctx.device, non_blocking=True)
        x2 = torch.cat(v2s, dim=0).to(ctx.device, non_blocking=True)
        z1 = model.apply_head(self.head, model.vision_tap(x1))
        z2 = model.apply_head(self.head, model.vision_tap(x2))
        return {"ntxent": _nt_xent(z1, z2, self.temperature)}
