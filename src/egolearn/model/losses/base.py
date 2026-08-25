"""Loss protocol: the interface every training loss implements.

A loss reads the slots it declared (slots are the named groups of items
inside one batch) and the model's encoders/heads, and returns named scalar
values:

    values = loss.compute(batch, model, ctx)   # {name: scalar tensor}

`requires_slots` / `requires_heads` are checked at build time (see
pipeline.validate) and again against the first batch, so a bad config fails
early.  Losses that need training-loop state (EMA teachers, momentum queues)
implement `on_step_end` -- the single lifecycle hook agreed in design
review; anything that cannot fit that hook belongs in a project-side Driver
subclass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class Loss:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = float(weight)

    # what this loss needs from the batch and model; checked at build time
    # (subclasses assign these per-instance in __init__)
    requires_slots: Sequence[str] = ()
    requires_heads: Sequence[str] = ()
    requires_text: bool = False
    requires_encoders: Sequence[str] = ()  # encoder names read via model.encode
    requires_frozen_encoders: Sequence[str] = ()  # must also be `frozen: true` (EMA targets)
    requires_predictors: Sequence[str] = ()
    # True: this loss runs its own encoder forwards (masked prediction and the
    # like), so the shared per-slot pooled precompute would be wasted work for
    # its slots -- see own_forward_slots below
    own_forward: bool = False
    # True: this loss changes what the composer would sample BETWEEN steps
    # (e.g. per-step replay priorities).  The driver contract says sampling
    # state -- memory contents, labels -- only changes when the stream
    # advances; a loss that breaks that declares it here, and the build then
    # refuses `driver.prefetch` (batches could not be composed ahead).
    mutates_sampling_state: bool = False

    def compute(self, batch: dict, model, ctx) -> dict[str, torch.Tensor]:
        raise NotImplementedError  # pragma: no cover

    def on_step_end(self, model, ctx) -> None:
        """Optional lifecycle hook (EMA updates etc.). Default: no-op."""

    def _slot(self, batch: dict, name: str) -> dict:
        if name not in batch:
            raise KeyError(
                f"loss {self.name!r} requires slot {name!r} but the "
                f"composed batch has {sorted(batch)} (strict slot fell back?)"
            )
        return batch[name]


def own_forward_slots(losses: list) -> frozenset[str]:
    """Slots whose every declared consumer runs its own encoder forwards:
    the drivers hand exactly these to `precompute_feats(skip=...)`, so the
    shared pooled forward is not computed on top of the loss's own.  A slot
    that a standard loss also reads keeps its precomputed feats."""
    own = {s for fn in losses if fn.own_forward for s in fn.requires_slots}
    shared = {s for fn in losses if not fn.own_forward for s in fn.requires_slots}
    return frozenset(own - shared)
