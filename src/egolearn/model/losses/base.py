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
