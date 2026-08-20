"""Predictors: representation(s) in, predicted representation out.

This is the component JEPA-style training adds on top of encoders: given
context representations (and optionally conditioning from another modality),
predict a target representation.  Losses then compare the prediction to the
target encoder's output.

Contract: an nn.Module whose forward takes one or more representation
tensors positionally, plus free keyword arguments (real V-JEPA predictors
take token masks; the interface leaves room for that).  Declared in the
config under `model.predictors:` with the encoders it reads:

    predictors:
      pred: {inputs: [video], _target_: "predictor:mlp", sizes: [64, 128, 64]}

Language-conditioned prediction is the same shape with two inputs:
`inputs: [video, text]`.  Which form of each encoder's output the caller
feeds in (pooled vector, token sequence) is chosen at the predict() call.

How the V-JEPA pattern maps onto this (interface check, not yet built):
the context encoder is a normal `video` encoder; the EMA target encoder is
a second encoder entry marked `frozen: true` (its weights are excluded from
the optimizer but included in snapshots); the loss updates the EMA copy in
its `on_step_end` hook and compares `predict(...)` to the frozen encoder's
output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egolearn.config.registry import PREDICTORS
from egolearn.model.heads import build_mlp


@PREDICTORS.register("mlp")
class MLPPredictor(nn.Module):
    """Concatenates its input representations and runs an MLP.  A minimal
    predictor: enough to exercise the interface and to serve as the toy
    stand-in in tests."""

    def __init__(self, sizes: list[int]):
        super().__init__()
        self.net = build_mlp(sizes)

    def forward(self, *reps: torch.Tensor, **_) -> torch.Tensor:
        x = torch.cat(reps, dim=-1) if len(reps) > 1 else reps[0]
        return self.net(x)
