"""Heads: small modules (projectors, classifiers) that sit on top of an
encoder's pooled output.  Heads name their encoder in the model config; the
encoder itself stays head-agnostic."""

from __future__ import annotations

import torch.nn as nn

from ..config.registry import HEADS


@HEADS.register("linear")
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.fc(x)


def build_mlp(sizes: list[int]) -> nn.Sequential:
    """Linear-ReLU-...-Linear from a size list; shared by MLPHead and
    MLPPredictor so the recipe can only ever change in one place."""
    if len(sizes) < 2:
        raise ValueError(f"an MLP needs at least [in, out] sizes, got {sizes}")
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


@HEADS.register("mlp")
class MLPHead(nn.Module):
    """Multi-layer perceptron: Linear-ReLU-...-Linear.  Used as the
    SimCLR-style projector and as the TCL classifier.

    e.g. sizes [2048, 2048, 128] (BabyCL / Memory Storyboard projector),
         sizes [2048, 2048, 2000] (TCL classifier).
    """

    def __init__(self, sizes: list[int]):
        super().__init__()
        self.net = build_mlp(sizes)

    def forward(self, x):
        return self.net(x)
