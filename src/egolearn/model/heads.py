"""Projection / classification heads attached to named taps."""

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


@HEADS.register("mlp")
class MLPHead(nn.Module):
    """SimCLR-style projector / TCL classifier: Linear-ReLU-...-Linear.

    e.g. sizes [2048, 2048, 128] (BabyCL/MS projector),
         sizes [2048, 2048, 2000] (TCL classifier).
    """

    def __init__(self, sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
