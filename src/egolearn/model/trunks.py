"""Core trunks: standard architectures (model zoo) + the toy reference trunk.

Architectures are tools, not method contributions -- they live in core.
Method-tied encoders (e.g. BabyCL's BOW text embedding) live in their
projects and register into the same TRUNKS registry.
"""

from __future__ import annotations

import torch.nn as nn

from ..config.registry import TRUNKS


@TRUNKS.register("resnext50")
class ResNeXt50Trunk(nn.Module):
    """torchvision ResNeXt-50; fc removed. Tap: vision.pool2048 [B, 2048].

    Both founding reproductions train FROM SCRATCH (pretrained=False):
    BabyCL passes --no_pretrained_cnn; Memory Storyboard builds its ResNet
    from scratch.
    """

    tap = "vision.pool2048"
    out_dim = 2048

    def __init__(self, pretrained: bool = False):
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnext50_32x4d(weights=weights)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):
        return self.net(x)


@TRUNKS.register("resnet18")
class ResNet18Trunk(nn.Module):
    """Small trunk for cheap experiments. Tap: vision.pool512 [B, 512]."""

    tap = "vision.pool512"
    out_dim = 512

    def __init__(self, pretrained: bool = False):
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet18(weights=weights)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):
        return self.net(x)


@TRUNKS.register("toy_cnn")
class ToyCNNTrunk(nn.Module):
    """3-block convnet for tests/examples (no downloads). Tap: vision.feat."""

    tap = "vision.feat"

    def __init__(self, dim: int = 64):
        super().__init__()
        self.out_dim = int(dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.out_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, x):
        return self.net(x)
