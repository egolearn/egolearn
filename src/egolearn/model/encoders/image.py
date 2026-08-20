"""Image encoders: standard architectures plus a toy one for tests.

Standard architectures are generic tools rather than method contributions,
so they live in core.  Encoders tied to one method (e.g. BabyCL's
bag-of-words text embedding) live in that method's package and register
into the same registry.
"""

from __future__ import annotations

import torch.nn as nn

from egolearn.config.registry import ENCODERS


@ENCODERS.register("resnext50")
class ResNeXt50Encoder(nn.Module):
    """torchvision ResNeXt-50 with the final fc layer removed; output is the
    2048-wide pooled feature.

    Both founding reproductions train FROM SCRATCH (pretrained=False):
    BabyCL passes --no_pretrained_cnn; Memory Storyboard builds its ResNet
    from scratch.
    """

    modality = "image"
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


@ENCODERS.register("resnet18")
class ResNet18Encoder(nn.Module):
    """Small encoder for cheap experiments; 512-wide pooled feature."""

    modality = "image"
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


@ENCODERS.register("toy_cnn")
class ToyCNNEncoder(nn.Module):
    """Tiny 3-block convnet for tests and examples (no weight downloads)."""

    modality = "image"

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
