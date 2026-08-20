"""Video encoders: clip in, representation out.

Contract (see base.py): forward takes a clip [B, T, C, H, W] and returns
[B, out_dim].  The loading side already delivers clips (recipe unit "clip",
temporally consistent augmentation); a video encoder is what consumes them.

Real video architectures (V-JEPA's ViT, VideoMAE) land here when a project
needs them; `toy_video` exists so the video path is exercised by tests and
so the composition can be tried end to end without weights.
"""

from __future__ import annotations

import torch.nn as nn

from egolearn.config.registry import ENCODERS
from egolearn.model.encoders.image import ToyCNNEncoder


@ENCODERS.register("toy_video")
class ToyVideoEncoder(nn.Module):
    """Per-frame toy CNN, mean-pooled over time.  For tests and interface
    checks only -- it has no temporal modeling."""

    modality = "video"

    def __init__(self, dim: int = 64):
        super().__init__()
        self.out_dim = int(dim)
        self.frame_encoder = ToyCNNEncoder(dim=dim)

    def forward(self, x):  # [B, T, C, H, W] -> [B, D]
        b, t = x.shape[:2]
        feats = self.frame_encoder(x.reshape(b * t, *x.shape[2:]))
        return feats.reshape(b, t, -1).mean(dim=1)
