"""Video encoders: clip in, representation out.

Contract (see base.py): forward takes a clip [B, T, C, H, W] and returns a
pooled [B, out_dim] tensor or an EncoderOutput carrying tokens too.  The
loading side already delivers clips (recipe unit "clip", temporally
consistent augmentation); a video encoder is what consumes them.

`video_vit` is the real architecture (V-JEPA 2's encoder design);
`toy_video` exists so the video path is exercised by tests and so a
composition can be tried end to end without weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egolearn.config.registry import ENCODERS
from egolearn.model.encoders.base import EncoderOutput
from egolearn.model.encoders.image import ToyCNNEncoder
from egolearn.model.transformer import Block, rope_3d


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


@ENCODERS.register("video_vit")
class VideoViTEncoder(nn.Module):
    """Video ViT the way V-JEPA 2 builds one: tubelet embedding (a 3D conv
    cuts the clip into tubelet x patch x patch pieces, each becoming one
    token) and pre-norm transformer blocks with 3D rotary position
    embeddings.  Position enters as a rotation instead of a learned table,
    so the same weights accept any clip length/resolution whose token grid
    divides evenly.

    Output: EncoderOutput with
      tokens  [B, N, dim] in t-major grid order (N = T/tubelet * H/patch * W/patch)
      pooled  mean over tokens
      aux     {"grid": tensor([T', H', W'])} -- the token grid; mask
              samplers (model/masking.py) need it

    `visible` ([B, K] flat token indices) restricts the forward to those
    tokens: all patches are embedded, then only the visible ones go through
    attention -- masked prediction's context path at context-sized cost,
    with every kept token still at its true (t, h, w) position.

    Reference sizes (V-JEPA 2 pretraining): ViT-L/16 = dim 1024, depth 24,
    heads 16, patch 16, tubelet 2.  Defaults are deliberately small.
    """

    modality = "video"

    def __init__(
        self,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        patch: int = 16,
        tubelet: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.out_dim = int(dim)
        self.heads = int(heads)
        self.patch = int(patch)
        self.tubelet = int(tubelet)
        self.embed = nn.Conv3d(
            3, dim, kernel_size=(tubelet, patch, patch), stride=(tubelet, patch, patch)
        )
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, visible: torch.Tensor | None = None) -> EncoderOutput:
        b, t, _, h, w = x.shape
        if t % self.tubelet or h % self.patch or w % self.patch:
            raise ValueError(
                f"clip [T={t}, H={h}, W={w}] is not divisible by "
                f"tubelet {self.tubelet} / patch {self.patch}"
            )
        z = self.embed(x.transpose(1, 2))  # [B, dim, T', H', W']
        grid = (z.size(2), z.size(3), z.size(4))
        tokens = z.flatten(2).transpose(1, 2)  # [B, N, dim], t-major order
        if visible is None:
            pos = torch.arange(tokens.size(1), device=x.device).expand(b, -1)
        else:
            pos = visible
            tokens = torch.gather(tokens, 1, visible.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        cos, sin = rope_3d(pos, grid, self.out_dim // self.heads)
        for blk in self.blocks:
            tokens = blk(tokens, cos, sin)
        tokens = self.norm(tokens)
        return EncoderOutput(
            pooled=tokens.mean(dim=1),
            tokens=tokens,
            aux={"grid": torch.tensor(grid, device=x.device)},
        )
