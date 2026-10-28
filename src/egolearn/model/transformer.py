# Copyright (c) Meta Platforms, Inc. and affiliates.
# V-JEPA 2 frequency expansion adapted from facebookresearch/vjepa2 (MIT).
# See egolearn/methods/vjepa2/LICENSE for the reference license.
"""Transformer pieces shared by the video ViT encoder and the V-JEPA
predictor: pre-norm attention/MLP blocks with 3D rotary position embeddings.

Rotary embeddings (RoPE) encode a token's (t, h, w) grid position by rotating
pairs of feature channels inside each attention head, following the V-JEPA 2
encoder/predictor design.  The head dimension is split into three equal
channel groups, one per axis; each group is rotated by angles proportional to
that axis' coordinate (channels left over after the three-way split are not
rotated).  Because position enters as a rotation instead of a learned table,
the same weights accept any clip length or resolution whose token grid fits,
and a token keeps its true position when the tokens around it are masked
away -- the property masked prediction relies on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_3d(
    positions: torch.Tensor,
    grid: tuple[int, int, int],
    head_dim: int,
    theta: float = 10000.0,
    convention: str = "paired",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation tables for tokens at `positions`.

    positions: [B, K] flat indices into the t-major (T', H', W') token grid.
    Returns (cos, sin), broadcast over heads: [B, 1, K, head_dim // 2]
    for paired rotation or [B, 1, K, head_dim] for the V-JEPA 2 convention.
    """
    if convention not in ("paired", "vjepa2"):
        raise ValueError(f"unknown RoPE convention {convention!r}")
    if head_dim % 2:
        raise ValueError("RoPE head_dim must be even")
    _, hh, ww = grid
    axis = (head_dim // 3) // 2 * 2  # even number of channels per axis
    if axis < 2:
        raise ValueError(f"head_dim {head_dim} is too small for 3D rotary embeddings (need >= 6)")
    inv = 1.0 / theta ** (
        torch.arange(0, axis, 2, device=positions.device, dtype=torch.float32) / axis
    )
    coords = (positions // (hh * ww), (positions % (hh * ww)) // ww, positions % ww)
    ang = torch.cat([c.unsqueeze(-1).float() * inv for c in coords], dim=-1)  # [B, K, 3*axis/2]
    if convention == "vjepa2":
        # Meta uses repeated half-frequency vectors within each axis, rather
        # than repeat_interleave. Preserve this checkpoint convention explicitly.
        # Adapted from facebookresearch/vjepa2 (MIT; methods/vjepa2/LICENSE).
        ang = torch.cat([a.repeat(1, 1, 2) for a in ang.split(axis // 2, dim=-1)], dim=-1)
    width = head_dim if convention == "vjepa2" else head_dim // 2
    pad = width - ang.size(-1)
    if pad:
        ang = torch.cat([ang, ang.new_zeros(*ang.shape[:-1], pad)], dim=-1)  # unrotated channels
    return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive channel pairs of [B, heads, K, head_dim]."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    if cos.size(-1) == x.size(-1):
        pair = x.unflatten(-1, (-1, 2))
        turned = torch.stack((-pair[..., 1], pair[..., 0]), dim=-1).flatten(-2)
        return x * cos + turned * sin
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, k, d = x.shape
        q, k_, v = self.qkv(x).reshape(b, k, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k_ = _rotate(q, cos, sin), _rotate(k_, cos, sin)
        out = F.scaled_dot_product_attention(q, k_, v)
        return self.proj(out.transpose(1, 2).reshape(b, k, d))


class Block(nn.Module):
    """Standard pre-norm ViT block (LayerNorm -> attention, LayerNorm -> GELU
    MLP, residuals), position handed in as rotation tables."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, norm_eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))
