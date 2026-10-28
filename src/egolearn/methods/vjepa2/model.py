# Copyright (c) Meta Platforms, Inc. and affiliates.
# Reference initialization adapted from facebookresearch/vjepa2 at 204698b (MIT).
# See LICENSE in this directory. EgoLearn adapters reuse the shared video modules.
"""Explicit V-JEPA 2 encoder/predictor presets; legacy defaults stay unchanged."""

from __future__ import annotations

import math

import torch
from torch import nn

from egolearn.config.registry import ENCODERS, PREDICTORS
from egolearn.model.encoders.video import VideoViTEncoder
from egolearn.model.predictors import VJEPAPredictor
from egolearn.model.transformer import Block


def initialize(module: VideoViTEncoder | VJEPAPredictor) -> None:
    """Reference ViT initialization and depth-dependent residual scaling."""
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.zeros_(layer.bias)
            nn.init.ones_(layer.weight)
    with torch.no_grad():
        for index, block in enumerate(module.blocks, 1):
            assert isinstance(block, Block)
            block.attn.proj.weight.div_(math.sqrt(2 * index))
            output = block.mlp[2]
            assert isinstance(output, nn.Linear)
            output.weight.div_(math.sqrt(2 * index))


@ENCODERS.register("vjepa2")
class VJEPA2Encoder(VideoViTEncoder):
    def __init__(
        self,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        patch: int = 16,
        tubelet: int = 2,
        mlp_ratio: float = 4.0,
    ):
        if min(dim, depth, heads, patch, tubelet) < 1 or dim % heads or dim // heads % 2:
            raise ValueError("V-JEPA 2 needs positive dimensions and even integral head width")
        if dim // heads < 6:
            raise ValueError("V-JEPA 2 head width must be at least 6")
        super().__init__(
            dim, depth, heads, patch, tubelet, mlp_ratio, norm_eps=1e-6, rope_convention="vjepa2"
        )
        initialize(self)


@PREDICTORS.register("vjepa2")
class VJEPA2Predictor(VJEPAPredictor):
    """Single clip-length group: only mask token 0 is trained and evaluated.

    Two tokens retain the reference architecture. The second is reserved for a
    future second clip-length group and must never be used for scoring this model.
    """

    def __init__(
        self,
        encoder_dim: int,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        trained_mask_indices: tuple[int, ...] = (0,),
    ):
        if tuple(trained_mask_indices) != (0,):
            raise ValueError("this single-group recipe supports trained_mask_indices=[0] only")
        if min(encoder_dim, dim, depth, heads) < 1 or dim % heads or dim // heads % 2:
            raise ValueError("V-JEPA 2 needs positive dimensions and even integral head width")
        if dim // heads < 6:
            raise ValueError("V-JEPA 2 head width must be at least 6")
        super().__init__(
            encoder_dim,
            dim,
            depth,
            heads,
            mlp_ratio,
            norm_eps=1e-6,
            rope_convention="vjepa2",
            num_mask_tokens=2,
            zero_init_mask_tokens=True,
        )
        initialize(self)

    def forward(self, tokens: torch.Tensor, *, mask_index: int = 0, **kwargs) -> torch.Tensor:
        if mask_index != 0:
            raise ValueError("mask_index must be 0: other tokens were not trained")
        return super().forward(tokens, mask_index=mask_index, **kwargs)
