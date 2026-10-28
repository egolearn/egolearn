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

`predictor:vjepa` is that pattern built out for V-JEPA: the context encoder
is a normal `video` encoder, the EMA target encoder is a second encoder
entry marked `frozen: true` (excluded from the optimizer, included in
snapshots), and `loss:vjepa` compares `predict(...)` against the frozen
encoder's tokens (see model/losses/vjepa.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egolearn.config.registry import PREDICTORS
from egolearn.model.heads import build_mlp
from egolearn.model.transformer import Block, rope_3d


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


@PREDICTORS.register("vjepa")
class VJEPAPredictor(nn.Module):
    """V-JEPA's predictor: a narrow transformer that maps the context
    encoder's visible tokens to predicted target tokens at the masked
    positions.  One learned mask token stands in at every target position
    (its position given by the same 3D rotary embeddings the encoder uses);
    self-attention runs over [context tokens; mask tokens] and the mask-token
    outputs are projected back to the encoder width.

    forward(tokens, ctx_pos=, tgt_pos=, grid=) -> [B, Kt, encoder_dim]
      tokens   [B, Kc, encoder_dim]  context encoder output
      ctx_pos  [B, Kc]  flat grid positions of those tokens
      tgt_pos  [B, Kt]  flat grid positions to predict
      grid     (T', H', W')  the token grid both index into

    Reference size (V-JEPA 2): dim 384, depth 12.  Defaults are small.
    """

    def __init__(
        self,
        encoder_dim: int,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-5,
        rope_convention: str = "paired",
        num_mask_tokens: int = 1,
        zero_init_mask_tokens: bool = False,
    ):
        super().__init__()
        if num_mask_tokens < 1:
            raise ValueError("num_mask_tokens must be positive")
        self.rope_convention = rope_convention
        self.heads = int(heads)
        self.dim = int(dim)
        self.embed = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(num_mask_tokens, 1, dim))
        if not zero_init_mask_tokens:
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, norm_eps) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, encoder_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        ctx_pos: torch.Tensor,
        tgt_pos: torch.Tensor,
        grid: tuple[int, int, int],
        mask_index: int = 0,
        **_,
    ) -> torch.Tensor:
        if not 0 <= mask_index < self.mask_token.size(0):
            raise ValueError("mask_index is out of range")
        b, kc, _d = tokens.shape
        kt = tgt_pos.size(1)
        x = torch.cat(
            [self.embed(tokens), self.mask_token[mask_index : mask_index + 1].expand(b, kt, -1)],
            dim=1,
        )
        cos, sin = rope_3d(
            torch.cat([ctx_pos, tgt_pos], dim=1),
            grid,
            self.dim // self.heads,
            convention=self.rope_convention,
        )
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.proj(self.norm(x[:, kc:]))
