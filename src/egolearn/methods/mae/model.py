"""Image MAE: visible-patch ViT encoder and pixel reconstruction decoder.

The encoder returns patch tokens separately from CLS, following EncoderOutput.
Positions are fixed 2D sine/cosine embeddings. Both transformers use pre-norm
attention, GELU MLPs, and no dropout. Full-image encoding needs no decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from egolearn.config.registry import ENCODERS, PREDICTORS
from egolearn.model.encoders.base import EncoderOutput


def _grid_size(image_size: int, patch_size: int) -> int:
    if image_size < 1 or patch_size < 1 or image_size % patch_size:
        raise ValueError("image_size must be positive and divisible by a positive patch_size")
    return image_size // patch_size


def _positions(side: int, dim: int) -> torch.Tensor:
    if dim < 4 or dim % 4:
        raise ValueError("MAE embedding dimension must be a positive multiple of four")
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    frequencies = 10000.0 ** (-torch.arange(dim // 4, dtype=torch.float64) / (dim // 4))
    angles = [coordinate.flatten()[:, None] * frequencies for coordinate in (x, y)]
    patches = torch.cat([term for angle in angles for term in (angle.sin(), angle.cos())], -1)
    # CLS has a zero positional embedding.
    return torch.cat([torch.zeros(1, dim), patches.float()], 0).unsqueeze(0)


def _transformer(dim: int, depth: int, heads: int, mlp_ratio: float) -> nn.Sequential:
    if depth < 1 or heads < 1 or dim % heads or mlp_ratio <= 0 or int(dim * mlp_ratio) < 1:
        raise ValueError("MAE needs positive depth, heads and MLP width; heads must divide dim")
    return nn.Sequential(
        *[
            nn.TransformerEncoderLayer(
                dim,
                heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0,
                activation="gelu",
                layer_norm_eps=1e-6,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ]
    )


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        assert module.in_proj_weight is not None and module.in_proj_bias is not None
        nn.init.xavier_uniform_(module.in_proj_weight)
        nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


@ENCODERS.register("mae_vit")
class MAEEncoder(nn.Module):
    """ViT-B/16 by default; configure dim=384, heads=6 for ViT-S/16.

    ``visible`` contains distinct patch indices [B, K] in the order the decoder
    receives them. Omitting it encodes all patches, for downstream evaluation.
    ``pooled`` is CLS by default; ``pooling=mean`` averages normalized patches.
    This pooling option is for frozen features, not a full fine-tuning recipe.
    """

    modality = "image"
    pos_embed: torch.Tensor

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        pooling: str = "cls",
    ):
        super().__init__()
        self.grid_size = _grid_size(image_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = self.grid_size**2
        self.out_dim = dim
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.pooling = pooling
        self.register_buffer("pos_embed", _positions(self.grid_size, dim))
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.blocks = _transformer(dim, depth, heads, mlp_ratio)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(_initialize)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        assert self.patch_embed.bias is not None
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self, images: torch.Tensor, *, visible: torch.Tensor | None = None
    ) -> EncoderOutput:
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"MAE expects RGB images [B, 3, {self.image_size}, {self.image_size}]")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches + self.pos_embed[:, 1:]
        if visible is not None:
            if visible.ndim != 2 or visible.size(0) != images.size(0) or visible.size(1) < 1:
                raise ValueError("visible must have shape [B, K] with at least one patch")
            patches = patches.gather(1, visible.unsqueeze(-1).expand(-1, -1, self.out_dim))
        cls = self.cls_token.expand(images.size(0), -1, -1)
        encoded = self.norm(self.blocks(torch.cat([cls, patches], dim=1)))
        tokens = encoded[:, 1:]
        pooled = encoded[:, 0] if self.pooling == "cls" else tokens.mean(dim=1)
        spatial = None
        if visible is None:
            spatial = tokens.transpose(1, 2).reshape(
                images.size(0), self.out_dim, self.grid_size, self.grid_size
            )
        return EncoderOutput(
            pooled=pooled, tokens=tokens, spatial=spatial, aux={"cls_token": encoded[:, :1]}
        )


@PREDICTORS.register("mae")
class MAEDecoder(nn.Module):
    """Restore the patch grid with learned mask tokens and predict RGB pixels.

    ``visible`` uses the same indices/order as the encoder's visible forward.
    Predictions are row-major patches, with pixels ordered (row, column, RGB).
    """

    pos_embed: torch.Tensor

    def __init__(
        self,
        encoder_dim: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 512,
        depth: int = 8,
        heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_patches = _grid_size(image_size, patch_size) ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.register_buffer("pos_embed", _positions(image_size // patch_size, dim))
        self.projection = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, dim))
        self.blocks = _transformer(dim, depth, heads, mlp_ratio)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pixels = nn.Linear(dim, patch_size**2 * 3)
        self.apply(_initialize)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, tokens: torch.Tensor, *, cls_token: torch.Tensor, visible: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3 or visible.shape != tokens.shape[:2]:
            raise ValueError("visible indices must match the encoder's [B, K] patch tokens")
        if cls_token.shape != (tokens.size(0), 1, tokens.size(2)):
            raise ValueError("cls_token must have shape [B, 1, encoder_dim]")
        encoded = self.projection(torch.cat([cls_token, tokens], dim=1))
        patches = self.mask_token.expand(tokens.size(0), self.num_patches, -1)
        patches = patches.scatter(
            1,
            visible.unsqueeze(-1).expand(-1, -1, encoded.size(-1)),
            encoded[:, 1:].to(patches.dtype),
        )
        sequence = torch.cat([encoded[:, :1], patches], dim=1)
        sequence = sequence + self.pos_embed
        return self.pixels(self.norm(self.blocks(sequence))[:, 1:])
