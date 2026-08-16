"""BabyCL's method-tied model component: the CVCL BOW text encoder.

(clip_infonce is a textbook loss and lives in the core loss library;
what is method-tied here is the specific text encoder choice.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egolearn.config.registry import TRUNKS


@TRUNKS.register("text_bow")
class TextBowTrunk(nn.Module):
    """CVCL text encoder: mean of learned word embeddings.

    Tap: text.embed [B, dim].  Matches multimodel.TextEncoder (flat/embedding
    path): sum(embeddings) / length, padding_idx = 0.
    """

    tap = "text.embed"

    def __init__(self, vocab_size: int, dim: int = 512, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.out_dim = dim

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)  # [B, T, E]
        return emb.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).to(emb.dtype)
