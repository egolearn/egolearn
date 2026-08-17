"""The model component specific to the BabyCL method: the CVCL bag-of-words text encoder.

(clip_infonce is a textbook loss and lives in the core loss library; the only
method-specific model choice here is the text encoder.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from egolearn.config.registry import TRUNKS


@TRUNKS.register("text_bow")
class TextBowTrunk(nn.Module):
    """CVCL text encoder: an utterance is encoded as the mean of its learned word
    embeddings.

    Output tap (the named feature other components read): text.embed [B, dim].
    Matches multimodel.TextEncoder (flat/embedding path): sum(embeddings) / length,
    padding_idx = 0.
    """

    tap = "text.embed"

    def __init__(self, vocab_size: int, dim: int = 512, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.out_dim = dim

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)  # [B, T, E]
        return emb.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).to(emb.dtype)
