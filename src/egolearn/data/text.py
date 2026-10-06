"""Tokenizer for the CVCL/BabyCL text side: turns utterance strings into
padded id tensors using a fixed word-level vocabulary.

Provenance: matches BabyCL's ms_dataloader tokenization -- vocab.json lookup,
optional <sos>/<eos> wrapping, and the SAYCam-vocab normalizer (lowercase,
split punctuation off words, split contractions).  Normalization is OFF for
SAYCam (its transcripts are already normalized) and ON for BabyView.
"""

from __future__ import annotations

import json
import re

import torch


def normalize_utterance(s: str) -> str:
    """Rewrite free text into the token forms the SAYCam vocab uses
    (lowercase, punctuation and contractions split off).  Running it twice
    gives the same result as running it once."""
    if not s:
        return s
    s = s.lower()
    s = re.sub(r'([.,!?;:"()])', r" \1 ", s)
    s = re.sub(r"'", " '", s)
    return s


class VocabTokenizer:
    def __init__(self, vocab: dict[str, int], add_sos_eos: bool = True, normalize: bool = False):
        self.vocab = vocab
        self.add_sos_eos = add_sos_eos
        self.normalize = normalize
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)
        self.sos_id = vocab.get("<sos>", vocab.get("<SOS>"))
        self.eos_id = vocab.get("<eos>", vocab.get("<EOS>"))

    @classmethod
    def from_json(cls, path: str, **kw) -> VocabTokenizer:
        from egolearn.hub import resolve

        with resolve(path).open() as f:
            return cls(json.load(f), **kw)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        if self.normalize:
            text = normalize_utterance(text)
        toks = [self.vocab.get(t, self.unk_id) for t in (text or "").strip().split()]
        if not toks:
            toks = [self.unk_id]
        if self.add_sos_eos and self.sos_id is not None and self.eos_id is not None:
            toks = [self.sos_id, *toks, self.eos_id]
        return toks

    def batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of utterances and pad them to equal length.
        Returns (ids [B,T], lengths [B])."""
        seqs = [self.encode(t) for t in texts]
        lens = [len(s) for s in seqs]
        T = max(lens) if lens else 1
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return ids, torch.tensor(lens, dtype=torch.long)
