"""What every encoder promises the rest of the framework.

An encoder turns one modality's raw tensor into a representation.  The
framework never asks which class an encoder is; it asks two questions,
answered by class attributes:

  modality  which kind of input it eats and, therefore, which capability
            the composed model gains by carrying it:
              "image"  forward(x[B, C, H, W])
              "video"  forward(x[B, T, C, H, W])
              "text"   forward(ids[B, L], lengths[B])
  out_dim   the width of the pooled representation, so heads/predictors
            can be sized.

Everything else addresses an encoder by the name the config gave it under
`model.encoders:` -- heads and predictors declare `encoder: vision`, and
which form of the output they consume is an EncoderOutput field name.
`model.encode()` passes extra keyword arguments through, so an encoder may
accept optional ones (the video ViT's `visible=` token indices for masked
prediction) without that widening anyone else's signature.

The output is wrapped.  `forward` may return either a plain [B, out_dim]
tensor (most encoders; it is treated as the pooled field) or an
EncoderOutput carrying more than one form of the same representation:

    pooled   [B, out_dim]      always available; what heads consume
    tokens   [B, N, D] or None patch/token sequence (ViT), for token-space
                               prediction and masking
    spatial  [B, C, H', W']    feature map before pooling, for dense heads
             or None           and attribution maps (GradCAM)
    aux      dict              anything else (a CLIP logit scale, masks, ...)

The point of the wrapper is that adding a field is additive: an encoder
that starts returning tokens changes no signature and breaks no existing
consumer, because consumers name the field they read.  `model.encode()`
always hands back an EncoderOutput (wrapping a bare tensor if needed).

There is no base class to inherit -- any nn.Module with the two
attributes works.  This file holds the contract and the output type;
concrete encoders live in image.py / video.py here, method-specific ones
(e.g. BabyCL's bag-of-words text encoder) in their method package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EncoderOutput:
    """One representation in however many forms the encoder offers."""

    pooled: torch.Tensor  # [B, out_dim]
    tokens: torch.Tensor | None = None  # [B, N, D]
    spatial: torch.Tensor | None = None  # [B, C, H', W']
    aux: dict[str, torch.Tensor] = field(default_factory=dict)

    @staticmethod
    def wrap(out) -> EncoderOutput:
        """Normalize an encoder's return value: a bare tensor becomes the
        pooled field, an EncoderOutput passes through unchanged."""
        if isinstance(out, EncoderOutput):
            return out
        if isinstance(out, torch.Tensor):
            return EncoderOutput(pooled=out)
        raise TypeError(
            f"encoder returned {type(out).__name__}; expected a Tensor or an EncoderOutput"
        )
