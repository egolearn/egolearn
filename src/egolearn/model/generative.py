"""Generative / autoregressive components: interface reserved, nothing built.

When a generative or autoregressive model enters the framework (per the
model-design review), it registers here and the composed model gains the
"generation" capability, which generative evaluations can then require.
The surface is deliberately small:

    generate(context, **kw) -> Tensor   produce continuations/samples
    score(context, target, **kw) -> Tensor   log-likelihood of a target

Nothing in the framework calls these yet; the registry kind and capability
name exist now so a future component slots in without touching the composed
model or the eval-compatibility check.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GenerativeComponent(nn.Module):  # pragma: no cover - interface only
    def generate(self, context: torch.Tensor, **kw) -> torch.Tensor:
        raise NotImplementedError

    def score(self, context: torch.Tensor, target: torch.Tensor, **kw) -> torch.Tensor:
        raise NotImplementedError
