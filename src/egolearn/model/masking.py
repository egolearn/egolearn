"""Mask samplers for masked video prediction.

A mask is drawn on the TOKEN grid (T', H', W'), not on pixels: `target` is
the set of token positions a predictor must fill in, `context` is the
complement the context encoder is allowed to see.

`multiblock` is V-JEPA's multiblock 3D masking: the union of a few random
rectangles, each covering a fraction of the spatial grid and extended over
every timestep (so a masked region cannot be trivially filled from an
adjacent frame).  V-JEPA draws two mask kinds per step -- many small blocks
(short-range) and few large ones (long-range); that pairing lives in the
loss config, this module just draws one kind.

Each batch item gets its own draw.  Index counts are equalized to the
smallest across the batch so the rows stack into one [B, K] tensor (the
official collator trims the same way); trimming drops trailing indices,
i.e. late-timestep tokens of the largest draws.

Randomness comes from the `random.Random` handed in, nothing global: the
caller derives it from (seed, step), so masks are reproducible and
resume-safe no matter who else draws random numbers.
"""

from __future__ import annotations

import math
import random

import torch


def multiblock(
    grid: tuple[int, int, int],
    n: int,
    num_blocks: int,
    scale: tuple[float, float],
    rng: random.Random,
    aspect: tuple[float, float] = (0.75, 1.5),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (context, target): [n, Kc] and [n, Kt] flat token indices.

    grid: the (T', H', W') token grid.
    num_blocks: rectangles per draw; their union is the target.
    scale: block area as a fraction of the spatial grid, sampled per block.
    aspect: block height/width ratio range, sampled per block.
    """
    tt, hh, ww = grid
    ctx_rows, tgt_rows = [], []
    for _ in range(n):
        masked: set[int] = set()
        for _ in range(num_blocks):
            area = rng.uniform(*scale) * hh * ww
            ar = rng.uniform(*aspect)
            bh = min(hh, max(1, round(math.sqrt(area * ar))))
            bw = min(ww, max(1, round(math.sqrt(area / ar))))
            top, left = rng.randint(0, hh - bh), rng.randint(0, ww - bw)
            masked.update(i * ww + j for i in range(top, top + bh) for j in range(left, left + bw))
        if len(masked) == hh * ww:  # keep at least one context position
            masked.remove(sorted(masked)[rng.randrange(len(masked))])
        tgt = sorted(masked)
        ctx = [p for p in range(hh * ww) if p not in masked]
        # extend over every timestep, t-major flat order
        tgt_rows.append([t * hh * ww + p for t in range(tt) for p in tgt])
        ctx_rows.append([t * hh * ww + p for t in range(tt) for p in ctx])
    kc = min(len(r) for r in ctx_rows)
    kt = min(len(r) for r in tgt_rows)
    context = torch.tensor([r[:kc] for r in ctx_rows], dtype=torch.long)
    target = torch.tensor([r[:kt] for r in tgt_rows], dtype=torch.long)
    return context, target
