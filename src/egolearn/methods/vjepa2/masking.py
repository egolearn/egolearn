# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from facebookresearch/vjepa2 at 204698b (MIT; see LICENSE).
"""Full-duration multiblock masks with reference block-size/truncation semantics."""

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
    tt, hh, ww = grid
    if min(grid) < 1 or hh * ww < 2 or n < 1 or num_blocks < 1:
        raise ValueError("masking requires a nonempty batch and at least two spatial tokens")
    if not 0 < scale[0] <= scale[1] < 1 or not 0 < aspect[0] <= aspect[1]:
        raise ValueError("invalid mask scale/aspect range")
    area = int(hh * ww * rng.uniform(*scale))
    ratio = rng.uniform(*aspect)
    bh = min(hh, round(math.sqrt(area * ratio)))
    bw = min(ww, round(math.sqrt(area / ratio)))
    if min(bh, bw) < 1 or bh * bw == hh * ww:
        raise ValueError(
            "mask block is empty or covers the grid; increase crop size or change masks"
        )
    contexts, targets = [], []
    for _ in range(n):
        for _attempt in range(1000):
            masked: set[int] = set()
            for _block in range(num_blocks):
                top, left = rng.randrange(hh - bh + 1), rng.randrange(ww - bw + 1)
                masked.update(
                    i * ww + j for i in range(top, top + bh) for j in range(left, left + bw)
                )
            if len(masked) < hh * ww:
                break
        else:
            raise ValueError("could not sample nonempty context after 1000 redraws")
        ctx = [i for i in range(hh * ww) if i not in masked]
        contexts.append([t * hh * ww + i for t in range(tt) for i in ctx])
        targets.append([t * hh * ww + i for t in range(tt) for i in sorted(masked)])
    kc, kt = min(map(len, contexts)), min(map(len, targets))
    return (
        torch.tensor([row[:kc] for row in contexts], dtype=torch.long),
        torch.tensor([row[:kt] for row in targets], dtype=torch.long),
    )
