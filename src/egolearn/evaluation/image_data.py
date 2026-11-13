"""Load labeled images with optional per-sample augmentation seeds."""

from __future__ import annotations

import random
from collections.abc import Sequence
from contextlib import contextmanager

import torch
from PIL import Image
from torch.utils.data import Dataset

from egolearn.evaluation.linear_probe import LabeledImage


class ImageDataset(Dataset):
    def __init__(self, items: Sequence[LabeledImage], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int | tuple[int, int]) -> tuple[torch.Tensor, int]:
        seed = None
        if isinstance(index, tuple):
            index, seed = index
        item = self.items[index]
        with Image.open(item.image) as image:
            converted = image.convert("RGB")
            if seed is None:
                tensor = self.transform(converted)
            else:
                with isolated_image_rng(seed):
                    tensor = self.transform(converted)
        return tensor, item.label


@contextmanager
def isolated_image_rng(seed: int):
    """Make stochastic decoding repeatable without advancing the training RNG."""
    python_state = random.getstate()
    cpu_state = torch.Generator().manual_seed(int(seed)).get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(cpu_state)
            random.seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
