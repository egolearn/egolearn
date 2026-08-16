"""View pipelines (online augmentation).

A "view" is a registered PIL -> Tensor[C,H,W] pipeline.  Slots declare a list
of view names; the composer stacks them into [V,C,H,W] per item.

Signature note (design review): views receive the whole item dict, not a bare
image, so future geometry-coupled keys (gaze points, boxes) can co-transform
with crops.  Core views only read/write video keys today; the door is open,
the room is not built.

Dataset-level preprocessing (decode fps, orientation flip, BabyView portrait
pre-crop) does NOT belong here -- it belongs to prepare scripts / adapters.
The one exception kept as a view flag is MS's vertical flip, because the
original Memory Storyboard applies it at train time (SAYCam raw frames are
upside-down); BabyCL frames are already flipped at decode time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision.transforms as T
from PIL import ImageOps

from ..config.registry import VIEWS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _finish(size):
    return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


@VIEWS.register("center_crop")
def center_crop(size: int = 224, vertical_flip: bool = False):
    ops = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
    if vertical_flip:
        ops.insert(0, T.Lambda(ImageOps.flip))
    return T.Compose(ops + _finish(size))


@VIEWS.register("simclr_strong")
def simclr_strong(
    size: int = 224,
    view_idx: int = 0,
    scale_min: float = 0.2,
    scale_max: float = 1.0,
    vertical_flip: bool = False,
):
    """Standard SimCLR strong augmentation on the RAW frame (core tool,
    paired with ntxent): view 0 blurs with p=1.0; view 1 blurs with p=0.1
    and solarizes with p=0.2.  `vertical_flip` reproduces Memory
    Storyboard's train-time flip of the upside-down raw SAYCam frames.

    NOTE: BabyCL's effective segment augmentation center-crops to `size`
    BEFORE the random crop (a two-stage artifact of its original pipeline);
    that exact chain is the project view `babycl_ms_aug`, not this one.
    """
    k = max(3, int(0.1 * size) // 2 * 2 + 1)
    ops: list = []
    if vertical_flip:
        ops.append(T.Lambda(ImageOps.flip))
    ops += [
        T.RandomResizedCrop(
            size, scale=(scale_min, scale_max), interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
    ]
    blur_p = 1.0 if view_idx == 0 else 0.1
    ops.append(T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=blur_p))
    if view_idx == 1:
        ops.append(
            T.Lambda(
                lambda img: (
                    ImageOps.solarize(img, threshold=128) if torch.rand(1).item() < 0.2 else img
                )
            )
        )
    return T.Compose(ops + _finish(size))


# Method-tied augmentation recipes (e.g. view:cvcl_aug) live in their project
# and register into the same VIEWS registry.


# ---------------------------------------------------------------------------
# Clip views: [T,C,H,W] float tensor -> [T,C,H,W], TEMPORALLY CONSISTENT.
#
# The mainstream video-SSL convention (VideoMAE / V-JEPA / video contrastive):
# spatial augmentation parameters are sampled ONCE PER CLIP and applied to all
# T frames identically -- otherwise each frame gets its own random crop and
# the clip's temporal structure is destroyed.  torchvision v1 transforms give
# this for free on batched tensors: one __call__ samples parameters once and
# applies them across the leading (time) dimension.
#
# Clip slots must use clip-domain views (convention: names start with clip_);
# image views expect PIL input and will fail loudly on a clip tensor.
# ---------------------------------------------------------------------------


@VIEWS.register("clip_center_crop")
def clip_center_crop(size: int = 224):
    return T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@VIEWS.register("clip_simclr")
def clip_simclr(size: int = 224, view_idx: int = 0, scale_min: float = 0.2, scale_max: float = 1.0):
    """SimCLR-style clip augmentation, one parameter draw per clip."""
    k = max(3, int(0.1 * size) // 2 * 2 + 1)
    blur_p = 1.0 if view_idx == 0 else 0.1
    return T.Compose(
        [
            T.RandomResizedCrop(
                size, scale=(scale_min, scale_max), interpolation=T.InterpolationMode.BILINEAR
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=blur_p),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ViewBank:
    """Named, parameterized view pipelines built from config.

    config example::

        views:
          ms_v1: {name: simclr_strong, size: 112, view_idx: 0, vertical_flip: true}
          ms_v2: {name: simclr_strong, size: 112, view_idx: 1, vertical_flip: true}
          pair:  {name: cvcl_aug, size: 224}
    """

    def __init__(self, specs: dict[str, dict]):
        self._pipes: dict[str, Callable] = {}
        for alias, spec in specs.items():
            spec = dict(spec)
            builder = VIEWS.get(spec.pop("name"))
            self._pipes[alias] = builder(**spec)

    def names(self):
        return sorted(self._pipes)

    def apply(self, pil_img, view_names: list[str]) -> torch.Tensor:
        """PIL -> stacked views [V,C,H,W]."""
        return torch.stack([self._pipes[v](pil_img) for v in view_names], dim=0)

    def apply_clip(self, clip: torch.Tensor, view_names: list[str]) -> torch.Tensor:
        """[T,C,H,W] float clip -> stacked views [V,T,C,H,W].

        Each named pipe is called once on the whole clip, so its random
        parameters are shared by all T frames (temporal consistency)."""
        return torch.stack([self._pipes[v](clip) for v in view_names], dim=0)
