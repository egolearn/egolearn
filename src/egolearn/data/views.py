"""View pipelines: the augmentations applied when a batch is built.

A "view" is a registered PIL -> Tensor[C,H,W] transform pipeline.  Each slot
(one named part of a batch recipe) declares a list of view names; the
composer stacks the results into a [V,C,H,W] tensor per item.

Ordinary views receive a PIL image. A view declaring requires_context=True
also receives the read-only video.view_context metadata attached by admission.
This supports temporal crop scores without mutable per-worker stream state.

Dataset-level preprocessing (decode fps, orientation flip, BabyView portrait
pre-crop) does NOT belong here -- it belongs in prepare scripts / adapters.
The one exception, kept as a view flag, is MS's vertical flip: the original
Memory Storyboard applies it at train time because raw SAYCam frames are
stored upside-down, whereas BabyCL frames are already flipped at decode time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision.transforms as T
from PIL import ImageOps

from egolearn.data.blur import gaussian_blur

from ..config.registry import VIEWS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _finish(size):
    return [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


@VIEWS.register("resize")
def resize(size: int = 224, vertical_flip: bool = False):
    """Squash the whole frame to size x size (aspect-distorting), no crop --
    CVCL's and BabyCL's evaluation preprocessing (multimodal_data_module's
    base transform).  Evaluations reproducing their numbers must use this,
    not center_crop: on wide egocentric frames the two differ by points."""
    ops: list = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
    if vertical_flip:
        ops.insert(0, T.Lambda(ImageOps.flip))
    return T.Compose(ops + _finish(size))


@VIEWS.register("center_crop")
def center_crop(size: int = 224, vertical_flip: bool = False):
    ops = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
    if vertical_flip:
        ops.insert(0, T.Lambda(ImageOps.flip))
    return T.Compose(ops + _finish(size))


@VIEWS.register("imagenet_center_crop")
def imagenet_center_crop(size: int = 224):
    """Bilinear resize to 256/224 of size, then crop for ImageNet linear decoding.

    The SGD linear probe uses this evaluation view by default; at
    size=112 this resizes the shorter edge to 128 before cropping to 112.
    """
    ops = [
        T.Resize(
            256 * size // 224,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.CenterCrop(size),
    ]
    return T.Compose(ops + _finish(size))


@VIEWS.register("simclr_strong")
def simclr_strong(
    size: int = 224,
    view_idx: int = 0,
    scale_min: float = 0.2,
    scale_max: float = 1.0,
    vertical_flip: bool = False,
    blur_impl: str = "separable",
):
    """Standard SimCLR strong augmentation applied to the RAW frame (usually
    paired with the ntxent loss): view 0 applies GaussianBlur with p=1.0;
    view 1 applies GaussianBlur with p=0.1 and solarizes with p=0.2.
    `vertical_flip` reproduces Memory Storyboard's train-time flip of the
    upside-down raw SAYCam frames.

    NOTE: BabyCL's effective segment augmentation center-crops to `size`
    BEFORE the random crop -- a two-stage chain left over from its original
    pipeline.  That exact chain is the project view `babycl_ms_aug`, not
    this one.

    `blur_impl` picks the Gaussian blur implementation (egolearn.data.blur):
    `separable` (default) applies the reference kernel in two 1-D passes and
    can differ from the reference by one intensity level per pixel;
    `torchvision` is the bit-exact reference; `none` skips the filter while
    consuming its sigma draw, so every later random decision is unchanged.
    """
    k = max(3, int(0.1 * size) // 2 * 2 + 1)
    blur = gaussian_blur(k, blur_impl)
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
    ops.append(T.RandomApply([blur], p=blur_p))
    if view_idx == 1:
        ops.append(
            T.Lambda(
                lambda img: (
                    ImageOps.solarize(img, threshold=128) if torch.rand(1).item() < 0.2 else img
                )
            )
        )
    return T.Compose(ops + _finish(size))


# Augmentation pipelines that belong to one method (e.g. view:cvcl_aug) live
# in that project's code and register into this same VIEWS registry.


# ---------------------------------------------------------------------------
# Clip views: [T,C,H,W] float tensor -> [T,C,H,W], TEMPORALLY CONSISTENT.
#
# This follows the mainstream video-SSL convention (VideoMAE / V-JEPA / video
# contrastive): spatial augmentation parameters are sampled ONCE PER CLIP and
# applied to all T frames identically -- otherwise each frame gets its own
# random crop and the clip's temporal structure is destroyed.  torchvision v1
# transforms give this for free on batched tensors: one __call__ samples
# parameters once and applies them across the leading (time) dimension.
#
# Clip slots must use clip-domain views (by convention their names start with
# clip_); image views expect PIL input and will fail loudly on a clip tensor.
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


@VIEWS.register("clip_rrc")
def clip_rrc(
    size: int = 224,
    scale_min: float = 0.3,
    scale_max: float = 1.0,
    ratio_min: float = 0.75,
    ratio_max: float = 1.35,
):
    """Random resized crop + horizontal flip, temporally consistent -- the
    V-JEPA pretraining augmentation (deliberately no color jitter: masked
    prediction supplies the invariance pressure that contrastive pipelines
    get from photometric augmentation)."""
    return T.Compose(
        [
            T.RandomResizedCrop(
                size,
                scale=(scale_min, scale_max),
                ratio=(ratio_min, ratio_max),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@VIEWS.register("clip_simclr")
def clip_simclr(size: int = 224, view_idx: int = 0, scale_min: float = 0.2, scale_max: float = 1.0):
    """SimCLR-style clip augmentation; random parameters are drawn once per
    clip and shared by all of its frames."""
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
    """Holds the view pipelines built from config, addressed by alias.

    config example::

        views:
          ms_v1: {name: simclr_strong, size: 112, view_idx: 0, vertical_flip: true}
          ms_v2: {name: simclr_strong, size: 112, view_idx: 1, vertical_flip: true}
          pair:  {name: cvcl_aug, size: 224}
    """

    def __init__(self, specs: dict[str, dict]):
        # kept so a worker process can rebuild the bank: the built pipelines
        # are closures and do not pickle
        self.specs = {alias: dict(spec) for alias, spec in specs.items()}
        self._pipes: dict[str, Callable] = {}
        for alias, spec in specs.items():
            spec = dict(spec)
            builder = VIEWS.get(spec.pop("name"))
            self._pipes[alias] = builder(**spec)

    def get(self, name: str):
        """Return a constructed view, including any declared context contract."""
        return self._pipes[name]

    def names(self):
        return sorted(self._pipes)

    def apply(self, pil_img, view_names: list[str], context: dict | None = None) -> torch.Tensor:
        """Run each named pipeline on the PIL image and stack the results
        into a [V,C,H,W] tensor, one row per view."""
        outputs = []
        for name in view_names:
            pipe = self._pipes[name]
            if getattr(pipe, "requires_context", False):
                if context is None:
                    raise ValueError(f"view {name!r} requires frame context from admission")
                outputs.append(pipe(pil_img, context))
            else:
                outputs.append(pipe(pil_img))
        return torch.stack(outputs, dim=0)

    def apply_clip(self, clip: torch.Tensor, view_names: list[str]) -> torch.Tensor:
        """Run each named pipeline on a [T,C,H,W] float clip and stack the
        results into [V,T,C,H,W].

        Each named pipeline is called once on the whole clip, so its random
        parameters are shared by all T frames (temporal consistency)."""
        return torch.stack([self._pipes[v](clip) for v in view_names], dim=0)
