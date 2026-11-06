"""Two-stage motion-biased cropping and appearance regularization for StreamMAE."""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from egolearn.config.registry import VIEWS
from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD
from egolearn.methods.mae.model import MAEEncoder
from egolearn.methods.streammae.motion import CONTEXT_KIND, frame_geometry, prepare_image


def select_crop(
    motion: np.ndarray, crop_patches: tuple[int, int], candidates: torch.Tensor
) -> tuple[int, int]:
    """Select highest mean-motion region from [K, 2] patch-grid (row, col) starts."""
    hh, ww = crop_patches
    if hh < 1 or ww < 1 or candidates.ndim != 2 or candidates.shape[1] != 2 or not len(candidates):
        raise ValueError("need positive crop dimensions and nonempty [K, 2] candidates")
    scores = []
    for row, col in candidates.tolist():
        if row < 0 or col < 0 or row + hh > motion.shape[0] or col + ww > motion.shape[1]:
            raise ValueError("candidate falls outside the motion grid")
        scores.append(float(motion[row : row + hh, col : col + ww].mean()))
    # All-zero maps choose the first uniformly drawn candidate, including at
    # session starts. Ties never force the same absolute spatial location.
    winner = int(np.argmax(scores))
    row, col = candidates[winner].tolist()
    return row, col


@VIEWS.register("streammae")
class StreamMAEView:
    requires_context = True
    context_kind = CONTEXT_KIND

    def __init__(
        self,
        size: int = 224,
        scale_min: float = 0.2,
        uint8: bool = False,
        frame_size=None,
        patch_size: int = 16,
        crop_size=None,
        crop_fraction: tuple[float, float] = (336 / 720, 592 / 1280),
        crop_probability: float = 0.5,
        motion_candidates: int = 4,
        jitter_probability: float = 0.5,
    ):
        self.frame_size = frame_geometry(frame_size)
        self.patch_size = int(patch_size)
        self.crop_size = frame_geometry(crop_size)
        if patch_size < 1 or motion_candidates < 1:
            raise ValueError("patch_size and motion_candidates must be positive")
        if not 0 <= crop_probability <= 1 or not 0 <= jitter_probability <= 1:
            raise ValueError("crop/jitter probabilities must be in [0, 1]")
        if len(crop_fraction) != 2 or any(not 0 < f <= 1 for f in crop_fraction):
            raise ValueError("crop_fraction must contain two fractions in (0, 1]")
        if self.crop_size is not None and any(n % patch_size for n in self.crop_size):
            raise ValueError("crop_size must align to the patch grid")
        self.crop_fraction = tuple(crop_fraction)
        self.crop_probability = float(crop_probability)
        self.motion_candidates = int(motion_candidates)
        self.final = T.Compose(
            [
                T.RandomResizedCrop(
                    size, scale=(scale_min, 1.0), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=jitter_probability),
                *(
                    [T.PILToTensor()]
                    if uint8
                    else [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
                ),
            ]
        )

    def validate_frame_size(self, frame_size: tuple[int, int]) -> None:
        if min(frame_size) < self.patch_size:
            raise ValueError("frame is smaller than one motion patch")
        if self.crop_size is not None and any(
            crop > frame for crop, frame in zip(self.crop_size, frame_size, strict=True)
        ):
            raise ValueError("first-stage crop does not fit the frame")

    def validate_context_producer(self, producer, model) -> None:
        if producer.patch_size != self.patch_size or producer.frame_size != self.frame_size:
            raise ValueError("StreamMAE admission/view geometry must agree")
        for encoder in model.encoders.values():
            if isinstance(encoder, MAEEncoder) and encoder.patch_size != self.patch_size:
                raise ValueError("StreamMAE view/encoder geometry must agree")
        if self.frame_size is not None:
            self.validate_frame_size(self.frame_size)

    def __call__(self, image: Image.Image, context: dict) -> torch.Tensor:
        image = prepare_image(image, self.frame_size)
        width, height = image.size
        self.validate_frame_size((height, width))
        patch = self.patch_size
        motion = context["motion"]
        if (
            context["patch_size"] != patch
            or tuple(context["frame_size"]) != (height, width)
            or motion.shape != (height // patch, width // patch)
        ):
            raise ValueError("motion metadata and augmentation geometry disagree")
        if torch.rand(()).item() < self.crop_probability:
            grid = (height // patch, width // patch)
            crop = (
                tuple(n // patch for n in self.crop_size)
                if self.crop_size is not None
                else tuple(
                    max(1, min(g, round(n * f / patch)))
                    for g, n, f in zip(grid, (height, width), self.crop_fraction, strict=True)
                )
            )
            if any(c > g for c, g in zip(crop, grid, strict=True)):
                raise ValueError("first-stage crop does not fit the frame")
            candidates = torch.stack(
                [
                    torch.randint(g - c + 1, (self.motion_candidates,))
                    for g, c in zip(grid, crop, strict=True)
                ],
                dim=1,
            )
            row, col = select_crop(motion, (crop[0], crop[1]), candidates)
            image = image.crop(
                (col * patch, row * patch, (col + crop[1]) * patch, (row + crop[0]) * patch)
            )
        return self.final(image)
