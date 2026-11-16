"""Standard single-crop image MAE augmentation."""

import torchvision.transforms as T
from PIL import ImageOps

from egolearn.config.registry import VIEWS
from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD


@VIEWS.register("mae")
def mae(
    size: int = 224,
    scale_min: float = 0.2,
    vertical_flip: bool = False,
    uint8: bool = False,
    jitter_probability: float = 0.0,
) -> T.Compose:
    """Bicubic random resized crop, horizontal flip, and ImageNet normalization."""
    if not 0 <= jitter_probability <= 1:
        raise ValueError("jitter_probability must be in [0, 1]")
    ops: list = []
    if vertical_flip:
        ops.append(T.Lambda(ImageOps.flip))
    return T.Compose(
        [
            *ops,
            T.RandomResizedCrop(
                size, scale=(scale_min, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(),
            *(
                [T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=jitter_probability)]
                if jitter_probability > 0
                else []
            ),
            *(
                [T.PILToTensor()]
                if uint8
                else [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
            ),
        ]
    )
