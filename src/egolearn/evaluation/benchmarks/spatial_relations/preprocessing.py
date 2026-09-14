"""Fixed preprocessing used by both the probe and checkpoint comparison."""

from torchvision import transforms

from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD


def spatial_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
