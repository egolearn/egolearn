"""CPU Gaussian blur for the PIL image views, with an explicit numerical tradeoff.

The SimCLR-style views draw sigma ~ U(0.1, 2.0) and filter with a kernel of
about a tenth of the view size (23 taps at 224 px).  torchvision's reference
implementation convolves with the full 2-D kernel: 529 multiply-adds per
pixel and channel.  The separable implementation below applies the same
kernel as a horizontal and a vertical pass (46 multiply-adds) with the same
sigma draw, padding and rounding point, so it is the default; the reference
stays available as `impl="torchvision"` for bit-exact comparisons, and
`impl="none"` skips the filter while consuming the sigma draw (an ablation
that leaves every later random decision in the pipeline unchanged).

Small sigmas put the kernel's tail weights in float32's subnormal range,
where x86 arithmetic is roughly a hundred times slower.  The Loader's
`flush_denormal` option (on by default) flushes those weights to zero inside
the worker processes; the rounded uint8 result is identical either way.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

BLUR_IMPLS = ("separable", "torchvision", "none")


def gaussian_blur(kernel_size: int, impl: str = "separable") -> torch.nn.Module:
    """Build an image blur with the reference's probability-independent sigma draw."""
    classes = {
        "torchvision": T.GaussianBlur,
        "separable": SeparableGaussianBlur,
        "none": SkipGaussianBlur,
    }
    if impl not in classes:
        raise ValueError(f"blur_impl must be one of {BLUR_IMPLS}, got {impl!r}")
    return classes[impl](kernel_size, sigma=(0.1, 2.0))


class SkipGaussianBlur(T.GaussianBlur):
    """Skip filtering while consuming the reference transform's sigma draw.

    Keep this inside the original RandomApply so a blur ablation leaves later
    crop, flip, and solarization randomness unchanged, including other views.
    """

    def forward(self, image):
        self.get_params(self.sigma[0], self.sigma[1])
        return image


class SeparableGaussianBlur(torch.nn.Module):
    """Apply the Gaussian as horizontal and vertical convolutions.

    Uses torchvision's kernel construction, reflection padding, and sigma RNG,
    retaining float32 intermediates and rounding only after both passes. This
    is the same filter mathematically, but changing floating-point accumulation
    order can change a rounded pixel by one intensity level, so the reference
    implementation is kept selectable for bit-exact comparisons.

    Only 8-bit RGB and grayscale PIL images are supported; this transform is
    intended for the PIL-based image views, not tensor/clip pipelines.
    """

    def __init__(
        self, kernel_size: int | tuple[int, int], sigma: float | tuple[float, float] = (0.1, 2.0)
    ) -> None:
        super().__init__()
        reference = T.GaussianBlur(kernel_size, sigma)
        self.kernel_size = reference.kernel_size
        self.sigma = reference.sigma

    def forward(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image) or image.mode not in ("RGB", "L"):
            raise TypeError("separable Gaussian blur requires an 8-bit RGB or L PIL image")
        sigma = T.GaussianBlur.get_params(self.sigma[0], self.sigma[1])
        # Contiguous channels avoid a slow CPU convolution layout for PIL RGB.
        tensor = TF.pil_to_tensor(image).to(torch.float32).contiguous()
        tensor = TF.gaussian_blur(tensor, [self.kernel_size[0], 1], [sigma, sigma])
        tensor = TF.gaussian_blur(tensor, [1, self.kernel_size[1]], [sigma, sigma])
        return TF.to_pil_image(tensor.round().to(torch.uint8), mode=image.mode)
