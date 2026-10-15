"""Data components specific to the BabyCL method: the CVCL augmentation views and the
frozen encoder its segmenter uses for embeddings.  (The BabyView dataset adapter is a
core tool -- see egolearn/data/sources/saycam.py.)"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps

from egolearn.config.registry import EMBEDDERS, VIEWS
from egolearn.core.refs import ItemRef
from egolearn.data.blur import gaussian_blur
from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD
from egolearn.memory.admission.base import EmbeddingProvider


class PILRadiusBlur:
    """Pillow's radius-based Gaussian filter.

    ``sigma`` bounds a uniform radius sampled from Python's global RNG. Each
    call consumes one draw. Pillow's filter is part of the numerical contract;
    substituting a tensor convolution changes pixels.
    """

    def __init__(self, sigma: tuple[float, float] = (0.1, 2.0)) -> None:
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        lower, upper = self.sigma
        radius = lower + (upper - lower) * random.random()
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


@VIEWS.register("cvcl_aug")
def cvcl_aug(size: int = 224):
    """CVCL/BabyCL multimodal-frame augmentation, 1:1 with
    ms_dataloader.an_transform: RandomResizedCrop(0.2-1.0) + PIL radius blur
    p=0.5 + horizontal flip (blur BEFORE flip, as in the original)."""
    return T.Compose(
        [
            T.RandomResizedCrop((size, size), scale=(0.2, 1.0)),
            T.RandomApply([PILRadiusBlur((0.1, 2.0))], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@VIEWS.register("babycl_ms_aug")
def babycl_ms_aug(
    size: int = 224,
    view_idx: int = 0,
    scale_min: float = 0.2,
    scale_max: float = 1.0,
    blur_impl: str = "separable",
):
    """The augmentation BabyCL EFFECTIVELY applies to visual-segment frames, 1:1 with
    the original, which runs in two stages:

      gather:  ms_transform  = Resize(size, BICUBIC) + CenterCrop(size)
      train:   two_crops     = de-normalize back to a PIL image, then the strong
               SimCLR pipeline (build_two_view_augment "strong") ON THE ALREADY
               CENTER-CROPPED image.

    Here the two stages are fused into one PIL pipeline (the de-normalize/PIL
    round-trip changes nothing beyond uint8 rounding).  This differs from the
    generic core `simclr_strong`, which random-crops the RAW frame -- BabyCL's
    random crops only ever see the center-cropped 224x224 content.
    view 0: blur p=1.0; view 1: blur p=0.1 + solarize p=0.2.

    `blur_impl` is the same switch as in the core `simclr_strong` view:
    `separable` (default, within one intensity level of the reference),
    `torchvision` (bit-exact reference), `none` (ablation; keeps the RNG
    draws).  Set it on both ms views."""
    k = max(3, int(0.1 * size) // 2 * 2 + 1)
    blur_p = 1.0 if view_idx == 0 else 0.1
    blur = gaussian_blur(k, blur_impl)
    ops: list = [
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.RandomResizedCrop(
            size, scale=(scale_min, scale_max), interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([blur], p=blur_p),
    ]
    if view_idx == 1:
        ops.append(
            T.Lambda(
                lambda img: (
                    ImageOps.solarize(img, threshold=128) if torch.rand(1).item() < 0.2 else img
                )
            )
        )
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


@EMBEDDERS.register("frozen_resnext50")
class FrozenResNeXt50Embedder(EmbeddingProvider):
    """Frozen torchvision ResNeXt-50 -- the encoder BabyCL's online segmenter uses to
    embed frames (matches online_ms_refine.build_backbone: IMAGENET1K_V1 weights,
    fc=Identity, eval mode, center-crop transform, per-frame L2 normalization).

    Note that BabyCL's segmenter depends on ImageNet pretraining; making that
    dependency visible here is deliberate.
    """

    def __init__(self, pretrained: bool = True, view: str = "eval", batch_size: int = 64):
        self.pretrained = pretrained
        self.view = view
        self.batch_size = batch_size
        self._model = None

    def _lazy_model(self):
        if self._model is None:
            import torchvision.models as tvm

            weights = tvm.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = tvm.resnext50_32x4d(weights=weights)
            m.fc = torch.nn.Identity()
            self._model = m.eval().to(self.ctx.device)
        return self._model

    @torch.no_grad()
    def embed_refs(self, refs: list[ItemRef], normalize: bool = True) -> torch.Tensor:
        model = self._lazy_model()
        outs = []
        for i in range(0, len(refs), self.batch_size):
            imgs = [
                self.loader.frame_views(r, 0, [self.view])[0] for r in refs[i : i + self.batch_size]
            ]
            x = torch.stack(imgs, dim=0).to(self.ctx.device)
            outs.append(model(x).cpu())
        feats = torch.cat(outs, dim=0)
        return F.normalize(feats, dim=1) if normalize else feats
