"""Per-image random masking and reconstruction of only the hidden RGB patches."""

from __future__ import annotations

import torch

from egolearn.config.registry import LOSSES
from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.methods.mae.model import MAEDecoder, MAEEncoder
from egolearn.model.losses.base import Loss
from egolearn.model.model import ComposedModel


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, 3, H, W] -> row-major patches [B, L, patch_size**2 * 3]."""
    if images.ndim != 4 or images.size(1) != 3 or patch_size < 1:
        raise ValueError("patchify requires RGB images [B, 3, H, W] and a positive patch_size")
    batch, channels, height, width = images.shape
    if height < 1 or width < 1 or height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be positive multiples of patch_size")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 4, 5, 1).reshape(batch, -1, patch_size**2 * channels)


def random_mask(
    batch_size: int, num_patches: int, mask_ratio: float, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return visible indices and a boolean hidden-patch mask on CPU.

    Sampling on CPU with an explicit generator makes masks independent of the
    training device and of unrelated global random draws.
    """
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be strictly between zero and one")
    keep = int(num_patches * (1 - mask_ratio))
    if batch_size < 1 or not 1 <= keep < num_patches:
        raise ValueError("MAE needs a nonempty batch and at least one visible and one masked patch")
    visible = torch.rand(batch_size, num_patches, generator=generator).argsort(dim=1)[:, :keep]
    hidden = torch.ones(batch_size, num_patches, dtype=torch.bool)
    hidden.scatter_(1, visible, False)
    return visible, hidden


def reconstruction_loss(
    images: torch.Tensor,
    predictions: torch.Tensor,
    hidden: torch.Tensor,
    *,
    patch_size: int,
    norm_pix_loss: bool = True,
) -> torch.Tensor:
    """Masked-patch MSE, optionally standardizing each target patch (sample variance).

    Inputs are the same ImageNet-normalized crops seen by the encoder. Targets
    are detached, and arithmetic is float32 even under mixed precision.
    """
    targets = patchify(images.detach().float(), patch_size)
    if predictions.shape != targets.shape or hidden.shape != targets.shape[:2]:
        raise ValueError("prediction and mask shapes must match the target patch grid")
    if hidden.dtype != torch.bool or not hidden.any():
        raise ValueError("hidden must be a boolean mask containing at least one masked patch")
    if norm_pix_loss:
        variance, mean = torch.var_mean(targets, dim=-1, correction=1, keepdim=True)
        targets = (targets - mean) * torch.rsqrt(variance + 1e-6)
    errors = (predictions.float() - targets).square().mean(dim=-1)
    return errors[hidden].mean()


@LOSSES.register("mae")
class MAE(Loss):
    """Named encoder/decoder MAE loss; masks derive from run seed, step and view."""

    own_forward = True

    def __init__(
        self,
        slot: str,
        encoder: str,
        predictor: str,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        precision: str = "float32",
        name: str = "mae",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be strictly between zero and one")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("MAE precision must be float32 or bfloat16")
        self.slot = slot
        self.encoder = encoder
        self.predictor = predictor
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.precision = precision
        self.requires_slots = [slot]
        self.requires_encoders = [encoder]
        self.requires_predictors = [predictor]

    def compute(self, batch: dict, model: ComposedModel, ctx: Context) -> dict[str, torch.Tensor]:
        views = self._slot(batch, self.slot)[K.SLOT_VIEWS]
        if views.ndim != 5 or views.size(0) < 1 or views.size(1) < 1:
            raise ValueError("MAE requires nonempty image views [B, V, 3, H, W]")
        encoder = model.encoders[self.encoder]
        decoder = model.predictors[self.predictor]
        if not isinstance(encoder, MAEEncoder) or not isinstance(decoder, MAEDecoder):
            raise TypeError("loss:mae requires encoder:mae_vit and predictor:mae")
        if (encoder.image_size, encoder.patch_size) != (decoder.image_size, decoder.patch_size):
            raise ValueError("MAE encoder and decoder must use the same image and patch sizes")
        terms = []
        for view in range(views.size(1)):
            generator = torch.Generator().manual_seed(
                ctx.derive_seed(f"mae.{self.name}.mask.{ctx.clock.step}.{view}")
            )
            visible, hidden = random_mask(
                views.size(0), encoder.num_patches, self.mask_ratio, generator=generator
            )
            images = views[:, view].contiguous().to(ctx.device, non_blocking=True)
            visible, hidden = visible.to(ctx.device), hidden.to(ctx.device)
            # Autocast only the forward pass; targets/loss, parameters, and
            # optimizer state remain float32. BF16 needs no gradient scaler.
            with torch.autocast(
                device_type=ctx.device.type,
                dtype=torch.bfloat16,
                enabled=self.precision == "bfloat16",
            ):
                output = model.encode(self.encoder, images, visible=visible)
                assert output.tokens is not None
                predictions = model.predict(
                    self.predictor,
                    output.tokens,
                    cls_token=output.aux["cls_token"],
                    visible=visible,
                )
            terms.append(
                reconstruction_loss(
                    images,
                    predictions,
                    hidden,
                    patch_size=encoder.patch_size,
                    norm_pix_loss=self.norm_pix_loss,
                )
            )
        return {"mse": torch.stack(terms).mean()}
