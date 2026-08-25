"""V-JEPA: masked prediction in representation space, on video clips.

The objective (Bardes et al. 2024; scaled up as V-JEPA 2, Assran et al.
2025): hide part of a clip, predict the hidden part's REPRESENTATION -- not
its pixels -- and take the training signal from how far the prediction lands
from a slowly-moving target encoder's view of the same clip.  One step:

  1. the frozen target encoder embeds the full clip (no gradient; token
     outputs are layer-normalized, as in the original);
  2. for each configured mask kind a multiblock mask is drawn (see
     model/masking.py) -- V-JEPA pairs many small blocks with few large ones;
  3. the context encoder sees only the unmasked tokens, the predictor fills
     in the masked positions from them;
  4. the L1 distance to the target tokens at those positions, averaged over
     the mask kinds and views, is the loss.

After every step `on_step_end` updates the target as an EMA of the context
encoder (momentum ramps `ema[0]` -> `ema[1]` over stream progress; without a
progress clock it stays at `ema[0]`).  At step 0 the target starts as an
exact copy; on resume its weights come from the snapshot instead.

This loss runs its own encoder forwards (`own_forward`): the shared pooled
precompute would be wasted work for its slot, so the drivers skip it.  Mask
randomness is drawn from seeds derived from (run seed, step, view, kind),
so runs are reproducible and resumable.

Config (encoders/predictor addressed by name, target declared frozen):

    model:
      encoders:
        video:        {_target_: "encoder:video_vit", dim: 1024, depth: 24, heads: 16}
        video_target: {_target_: "encoder:video_vit", dim: 1024, depth: 24, heads: 16,
                       frozen: true}
      predictors:
        pred: {inputs: [video], _target_: "predictor:vjepa",
               encoder_dim: 1024, dim: 384, depth: 12, heads: 12}
    losses:
      - _target_: "loss:vjepa"
        name: jepa
        slot: clips
        context: video
        target: video_target
        predictor: pred
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES
from egolearn.core import keys as K
from egolearn.model.masking import multiblock

from .base import Loss

# V-JEPA's two mask kinds: 8 small blocks / 2 large ones, full clip duration
DEFAULT_MASKS = {
    "short": {"num_blocks": 8, "scale": (0.15, 0.15), "aspect": (0.75, 1.5)},
    "long": {"num_blocks": 2, "scale": (0.7, 0.7), "aspect": (0.75, 1.5)},
}


@LOSSES.register("vjepa")
class VJEPA(Loss):
    own_forward = True  # runs masked encoder forwards itself; skip precompute

    def __init__(
        self,
        slot: str,
        context: str,
        target: str,
        predictor: str,
        masks: dict[str, dict] | None = None,
        ema: tuple[float, float] = (0.998, 1.0),
        name: str = "jepa",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.slot = slot
        self.context = context
        self.target = target
        self.predictor = predictor
        self.masks = {k: dict(v) for k, v in (masks or DEFAULT_MASKS).items()}
        self.ema = (float(ema[0]), float(ema[1]))
        self.requires_slots = [slot]
        self.requires_encoders = [context]
        self.requires_frozen_encoders = [target]
        self.requires_predictors = [predictor]
        self._synced = False  # step-0 copy done (or skipped because resumed)

    def compute(self, batch, model, ctx) -> dict[str, torch.Tensor]:
        if not self._synced:
            if ctx.clock.step == 0:
                # the target starts as an exact copy of the context encoder;
                # on resume (step > 0) the snapshot already carries its state
                tgt_enc = model.encoders[self.target]
                tgt_enc.load_state_dict(model.encoders[self.context].state_dict())
            self._synced = True

        views = self._slot(batch, self.slot)[K.SLOT_VIEWS]  # [B, V, T, C, H, W]
        n_terms = views.size(1) * len(self.masks)
        total: torch.Tensor | None = None
        for v in range(views.size(1)):
            x = views[:, v].contiguous().to(ctx.device, non_blocking=True)
            with torch.no_grad():
                tout = model.encode(self.target, x)
                targets = F.layer_norm(tout.tokens, (tout.tokens.size(-1),))
            gt, gh, gw = (int(g) for g in tout.aux["grid"].tolist())
            grid = (gt, gh, gw)
            for kind, spec in self.masks.items():
                rng = ctx.fresh_rng(f"vjepa.{self.name}.mask.{ctx.clock.step}.{v}.{kind}")
                ctx_idx, tgt_idx = multiblock(grid, views.size(0), rng=rng, **spec)
                ctx_idx, tgt_idx = ctx_idx.to(x.device), tgt_idx.to(x.device)
                cout = model.encode(self.context, x, visible=ctx_idx)
                pred = model.predict(
                    self.predictor, cout.tokens, ctx_pos=ctx_idx, tgt_pos=tgt_idx, grid=grid
                )
                want = torch.gather(
                    targets, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, targets.size(-1))
                )
                term = F.l1_loss(pred, want) / n_terms
                total = term if total is None else total + term
        assert total is not None
        return {"l1": total}

    def on_step_end(self, model, ctx) -> None:
        p = ctx.clock.progress
        m = self.ema[0] if p is None else self.ema[0] + (self.ema[1] - self.ema[0]) * p
        src = model.encoders[self.context]
        tgt = model.encoders[self.target]
        with torch.no_grad():
            for pc, pt in zip(src.parameters(), tgt.parameters(), strict=True):
                pt.lerp_(pc, 1.0 - m)
