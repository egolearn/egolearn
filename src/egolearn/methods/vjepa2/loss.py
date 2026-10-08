"""V-JEPA 2 masked latent L1, reusing the shared frozen-teacher EMA lifecycle."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egolearn.config.registry import LOSSES
from egolearn.core import keys as K
from egolearn.methods.vjepa2.masking import multiblock
from egolearn.methods.vjepa2.model import VJEPA2Encoder, VJEPA2Predictor
from egolearn.model.losses.vjepa import VJEPA


@LOSSES.register("vjepa2")
class VJEPA2(VJEPA):
    requires_offline = True
    supports_grad_accumulation = False

    def __init__(
        self,
        slot: str,
        context: str,
        target: str,
        predictor: str,
        masks: dict[str, dict] | None = None,
        ema: tuple[float, float] = (0.998, 0.998),
        precision: str = "float32",
        name: str = "jepa",
        weight: float = 1.0,
    ):
        super().__init__(slot, context, target, predictor, masks, ema, name, weight)
        if precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if not 0 <= self.ema[0] <= self.ema[1] <= 1:
            raise ValueError("EMA must satisfy 0 <= start <= end <= 1")
        if masks is not None and not masks:
            raise ValueError("V-JEPA 2 requires at least one mask kind")
        self.precision = precision
        self.diagnostics: dict[str, torch.Tensor | float] = {}

    def validate_model(self, model) -> None:
        context, target = model.encoders[self.context], model.encoders[self.target]
        if not isinstance(context, VJEPA2Encoder) or not isinstance(target, VJEPA2Encoder):
            raise ValueError("loss:vjepa2 requires encoder:vjepa2 for context and target")
        pred = model.predictors[self.predictor]
        if not isinstance(pred, VJEPA2Predictor) or pred.embed.in_features != context.out_dim:
            raise ValueError("loss:vjepa2 requires a width-compatible predictor:vjepa2")
        if self.target not in model.frozen_names():
            raise ValueError("V-JEPA 2 target must be frozen")
        config = model.construction_config
        if config is not None:
            specs = config["model"]["encoders"]
            a, b = (dict(specs[name]) for name in (self.context, self.target))
            a.pop("frozen", None)
            b.pop("frozen", None)
            if a != b:
                raise ValueError("V-JEPA 2 context and target construction specs must match")
        if context.state_dict().keys() != target.state_dict().keys():
            raise ValueError("V-JEPA 2 context and target architecture must match")

    def initialize_target(self, model) -> None:
        """Synchronize a fresh run before baseline saving; never call after resume."""
        self.validate_model(model)
        model.encoders[self.target].load_state_dict(model.encoders[self.context].state_dict())
        self._synced = True

    def compute(self, batch, model, ctx) -> dict[str, torch.Tensor]:
        if not self._synced:
            if ctx.clock.step == 0:
                self.initialize_target(model)
            self._synced = True
        views = self._slot(batch, self.slot)[K.SLOT_VIEWS]
        if views.ndim != 6 or views.size(0) < 1 or views.size(1) < 1:
            raise ValueError("V-JEPA 2 requires nonempty [B,V,T,C,H,W] clips")
        terms = []
        diagnostics: dict[str, torch.Tensor | float] = {}
        device = torch.device(ctx.device)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=self.precision == "bfloat16"
        ):
            for v in range(views.size(1)):
                x = views[:, v].contiguous().to(device, non_blocking=True)
                with torch.no_grad():
                    out = model.encode(self.target, x)
                    targets = F.layer_norm(out.tokens.float(), (out.tokens.size(-1),))
                    diagnostics[f"view{v}/target_std"] = targets.std((0, 1), correction=0).mean()
                gt, gh, gw = (int(g) for g in out.aux["grid"].tolist())
                grid = (gt, gh, gw)
                for kind, spec in self.masks.items():
                    rng = ctx.fresh_rng(f"vjepa2.{self.name}.{ctx.clock.step}.{v}.{kind}")
                    context, target = multiblock(grid, x.size(0), rng=rng, **spec)
                    context, target = context.to(device), target.to(device)
                    encoded = model.encode(self.context, x, visible=context)
                    pred = model.predict(
                        self.predictor,
                        encoded.tokens,
                        ctx_pos=context,
                        tgt_pos=target,
                        grid=grid,
                        mask_index=0,
                    )
                    want = targets.gather(1, target.unsqueeze(-1).expand(-1, -1, targets.size(-1)))
                    term = F.l1_loss(pred.float(), want)
                    terms.append(term)
                    prefix = f"view{v}/{kind}"
                    diagnostics[f"{prefix}/l1"] = term.detach()
                    diagnostics[f"{prefix}/context_tokens"] = float(context.size(1))
                    diagnostics[f"{prefix}/target_tokens"] = float(target.size(1))
                    diagnostics[f"{prefix}/prediction_std"] = (
                        pred.detach().float().std((0, 1), correction=0).mean()
                    )
        self.diagnostics = diagnostics
        return {"l1": torch.stack(terms).mean()}
