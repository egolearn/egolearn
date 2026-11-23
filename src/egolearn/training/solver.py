"""Solver: assigns model parameters to optimizer groups and computes the LR schedule.

Both reproductions need this, which is why it is shared code: BabyCL's four LR tiers with
BN/bias decay exemption become two matcher rules; Memory Storyboard's single SGD group is one
rule.  Each parameter goes to the first rule whose regex matches its fully-qualified name.

The schedule is a function of progress -- the fraction of the stream consumed so far, used
instead of epochs -- not of a precomputed total step count.  This removes BabyCL's need to
estimate the step count from the offline storyboard segment count.  A reproduction of a
step-indexed original can opt back in with `total_steps` (see ProgressCosine).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch

# substring semantics exactly as ms_train._split_decay_named_params_cnn:
# .bias suffix, or "bn"/"norm"/"gn"/"downsample.1" anywhere in the name
_NO_DECAY_DEFAULT = r"(\.bias$|bn|norm|gn|downsample\.1)"


def build_optimizer(
    model: torch.nn.Module,
    groups: list[dict],
    opt: str = "adamw",
    momentum: float = 0.9,
    warnings: list[str] | None = None,
    betas: Sequence[float] = (0.9, 0.999),
):
    """groups: [{match, lr, weight_decay, split_no_decay?, no_decay_pattern?}]

    With split_no_decay=true a rule contributes two torch param groups: one
    with the given weight_decay and one with 0.0 for params whose name matches
    `no_decay_pattern` (default: bias / BN / norm, matching BabyCL's
    _split_decay_named_params_cnn).
    `betas` configures Adam/AdamW moments; defaults retain PyTorch's values.
    """
    compiled: list[dict[str, Any]] = []
    for g in groups:
        compiled.append(
            {
                "re": re.compile(g["match"]),
                "lr": float(g["lr"]),
                "wd": float(g.get("weight_decay", 0.0)),
                "split": bool(g.get("split_no_decay", False)),
                "nd_re": re.compile(g.get("no_decay_pattern", _NO_DECAY_DEFAULT)),
                "decay_params": [],
                "no_decay_params": [],
            }
        )

    unmatched = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for g in compiled:
            if g["re"].search(name):
                bucket = (
                    "no_decay_params"
                    if g["split"] and g["nd_re"].search(name.lower())
                    else "decay_params"
                )
                g[bucket].append(p)
                break
        else:
            unmatched.append(name)
    if warnings is not None:
        for g, spec in zip(compiled, groups, strict=True):
            if not g["decay_params"] and not g["no_decay_params"]:
                warnings.append(
                    f"solver group {spec['match']!r} matched no parameters -- "
                    "stale regex? (parameter names look like encoders.vision.*)"
                )
    if unmatched:
        raise ValueError(
            f"solver: {len(unmatched)} parameters matched no group rule, "
            f"e.g. {unmatched[:3]} -- add a catch-all rule {{match: '.*'}}"
        )

    torch_groups = []
    for g in compiled:
        if g["decay_params"]:
            torch_groups.append(
                {"params": g["decay_params"], "lr": g["lr"], "weight_decay": g["wd"]}
            )
        if g["no_decay_params"]:
            torch_groups.append(
                {"params": g["no_decay_params"], "lr": g["lr"], "weight_decay": 0.0}
            )

    if opt in ("adam", "adamw"):
        if len(betas) != 2:
            raise ValueError("Adam betas must contain exactly two values")
        moments = (float(betas[0]), float(betas[1]))
        if opt == "adamw":
            return torch.optim.AdamW(torch_groups, betas=moments)
        return torch.optim.Adam(torch_groups, betas=moments)
    if opt == "sgd":
        for tg in torch_groups:
            tg["momentum"] = momentum
        return torch.optim.SGD(torch_groups)
    raise ValueError(f"unknown optimizer {opt!r}")


class ProgressCosine:
    """LR multiplier driven by progress (the fraction of the stream consumed so far):
    linear warmup over `warmup_frac` of the TRAINING RUN, then cosine decay to `floor`.

    Anchoring (`anchor=True`, default): the LR warmup starts counting from the first
    training step, not from the start of buffer filling.  Concretely, the schedule is
    indexed by progress made SINCE TRAINING STARTED rather than raw stream progress.
    The driver's buffer-warmup phase consumes stream before the first gradient step
    (BabyCL fills the short tier with ~10% of the stream); without anchoring the LR
    warmup would already be over by then and the cosine curve shifted.  With it, the
    curve matches the original's step-indexed schedule: warmup over 2% of the run,
    cosine over the remainder.

    `warmup_min_steps` is a LOWER bound on warmup length (the original BabyCL:
    `max(100, int(0.02 * total_steps))`) -- on very short runs warmup stretches to
    at least this many gradient steps.

    When the stream announces no total length (open-ended), progress is None and the
    multiplier stays at 1.0 once the step-count warmup is done -- constant LR is the
    sane open-ended default; smarter open-ended schedules are a deferred question.

    `total_steps` switches to a STEP-indexed schedule, for reproducing an original
    that fixes its step budget up front (the original BabyCL: the step count from the offline
    storyboard, with `driver.max_steps` set to the same value).  Gradient step `s`
    (0-based, `clock.step`) gets exactly the original's LambdaLR factor: with
    `warm = max(warmup_min_steps, int(warmup_frac * total_steps))`, `(s + 1) / warm`
    while `s < warm`, then cosine from 1 to `floor` over the remaining
    `total_steps - warm` steps.  Stream progress and `anchor` are ignored.
    """

    def __init__(
        self,
        warmup_frac: float = 0.02,
        warmup_min_steps: int = 100,
        floor: float = 0.0,
        constant: bool = False,
        anchor: bool = True,
        total_steps: int | None = None,
    ):
        self.warmup_frac = float(warmup_frac)
        self.warmup_min_steps = int(warmup_min_steps)
        self.floor = float(floor)
        self.constant = bool(constant)
        self.anchor = bool(anchor)
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError(f"schedule.total_steps must be positive, got {total_steps}")
        self.total_steps = int(total_steps) if total_steps is not None else None
        self._prog0: float | None = None  # progress at first gradient step

    def multiplier(self, ctx) -> float:
        import math

        prog = ctx.clock.progress
        if self.constant:
            return 1.0
        if self.total_steps is not None:
            total = self.total_steps
            warm = max(self.warmup_min_steps, int(self.warmup_frac * total))
            step = ctx.clock.step
            if step < warm:
                return (step + 1) / max(1, warm)
            p = min(1.0, (step - warm) / max(1, total - warm))
            return self.floor + (1.0 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * p))
        if prog is None:
            warm = max(1, self.warmup_min_steps)
            return min(1.0, (ctx.clock.step + 1) / warm)
        if self.anchor:
            if self._prog0 is None:
                self._prog0 = prog
            prog = max(0.0, (prog - self._prog0) / max(1e-8, 1.0 - self._prog0))
        if self.warmup_frac > 0 or self.warmup_min_steps > 0:
            by_progress = (prog / self.warmup_frac) if self.warmup_frac > 0 else 1.0
            by_steps = (ctx.clock.step + 1) / max(1, self.warmup_min_steps)
            # warmup length = max(frac-of-run, min_steps)  ->  ramp = the slower
            w = min(by_progress, by_steps)
            if w < 1.0:
                return w
        span = max(1e-8, 1.0 - self.warmup_frac)
        p = (prog - self.warmup_frac) / span
        return self.floor + (1.0 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    # anchor state must survive snapshots (resume would otherwise re-anchor)
    def state_dict(self) -> dict:
        return {"prog0": self._prog0}

    def load_state_dict(self, st: dict) -> None:
        self._prog0 = st.get("prog0")


def apply_lr(optimizer, base_lrs: list[float], mult: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs, strict=False):
        group["lr"] = base * mult
