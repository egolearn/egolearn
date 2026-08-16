"""OfflineDriver: traditional IID training -- the PARALLEL mode to streaming.

Design decision (2026-08 working group): IID and streaming are parallel
training modes, not one degenerate config of the same loop.  Offline IID is
the completely conventional recipe -- index the whole dataset up front,
shuffle, train in epochs -- with **no memory tables, no admission, no rate
limiter**.  It lives inside the framework (instead of borrowing someone
else's implementation) because it is the upper bound every streaming method
is compared against.

Shared with the streaming path: sources/decoding/Loader/views (the whole
data plane), model/losses/solver, hooks/probes, and the stage-latency
ledger.  Only the loop differs.  Config switch:

    driver:
      mode: offline                       # default: streaming
      batch: {slot: frames, n: 256, views: [v1, v2]}
      epochs: 10                          # and/or max_steps
      subsample: 1                        # take every k-th frame into the index

Clocks: in offline mode `progress` is the fraction of planned gradient steps
(epochs are conventional here; the "no epochs" rule applies to the streaming
loop only).  Epoch e's shuffle order comes from the named RNG
"offline.shuffle.epoch<e>" -- derived from (seed, epoch), never consumed
across epochs -- so runs are deterministic AND resumable: `load_snapshot`
restores the clock and fit() continues from step % batches_per_epoch of the
regenerated epoch order.

V1 scope: unit=frame -- the offline SSL upper bound both founding papers
compare against (shuffled frames, two views, ntxent).  Utterance-pair IID
(original CVCL) needs the utterance->frame mapping that admission components
own in streaming; add unit=pair when a project needs it.
"""

from __future__ import annotations

from typing import cast

import torch

from egolearn.core.context import Context
from egolearn.data.loading import Loader
from egolearn.data.sources.base import DenseFrameChannel, Stream

from .profiling import StageTimer
from .solver import ProgressCosine, apply_lr


class OfflineDriver:
    def __init__(
        self,
        stream: Stream,
        loader: Loader,
        model,
        losses: list,
        optimizer,
        schedule: ProgressCosine | None,
        ctx: Context,
        slot: str = "frames",
        n: int = 256,
        views: list[str] | None = None,
        unit: str = "frame",
        epochs: int | None = None,
        max_steps: int | None = None,
        subsample: int = 1,
        hooks: list | None = None,
        grad_accumulation: int = 1,
    ):
        if unit != "frame":
            raise ValueError(
                "offline mode v1 supports unit=frame only (the offline SSL "
                "upper bound); unit=pair needs the utterance->frame mapping "
                "that admission components own in streaming"
            )
        if not views:
            raise ValueError("offline driver: `batch.views` must name at least one view")
        if epochs is None and max_steps is None:
            raise ValueError("offline driver: set `epochs` and/or `max_steps`")
        self.stream = stream
        self.loader = loader
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.schedule = schedule
        self.ctx = ctx
        self.slot = slot
        self.n = int(n)
        self.views = list(views)
        self.epochs = None if epochs is None else int(epochs)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.subsample = max(1, int(subsample))
        self.hooks = hooks or []
        self.grad_accumulation = max(1, int(grad_accumulation))
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.timer = StageTimer()
        loader.timer = self.timer  # data.load measurements
        self.memories: dict = {}  # hook compatibility (nothing stored)

    # ------------------------------------------------------------------
    def _build_index(self):
        """Enumerate every (subsampled) dense-channel frame ref up front --
        the conventional 'dataset as a list' view of the same stream."""
        refs = []
        for session in self.stream.sessions():
            for ch in session.channels.values():
                if getattr(ch, "kind", None) != "dense":
                    continue
                dense = cast(DenseFrameChannel, ch)
                for i in range(0, len(dense), self.subsample):
                    refs.append(dense.ref_at(i))
        return refs

    # ------------------------------------------------------------------
    def _compose(self, refs) -> dict:
        views = torch.stack([self.loader.frame_views(r, 0, self.views) for r in refs], dim=0)
        return {self.slot: {"views": views, "meta": list(refs), "weights": None}}

    def _train_step(self, batch: dict) -> bool:
        logs: dict[str, float] = {}
        total = None
        with self.timer.measure("model.forward"):
            self.model.precompute_feats(batch, self.ctx.device)
            for loss_fn in self.losses:
                values = loss_fn.compute(batch, self.model, self.ctx)
                for lname, lval in values.items():
                    logs[f"{loss_fn.name}/{lname}"] = float(lval.detach())
                    term = loss_fn.weight * lval
                    total = term if total is None else total + term
        if total is None or not torch.isfinite(total):
            print(
                f"[offline] skipping step {self.ctx.clock.step}: non-finite or empty loss",
                flush=True,
            )
            return False
        logs["loss"] = float(total.detach())

        with self.timer.measure("model.backward"):
            (total / self.grad_accumulation).backward()
        if (self.ctx.clock.step + 1) % self.grad_accumulation == 0:
            with self.timer.measure("optim.step"):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        for loss_fn in self.losses:
            loss_fn.on_step_end(self.model, self.ctx)

        self.ctx.clock.step += 1
        self.ctx.clock.admitted += 1  # offline: both clocks tick per step
        if self.schedule is not None:
            apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
        for h in self.hooks:
            h.on_step(self, logs)
        return True

    # ------------------------------------------------------------------
    def fit(self) -> None:
        self.model.to(self.ctx.device)
        self.model.train()

        refs = self._build_index()
        if len(refs) < self.n:
            raise RuntimeError(
                f"offline driver: dataset index has {len(refs)} frames < batch n={self.n}"
            )
        batches_per_epoch = len(refs) // self.n  # drop last, standard
        planned = self.max_steps
        if self.epochs is not None:
            by_epochs = self.epochs * batches_per_epoch
            planned = by_epochs if planned is None else min(planned, by_epochs)
        if planned is None:
            raise ValueError("offline driver needs `epochs` or `max_steps`")
        # offline progress clock = fraction of planned gradient steps
        self.ctx.clock.total_observations = planned
        print(
            f"[offline] index: {len(refs)} frames, "
            f"{batches_per_epoch} batches/epoch, {planned} planned steps",
            flush=True,
        )

        if self.ctx.clock.step > 0:
            print(f"[offline] resuming at step {self.ctx.clock.step}/{planned}", flush=True)
        while self.ctx.clock.step < planned:
            # per-epoch DERIVED shuffle: the order for epoch e depends only on
            # (seed, e), so a resumed run regenerates it exactly and continues
            # from step % batches_per_epoch
            epoch = self.ctx.clock.step // batches_per_epoch
            order = list(range(len(refs)))
            self.ctx.rng(f"offline.shuffle.epoch{epoch}").shuffle(order)
            for b in range(self.ctx.clock.step % batches_per_epoch, batches_per_epoch):
                if self.ctx.clock.step >= planned:
                    break
                picked = [refs[order[b * self.n + j]] for j in range(self.n)]
                with self.timer.measure("batch.compose"):
                    batch = self._compose(picked)
                self._train_step(batch)
        for h in self.hooks:
            h.on_end(self)
        print(f"[offline] done: steps={self.ctx.clock.step}", flush=True)
        print(self.timer.summary(), flush=True)
