"""OfflineDriver: conventional IID training, a second training mode alongside streaming.

Design decision (2026-08 working group): IID and streaming are two separate, parallel training
modes, not one degenerate config of the same loop.  Offline IID is the completely conventional
recipe -- index the whole dataset up front, shuffle, train in epochs -- with **no memory tables
and no admission**; the streaming driver's control over how many training steps run per unit of
new stream data does not apply here.  It lives inside the framework (instead of borrowing
another repo's implementation) because it is the upper bound every streaming method is compared
against.

Shared with the streaming path: sources/decoding/Loader/views (the whole data plane),
model/losses/solver, hooks/probes, and the per-stage timing report.  Only the loop differs.
Config switch:

    driver:
      mode: offline                       # default: streaming
      batch: {slot: frames, n: 256, views: [v1, v2]}
      epochs: 10                          # and/or max_steps
      subsample: 1                        # take every k-th frame into the index

Units: `batch.unit: frame` (default) indexes single frames -- the offline SSL
upper bound both founding papers compare against (shuffled frames, two views,
ntxent).  `batch.unit: clip` indexes non-overlapping clip_len x clip_stride
windows instead -- the conventional video pretraining epoch (V-JEPA /
VideoMAE: shuffled clips, one view, a masked-prediction loss); `subsample`
then thins windows the way it thins frames.  Utterance-pair IID (original
CVCL) needs the utterance->frame mapping that admission components own in
streaming; add unit=pair when a project needs it.

Clocks: in streaming runs, progress = the fraction of the stream consumed so far, used instead
of epochs.  In offline mode `progress` instead means the fraction of planned gradient steps
(epochs are conventional here; the "no epochs" rule applies to the streaming loop only).
Epoch e's shuffle order comes from a fresh RNG derived from (seed, epoch) each time it
is needed (ctx.fresh_rng), never cached or snapshotted, so runs are deterministic AND
resumable:
`load_snapshot` restores the clock and fit() continues from step % batches_per_epoch of the
regenerated epoch order.
"""

from __future__ import annotations

from typing import cast

import torch

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.batching.recipe import modality_for_unit
from egolearn.data.loading import Loader
from egolearn.data.sources.base import DenseFrameChannel, Stream
from egolearn.model.losses.base import own_forward_slots

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
        clip_len: int = 16,
        clip_stride: int = 1,
        epochs: int | None = None,
        max_steps: int | None = None,
        subsample: int = 1,
        hooks: list | None = None,
        grad_accumulation: int = 1,
        prefetch: bool = False,
    ):
        if unit not in ("frame", "clip"):
            raise ValueError(
                "offline mode supports unit=frame or unit=clip; unit=pair "
                "needs the utterance->frame mapping that admission "
                "components own in streaming"
            )
        if unit == "clip" and (clip_len < 1 or clip_stride < 1):
            raise ValueError("offline driver: clip_len/clip_stride must be >= 1")
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
        self.unit = unit
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride)
        self.n = int(n)
        self.views = list(views)
        self.epochs = None if epochs is None else int(epochs)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.subsample = max(1, int(subsample))
        self.hooks = hooks or []
        self.grad_accumulation = max(1, int(grad_accumulation))
        self.prefetch = bool(prefetch)
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._skip_feats = own_forward_slots(losses)
        self.timer = StageTimer()
        loader.timer = self.timer  # data.load measurements
        self.memories: dict = {}  # hook compatibility (nothing stored)

    # ------------------------------------------------------------------
    def _build_index(self):
        """List every (subsampled) item ref up front, turning the stream into
        the conventional 'dataset as a list' that IID training expects:
        unit=frame lists dense-channel frames, unit=clip lists non-overlapping
        clip windows."""
        refs = []
        span = (self.clip_len - 1) * self.clip_stride + 1
        for session in self.stream.sessions():
            for ch in session.channels.values():
                if getattr(ch, "kind", None) != "dense":
                    continue
                dense = cast(DenseFrameChannel, ch)
                if self.unit == "clip":
                    for j, i0 in enumerate(range(0, len(dense) - span + 1, span)):
                        if j % self.subsample == 0:
                            refs.append(dense.interval_ref(i0, i0 + span))
                else:
                    for i in range(0, len(dense), self.subsample):
                        refs.append(dense.ref_at(i))
        return refs

    # ------------------------------------------------------------------
    def _pick(self, refs, order, b: int) -> list:
        return [refs[order[b * self.n + j]] for j in range(self.n)]

    def _compose(self, refs, seq: int | None = None) -> dict:
        """Batch randomness derives from the batch's sequence number (see
        Composer.compose for the identical contract): without `seq` it is
        taken from the clock and committed immediately; the prefetch path
        allocates it in the training thread and commits on consumption."""
        if seq is None:
            seq = self.ctx.clock.compose_seq
            self.ctx.clock.compose_seq = seq + 1
        if self.unit == "clip":
            # index windows are exact-span, so the start pick consumes no
            # randomness; augmentation comes from each item's derived seed
            rng = self.ctx.fresh_rng(f"compose.{seq}")
            work: list[tuple] = [
                (
                    "clip",
                    r,
                    self.loader.pick_clip_start(r, self.clip_len, self.clip_stride, rng),
                    self.views,
                    self.clip_len,
                    self.clip_stride,
                    self.ctx.derive_seed(f"aug:{seq}:{self.slot}:{i}"),
                )
                for i, r in enumerate(refs)
            ]
        else:
            work = [
                ("frame", r, 0, self.views, self.ctx.derive_seed(f"aug:{seq}:{self.slot}:{i}"))
                for i, r in enumerate(refs)
            ]
        views = torch.stack(self.loader.views_batch(work), dim=0)
        slot = {
            K.SLOT_VIEWS: views,
            K.SLOT_MODALITY: modality_for_unit(self.unit),
            K.SLOT_META: list(refs),
            K.SLOT_WEIGHTS: None,
        }
        return {self.slot: slot}

    def _train_step(self, batch: dict) -> bool:
        logs: dict[str, float] = {}
        total = None
        with self.timer.measure("model.forward"):
            self.model.precompute_feats(batch, self.ctx.device, skip=self._skip_feats)
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
            # schedule for THIS step is applied before stepping (adjust then
            # step, so warmup covers step 1 too)
            if self.schedule is not None:
                apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
            with self.timer.measure("optim.step"):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        for loss_fn in self.losses:
            loss_fn.on_step_end(self.model, self.ctx)

        self.ctx.clock.step += 1
        self.ctx.clock.admitted += 1  # offline: both clocks tick per step
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
                f"offline driver: dataset index has {len(refs)} {self.unit}s < batch n={self.n}"
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
            f"[offline] index: {len(refs)} {self.unit}s, "
            f"{batches_per_epoch} batches/epoch, {planned} planned steps",
            flush=True,
        )

        if self.ctx.clock.step > 0:
            print(f"[offline] resuming at step {self.ctx.clock.step}/{planned}", flush=True)
        executor = None
        if self.prefetch:
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=1)
        while self.ctx.clock.step < planned:
            # per-epoch DERIVED shuffle: the order for epoch e depends only on
            # (seed, e), so a resumed run regenerates it exactly and continues
            # from step % batches_per_epoch
            epoch = self.ctx.clock.step // batches_per_epoch
            order = list(range(len(refs)))
            # fresh_rng, not rng: the cached generator would be restored
            # already-consumed on resume and shuffle a different order
            self.ctx.fresh_rng(f"offline.shuffle.epoch{epoch}").shuffle(order)
            pending = None  # (future, seq): the next batch, composing ahead
            for b in range(self.ctx.clock.step % batches_per_epoch, batches_per_epoch):
                if self.ctx.clock.step >= planned:
                    break
                if pending is not None:
                    fut, seq = pending
                    pending = None
                    with self.timer.measure("batch.compose"):
                        batch = fut.result()
                    # committed only now that the batch is handed to training,
                    # so a snapshot taken mid-flight stays consistent
                    self.ctx.clock.compose_seq = seq + 1
                else:
                    with self.timer.measure("batch.compose"):
                        batch = self._compose(self._pick(refs, order, b))
                if (
                    executor is not None
                    and b + 1 < batches_per_epoch
                    and self.ctx.clock.step + 1 < planned
                ):
                    seq = self.ctx.clock.compose_seq
                    nxt = self._pick(refs, order, b + 1)
                    pending = (executor.submit(self._compose, nxt, seq), seq)
                self._train_step(batch)
        if executor is not None:
            executor.shutdown(wait=True)
        if self.grad_accumulation > 1 and self.ctx.clock.step % self.grad_accumulation != 0:
            # apply the tail of an unfinished accumulation window
            if self.schedule is not None:
                apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        self.loader.close()
        for h in self.hooks:
            h.on_end(self)
        print(f"[offline] done: steps={self.ctx.clock.step}", flush=True)
        print(self.timer.summary(), flush=True)
