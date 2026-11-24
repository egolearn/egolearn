"""LockstepDriver: the main training loop for streaming runs.

The driver (the object that owns the training loop) runs a fixed loop: pull new data from the
stream, then run k training steps, repeat.  One cycle:

  1. ADVANCE  -- read observations (frames, transcript lines) from the stream in time order,
     feed each one to every admission component subscribed to its channel (admission components
     decide what enters memory), collect the events they finalize, and route each event into a
     memory table (`routes: {schema: memory}`).  How much stream one advance covers is
     configurable (the `advance` setting): either a fixed number of observations (Memory
     Storyboard: 64 frames) or a number of emitted events of one schema (BabyCL: 1 finalized
     VisualSegment).
  2. TRAIN    -- run `rate.steps_per_admit` gradient steps.  This setting controls how many
     training steps run per unit of new stream data, which turns "compute-matched baseline"
     into a config flag.  With `rate.resample=true` each step draws a fresh batch from memory
     (as in BabyCL); with false the same batch is reused (Memory Storyboard's
     num_grads).
  3. Repeat until the stream runs out.  Training ends when the stream ends, never at a
     precomputed step count.

Warmup is a phase that only fills memory and runs no gradient steps:
`warmup: {tier_full: "memory.tier"}` (both reproductions fill a buffer before the first
gradient step) or `warmup: {admitted: N}`.

Every run carries a StageTimer (see training/profiling.py) that records how long each stage
actually takes -- stream advance, data loading, batch composition, forward, backward.

Mixed precision is opt-in (`amp: {dtype: float16, fp32_slots: [...]}`): encoder forwards and
losses run under autocast, except the forwards of the named slots and every loss that reads
one, which stay in float32.  float16 also scales the loss (GradScaler).  BabyCL's original ran
the visual branch in fp16 and forced the cross-modal branch to fp32 exactly this way.

`max_steps` caps gradient steps exactly, also inside a quantum; resuming the run finishes that
quantum before the next advance.

V1 scope decisions (design review): single process, one fixed synchronous loop, deterministic;
no async ingestion.  Anything that does not fit this loop should subclass or replace the
driver -- the loop is small on purpose.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import torch

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.batching.composer import Composer
from egolearn.data.batching.recipe import Recipe
from egolearn.data.sources.base import Observation, Stream
from egolearn.memory.admission.base import EventBuilder
from egolearn.memory.table import TieredMemory
from egolearn.model.losses.base import own_forward_slots
from egolearn.run_record import recorded_fit

from .profiling import StageTimer
from .solver import ProgressCosine, apply_lr

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _parse_amp(amp: dict | None) -> tuple[torch.dtype | None, frozenset[str]]:
    """`driver.amp` -> (autocast dtype or None, slots kept in float32)."""
    if not amp or not amp.get("enabled", True):
        return None, frozenset()
    unknown = set(amp) - {"enabled", "dtype", "fp32_slots"}
    if unknown:
        raise ValueError(f"driver.amp: unknown keys {sorted(unknown)}")
    dtype = str(amp.get("dtype", "float16"))
    if dtype not in _AMP_DTYPES:
        raise ValueError(f"driver.amp.dtype must be one of {sorted(_AMP_DTYPES)}, got {dtype!r}")
    return _AMP_DTYPES[dtype], frozenset(amp.get("fp32_slots") or ())


class LockstepDriver:
    def __init__(
        self,
        stream: Stream,
        admission: list[EventBuilder],
        memories: dict[str, TieredMemory],
        routes: dict[str, str],
        recipe: Recipe,
        composer: Composer,
        model,
        losses: list,
        optimizer,
        schedule: ProgressCosine | None,
        ctx: Context,
        advance: dict | None = None,
        rate: dict | None = None,
        warmup: dict | None = None,
        hooks: list | None = None,
        max_steps: int | None = None,
        grad_accumulation: int = 1,
        stream_head_size: int = 512,
        prefetch: bool = False,
        strict_batches: bool = False,
        amp: dict | None = None,
    ):
        self.run_config: dict | None = None
        self.run_inputs: dict | None = None
        self.stream = stream
        self.admission = admission
        self.memories = memories
        self.routes = routes
        self.recipe = recipe
        self.composer = composer
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.schedule = schedule
        self.ctx = ctx
        self.advance = advance or {"kind": "observations", "n": 64}
        self.rate = {"steps_per_admit": 1, "resample": True, **(rate or {})}
        self.warmup = warmup or {}
        self.hooks = hooks or []
        self.skipped_steps = 0
        self.nonfinite_steps = 0
        self.max_steps = max_steps
        self.grad_accumulation = max(1, int(grad_accumulation))
        # Prefetch: compose the NEXT batch (CPU: sampling, decode, augment)
        # while the current one trains on the GPU.  Valid because sampling
        # state -- memory contents, label stores, stream head -- only changes
        # when the stream advances, never between the steps of one quantum
        # (components that break that rule declare mutates_sampling_state,
        # and the build refuses to prefetch; see pipeline.build).  Batch
        # randomness derives from the batch's sequence number, so prefetch
        # on/off is a pure throughput knob: same batches either way.
        self.prefetch = bool(prefetch)
        self.strict_batches = bool(strict_batches)
        self._prefetch_pool = None
        if self.prefetch:
            from concurrent.futures import ThreadPoolExecutor

            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._skip_feats = own_forward_slots(losses)
        self.amp_dtype, self.fp32_slots = _parse_amp(amp)
        self.scaler = (
            torch.amp.GradScaler(torch.device(ctx.device).type)
            if self.amp_dtype is torch.float16
            else None
        )
        self.timer = StageTimer()
        if getattr(composer, "loader", None) is not None:
            composer.loader.timer = self.timer  # data.load measurements
        # most recent admitted events per schema, for stream_head slots
        self.stream_head: dict[str, deque] = {}
        self._stream_head_size = stream_head_size
        self._obs_iter: Iterator[Observation] | None = None
        self._exhausted = False
        self._current_sid: str | None = None
        # events emitted per schema, including those flushed at a session end
        # (routed inside _observations, not returned by _step_stream)
        self._emitted_by_schema: dict[str, int] = {}
        # resume: while replaying, events refresh stream_head but are NOT
        # admitted to memory (memories come from the snapshot)
        self._replaying = False

    # ------------------------------------------------------------------
    # stream plumbing
    # ------------------------------------------------------------------
    def _observations(self) -> Iterator[Observation]:
        for session in self.stream.sessions():
            self._current_sid = session.sid
            for b in self.admission:
                b.begin_session(session)
            yield from session.observations()
            for b in self.admission:
                b.end_session()
            # flush: events finalized by end_session must still be routed
            self._route_events(self._poll_admission())

    def _poll_admission(self) -> list[dict]:
        events: list[dict] = []
        for b in self.admission:
            events.extend(b.poll())
        return events

    def _route_events(self, events: list[dict]) -> int:
        routed = 0
        for ev in events:
            schema = ev[K.EVENT_SCHEMA]
            if not self._replaying:
                mem = self.routes.get(schema)
                if mem is not None:
                    self.memories[mem].admit(ev, self.ctx)
                    routed += 1
                self.ctx.clock.events_emitted += 1
                self._emitted_by_schema[schema] = self._emitted_by_schema.get(schema, 0) + 1
            head = self.stream_head.setdefault(schema, deque(maxlen=self._stream_head_size))
            head.append(ev)
        return routed

    def _step_stream(self) -> list[dict] | None:
        """Consume one observation; return newly routed events (or None at end)."""
        if self._obs_iter is None:
            self._obs_iter = self._observations()
        try:
            obs = next(self._obs_iter)
        except StopIteration:
            self._exhausted = True
            return None
        for b in self.admission:
            if obs.channel in b.subscribes:
                b.ingest(obs)
        self.ctx.clock.admitted += 1
        return self._poll_and_route()

    def _poll_and_route(self) -> list[dict]:
        events = self._poll_admission()
        self._route_events(events)
        return events

    def _advance_once(self) -> bool:
        """One advance quantum.  Returns True if anything new arrived -- a
        partial quantum at the end of the stream still counts, so the tail
        of the stream (and events flushed at the final session end) get the
        same training as any other data.  False means the stream is done
        and nothing new was admitted."""
        events_before = self.ctx.clock.events_emitted
        kind = self.advance["kind"]
        if kind == "observations":
            consumed = 0
            while consumed < int(self.advance["n"]) and self._step_stream() is not None:
                consumed += 1
        elif kind == "events":
            # events flushed at a session end count too: they end the quantum
            # like any other, so a session's last segment gets its own steps
            target_schema = self.advance["schema"]
            want = int(self.advance.get("n", 1))
            start = self._emitted_by_schema.get(target_schema, 0)
            consumed = 0
            while self._emitted_by_schema.get(target_schema, 0) - start < want:
                if self._step_stream() is None:
                    break
                consumed += 1
            got = self._emitted_by_schema.get(target_schema, 0) - start
        else:
            raise ValueError(f"unknown advance kind {kind!r}")
        if kind == "events" and self.advance.get("require_events", False):
            return got > 0
        return consumed > 0 or self.ctx.clock.events_emitted > events_before

    # ------------------------------------------------------------------
    # resume
    # ------------------------------------------------------------------
    def _resume_replay(self, exhausted: bool = False) -> None:
        """Rebuild the admission components' internal state after a snapshot restore.

        The stream and the admission algorithms are deterministic, so replaying the first
        `clock.admitted` observations reconstructs every sliding window, every watermark
        position (how far past a segment's end the stream must be before the segment is
        finalized), and the global-id handover, with no extra bookkeeping.  During replay,
        emitted events refresh the stream_head but are NOT admitted to memory (memories come
        from the snapshot).  The replay consumes admission RNGs exactly like the original
        pass did, starting from fresh seed-derived state; the snapshot's saved RNG states,
        label stores and memories are all restored after it (see utils.snapshot).

        Cost: one pass over the already-consumed stream, including any frozen-embedder work the
        admission components do -- proportional to the resume position.  A future version could
        store the emitted events and skip the replay entirely; that is deferred for now.
        """
        target = self.ctx.clock.admitted
        if target <= 0:
            return
        if self._obs_iter is not None:
            raise RuntimeError("_resume_replay must run before the driver starts")
        events_before = self.ctx.clock.events_emitted
        self.ctx.clock.admitted = 0
        self._replaying = True
        try:
            for builder in self.admission:
                builder.set_replaying(True)
            while self.ctx.clock.admitted < target:
                if self._step_stream() is None:
                    print(
                        "[driver] resume: stream exhausted during replay "
                        f"({self.ctx.clock.admitted}/{target})"
                    )
                    break
            if exhausted:
                # the original run had consumed the stream to the end,
                # including the final session-end flush -- do the same here
                # (still replaying, so nothing is admitted twice)
                while self._step_stream() is not None:
                    pass
        finally:
            self._replaying = False
            for builder in self.admission:
                builder.set_replaying(False)
            self.ctx.clock.events_emitted = events_before
        print(
            f"[driver] resume: replayed {self.ctx.clock.admitted} "
            "observations, admission state rebuilt",
            flush=True,
        )

    # ------------------------------------------------------------------
    # warmup
    # ------------------------------------------------------------------
    def _warmup_done(self) -> bool:
        if "events_emitted" in self.warmup:
            return self.ctx.clock.events_emitted >= int(self.warmup["events_emitted"])
        if "tier_full" in self.warmup:
            mem_name, tier = self.warmup["tier_full"].split(".", 1)
            return self.memories[mem_name].tier_full(tier)
        if "admitted" in self.warmup:
            return self.ctx.clock.admitted >= int(self.warmup["admitted"])
        return True

    def _run_warmup(self) -> None:
        while not self._warmup_done():
            if self._step_stream() is None:
                print("[driver] stream exhausted during warmup")
                return
        stats = " | ".join(m.stats() for m in self.memories.values())
        print(f"[driver] warmup done: admitted={self.ctx.clock.admitted}  {stats}", flush=True)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _train_steps(self) -> None:
        """One quantum: `rate.steps_per_admit` iterations after an advance."""
        self.ctx.clock.quantum_left = int(self.rate["steps_per_admit"])
        self._run_quantum()

    def _run_quantum(self) -> None:
        """Run the iterations the last advance is still owed
        (`clock.quantum_left`): a full quantum from `_train_steps`, or the
        rest of one that a snapshot interrupted."""
        clock = self.ctx.clock
        k = int(self.rate["steps_per_admit"])
        resample = bool(self.rate["resample"])
        batch = None
        pending = None  # (future, seq): the next batch, composing ahead
        while clock.quantum_left > 0:
            if self.max_steps is not None and clock.step >= self.max_steps:
                break
            if batch is None or resample:
                if pending is not None:
                    fut, seq = pending
                    pending = None
                    with self.timer.measure("batch.compose"):
                        batch = fut.result()
                    # committed only now that the batch is handed to training,
                    # so a snapshot taken mid-flight stays consistent (the
                    # resumed run rebuilds the identical batch from this seq)
                    clock.compose_seq = seq + 1
                else:
                    # resuming inside a quantum that reuses one batch: rebuild
                    # the batch it composed, without allocating a new seq
                    reused = None if resample or clock.quantum_left == k else clock.compose_seq - 1
                    with self.timer.measure("batch.compose"):
                        batch = self.composer.compose(
                            self.recipe,
                            self.memories,
                            {s: list(d) for s, d in self.stream_head.items()},
                            self.ctx,
                            reused,
                        )
            if batch is None:
                clock.quantum_left = 0
                if self.strict_batches:
                    raise RuntimeError("streaming batch is missing after admission")
                return  # strict slot unfilled and no fallback: skip quietly
            clock.quantum_left -= 1
            if (
                self._prefetch_pool is not None
                and resample
                and clock.quantum_left > 0
                and (self.max_steps is None or clock.step + 1 < self.max_steps)
            ):
                # never across the quantum boundary: the next advance changes
                # what the composer would sample
                seq = self.ctx.clock.compose_seq
                head = {s: list(d) for s, d in self.stream_head.items()}
                pending = (
                    self._prefetch_pool.submit(
                        self.composer.compose, self.recipe, self.memories, head, self.ctx, seq
                    ),
                    seq,
                )
            # Actual consumed tensor sizes, including a fallback recipe if used.
            self.last_batch_counts = {"clips": 0, "frames": 0}
            for slot in batch.values():
                views = slot[K.SLOT_VIEWS]
                n = views.shape[0]
                video = slot[K.SLOT_MODALITY] == "video"
                self.last_batch_counts["clips"] += n if video else 0
                self.last_batch_counts["frames"] += n * (views.shape[2] if video else 1)
            logs: dict[str, float] = {}
            total = None
            with self.timer.measure("model.forward"):
                # each encoder runs once per slot/view; losses share the features.
                # Under amp the autocast slots go first, then the fp32 ones --
                # recipe order is kept, so BatchNorm sees the same sequence.
                with self._autocast(True):
                    self.model.precompute_feats(
                        batch, self.ctx.device, skip=self._skip_feats | self.fp32_slots
                    )
                if self.fp32_slots:
                    self.model.precompute_feats(
                        batch,
                        self.ctx.device,
                        skip=self._skip_feats | (frozenset(batch) - self.fp32_slots),
                    )
                for loss_fn in self.losses:
                    # A fallback recipe may drop slots (BabyCL: multimodal slot
                    # starved -> all-visual batch). Losses whose slots are
                    # absent contribute nothing this step.
                    if any(s not in batch for s in loss_fn.requires_slots):
                        continue
                    fp32 = any(s in self.fp32_slots for s in loss_fn.requires_slots)
                    with self._autocast(not fp32):
                        values = loss_fn.compute(batch, self.model, self.ctx)
                    for lname, lval in values.items():
                        logs[f"{loss_fn.name}/{lname}"] = float(lval.detach())
                        term = loss_fn.weight * lval
                        total = term if total is None else total + term
            if total is None or not torch.isfinite(total):
                self.skipped_steps += 1
                self.nonfinite_steps += int(total is not None)
                for hook in self.hooks:
                    if callback := getattr(hook, "on_skip", None):
                        callback(self, nonfinite=total is not None)
                print(
                    f"[driver] skipping step {self.ctx.clock.step}: non-finite or empty loss",
                    flush=True,
                )
                continue
            logs["loss"] = float(total.detach())

            with self.timer.measure("model.backward"):
                scaled = total / self.grad_accumulation
                (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
            if (self.ctx.clock.step + 1) % self.grad_accumulation == 0:
                # schedule for THIS step is applied before stepping (the
                # originals adjust then step, so warmup covers step 1 too)
                if self.schedule is not None:
                    apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
                with self.timer.measure("optim.step"):
                    self._optimizer_step()
            for loss_fn in self.losses:
                loss_fn.on_step_end(self.model, self.ctx)

            self.ctx.clock.step += 1
            for h in self.hooks:
                h.on_step(self, logs)

    def _flush_grad_accumulation(self) -> None:
        """The loop steps the optimizer every `grad_accumulation` backwards;
        when the stream and its final quantum end between optimizer steps,
        apply what accumulated instead of silently dropping those gradients.

        Deliberately not done on a max_steps stop: a capped run may be
        snapshotted and continued, and the continuation must fold these
        micro-steps into one full window exactly like an uninterrupted run
        (pending gradients travel inside the snapshot).  A capped run that
        is never continued leaves them unapplied, and says so.

        Every optimizer step clears the gradients, so none left means the
        window was already applied (a snapshot taken after this flush) or
        every micro-step in it was skipped: nothing to do."""
        pending = self.grad_accumulation > 1 and self.ctx.clock.step % self.grad_accumulation != 0
        if not pending or all(p.grad is None for p in self.model.parameters()):
            return
        if not self._exhausted or self.ctx.clock.quantum_left > 0:
            n = self.ctx.clock.step % self.grad_accumulation
            print(
                f"[driver] stopped at max_steps with {n} accumulated micro-step(s) "
                "pending; they apply on resume (or align max_steps to "
                "grad_accumulation)",
                flush=True,
            )
            return
        if self.scaler is not None:
            # a restored scaler sets its scale lazily, on the next scale() call;
            # gradients restored from a snapshot reach here without one
            self.scaler.scale(torch.zeros((), device=self.ctx.device))
        if self.schedule is not None:
            apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
        self._optimizer_step()

    def _autocast(self, enabled: bool):
        device_type = torch.device(self.ctx.device).type
        return torch.autocast(
            device_type, dtype=self.amp_dtype, enabled=enabled and self.amp_dtype is not None
        )

    def _optimizer_step(self) -> None:
        """One optimizer step.  A float16 run unscales first and skips the step
        when the gradients overflowed (GradScaler); the schedule and the step
        clock still advance, as in the original's LambdaLR loop."""
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    @recorded_fit
    def fit(self) -> None:
        self.model.to(self.ctx.device)
        self.model.train()
        self._run_warmup()
        # a snapshot taken inside a quantum: finish it before advancing again
        self._run_quantum()
        while not self._exhausted:
            if self.max_steps is not None and self.ctx.clock.step >= self.max_steps:
                break
            with self.timer.measure("stream.advance"):
                advanced = self._advance_once()
            if not advanced:
                break
            for h in self.hooks:
                h.on_admit(self, len(self.stream_head))
            self._train_steps()
        self._flush_grad_accumulation()
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=True)
        self.composer.loader.close()
        for h in self.hooks:
            h.on_end(self)
        print(
            f"[driver] done: steps={self.ctx.clock.step} "
            f"admitted={self.ctx.clock.admitted} "
            f"events={self.ctx.clock.events_emitted}",
            flush=True,
        )
        print(self.timer.summary(), flush=True)
