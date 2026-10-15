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
     (BabyCL's repeats_per_push); with false the same batch is reused (Memory Storyboard's
     num_grads).
  3. Repeat until the stream runs out.  Training ends when the stream ends, never at a
     precomputed step count.

Warmup is a phase that only fills memory and runs no gradient steps:
`warmup: {tier_full: "memory.tier"}` (both reproductions fill a buffer before the first
gradient step) or `warmup: {admitted: N}`.

Every run carries a StageTimer (see training/profiling.py) that records how long each stage
actually takes -- stream advance, data loading, batch composition, forward, backward.

V1 scope decisions (design review): single process, one fixed synchronous loop, deterministic;
no AMP (a known deviation from the originals); no async ingestion.  Anything that does not fit
this loop should subclass or replace the driver -- the loop is small on purpose.
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
        self._prefetch_pool = None
        if self.prefetch:
            from concurrent.futures import ThreadPoolExecutor

            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._skip_feats = own_forward_slots(losses)
        self.timer = StageTimer()
        if getattr(composer, "loader", None) is not None:
            composer.loader.timer = self.timer  # data.load measurements
        # most recent admitted events per schema, for stream_head slots
        self.stream_head: dict[str, deque] = {}
        self._stream_head_size = stream_head_size
        self._obs_iter: Iterator[Observation] | None = None
        self._exhausted = False
        self._current_sid: str | None = None
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
                b.start_session(session.sid)
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
            target_schema = self.advance["schema"]
            want = int(self.advance.get("n", 1))
            consumed = got = 0
            while got < want:
                events = self._step_stream()
                if events is None:
                    break
                consumed += 1
                got += sum(1 for e in events if e[K.EVENT_SCHEMA] == target_schema)
        else:
            raise ValueError(f"unknown advance kind {kind!r}")
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
        k = int(self.rate["steps_per_admit"])
        resample = bool(self.rate["resample"])
        batch = None
        pending = None  # (future, seq): the next batch, composing ahead
        for i in range(k):
            if batch is None or resample:
                if pending is not None:
                    fut, seq = pending
                    pending = None
                    with self.timer.measure("batch.compose"):
                        batch = fut.result()
                    # committed only now that the batch is handed to training,
                    # so a snapshot taken mid-flight stays consistent (the
                    # resumed run rebuilds the identical batch from this seq)
                    self.ctx.clock.compose_seq = seq + 1
                else:
                    with self.timer.measure("batch.compose"):
                        batch = self.composer.compose(
                            self.recipe,
                            self.memories,
                            {s: list(d) for s, d in self.stream_head.items()},
                            self.ctx,
                        )
            if batch is None:
                return  # strict slot unfilled and no fallback: skip quietly
            if self._prefetch_pool is not None and resample and i + 1 < k:
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
            logs: dict[str, float] = {}
            total = None
            with self.timer.measure("model.forward"):
                # each encoder runs once per slot/view; losses share the features
                self.model.precompute_feats(batch, self.ctx.device, skip=self._skip_feats)
                for loss_fn in self.losses:
                    # A fallback recipe may drop slots (BabyCL: multimodal slot
                    # starved -> all-visual batch). Losses whose slots are
                    # absent contribute nothing this step.
                    if any(s not in batch for s in loss_fn.requires_slots):
                        continue
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
                (total / self.grad_accumulation).backward()
            if (self.ctx.clock.step + 1) % self.grad_accumulation == 0:
                # schedule for THIS step is applied before stepping (the
                # originals adjust then step, so warmup covers step 1 too)
                if self.schedule is not None:
                    apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
                with self.timer.measure("optim.step"):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            for loss_fn in self.losses:
                loss_fn.on_step_end(self.model, self.ctx)

            self.ctx.clock.step += 1
            for h in self.hooks:
                h.on_step(self, logs)

    def _flush_grad_accumulation(self) -> None:
        """The loop steps the optimizer every `grad_accumulation` backwards;
        when the STREAM ends between optimizer steps, apply what accumulated
        instead of silently dropping those gradients.

        Deliberately not done on a max_steps stop: a capped run may be
        snapshotted and continued, and the continuation must fold these
        micro-steps into one full window exactly like an uninterrupted run
        (pending gradients travel inside the snapshot).  A capped run that
        is never continued leaves them unapplied, and says so."""
        pending = self.grad_accumulation > 1 and self.ctx.clock.step % self.grad_accumulation != 0
        if not pending:
            return
        if not self._exhausted:
            n = self.ctx.clock.step % self.grad_accumulation
            print(
                f"[driver] stopped at max_steps with {n} accumulated micro-step(s) "
                "pending; they apply on resume (or align max_steps to "
                "grad_accumulation)",
                flush=True,
            )
            return
        if self.schedule is not None:
            apply_lr(self.optimizer, self._base_lrs, self.schedule.multiplier(self.ctx))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    @recorded_fit
    def fit(self) -> None:
        self.model.to(self.ctx.device)
        self.model.train()
        self._run_warmup()
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
