"""Loader: turns an ItemRef into tensors, as late as possible.

Everything else passes around refs (lightweight pointers into the source
data) plus metadata; this is the only place images are decoded and augmented.
Usually that happens while a batch is being built, but it is not the only
caller: an admission component whose algorithm needs to look at image content
pulls frames through the Loader too (see EmbeddingProvider -- BabyCL's
segmenter embeds every 60th frame this way).  Alternative loaders (embedding
caches, generative memory) implement this same small interface, so replacing
how items become tensors means implementing these few methods and touching
nothing else.

If a StageTimer is attached (`loader.timer`, set by the training driver),
decode+augment time is recorded under "data.load" -- this is the number that
answers "how much latency do the data abstractions cost?".

Parallelism: `frame_views_batch` can fan decode+augment out to a pool of
worker processes (`loader.workers` in the config; 0 = in-process).  The
worker count is a pure throughput knob and never changes what a run
computes, because every frame's augmentation randomness comes from its own
derived seed (Context.derive_seed) rather than from a shared RNG stream --
the same seeds give the same tensors whether 0 or 8 workers ran them.
Streaming order stays intact: only stage two (refs -> tensors) runs in
parallel; everything that decides WHICH refs to load is still sequential.
Clips keep the serial path for now (BabyCL/MS batches are frames).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as TF

from .sources.base import Stream
from .views import ViewBank

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from egolearn.training.profiling import StageTimer

# WorkItem: what one output row of frame_views_batch needs.
# (ref, frame index within the ref, view names, augmentation seed)
WorkItem = tuple

_WORKER: dict = {}  # per-process state, filled by _pool_init in each worker


def _pool_init(stream: Stream, view_specs: dict) -> None:
    torch.set_num_threads(1)
    _WORKER["stream"] = stream
    _WORKER["views"] = ViewBank(view_specs)


def _pool_task(item: WorkItem) -> torch.Tensor:
    ref, index, view_names, seed = item
    # worker processes own their RNGs, so plain seeding is race-free here
    torch.manual_seed(seed)
    random.seed(seed)
    pil = _WORKER["stream"].open_frame(ref, index)
    return _WORKER["views"].apply(pil, view_names)


class Loader:
    def __init__(self, stream: Stream, views: ViewBank, workers: int = 0):
        self.stream = stream
        self.views = views
        self.workers = int(workers)
        self._pool: ProcessPoolExecutor | None = None
        # optional StageTimer (egolearn.training.profiling); the driver attaches it
        self.timer: StageTimer | None = None

    # ---- worker pool --------------------------------------------------------
    def _ensure_pool(self):
        if self._pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # spawn: safe next to CUDA in the parent; workers rebuild the
            # view pipelines from specs (closures do not pickle)
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_pool_init,
                initargs=(self.stream, self.views.specs),
            )
        return self._pool

    def close(self) -> None:
        """Shut the worker pool down (the drivers call this when a run ends)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # ---- frame selection --------------------------------------------------
    def pick_frame_indices(self, ref, k: int, rng: random.Random) -> list[int]:
        """Choose k frame indices inside `ref`.

        k == 1: one index drawn uniformly at random from the interval (this
                is CVCL's pair sampling).
        k > 1 : split the interval into k equal bins and draw one index per
                bin.  The details are bit-faithful to BabyCL's online segment
                gather (_gather_segments_online): integer bin edges computed
                by truncation (np.linspace(..., dtype=int)), an INCLUSIVE
                randint within each bin, empty bins fall back to drawing from
                the whole interval, and every pick is clamped to [0, n-1].
        """
        n = self.stream.num_frames(ref)
        if n <= 0:
            raise ValueError(f"empty ref {ref!r}")
        if k == 1:
            return [rng.randrange(n)]
        edges = [int(i * n / k) for i in range(k + 1)]
        out = []
        for i in range(k):
            a, b = edges[i], edges[i + 1]
            if b <= a:
                a, b = 0, max(1, n)
            choice = rng.randint(a, b) if b > a else a
            out.append(min(max(choice, 0), n - 1))
        return out

    def pick_clip_start(self, ref, clip_len: int, stride: int, rng: random.Random) -> int:
        """Pick a uniform-random start index for a contiguous window of
        clip_len frames spaced `stride` apart inside `ref`.  Returns 0 when
        the ref is shorter than the window."""
        n = self.stream.num_frames(ref)
        if n <= 0:
            raise ValueError(f"empty ref {ref!r}")
        span = (clip_len - 1) * stride + 1
        return rng.randrange(n - span + 1) if n > span else 0

    # ---- loading ------------------------------------------------------------
    def frame_views(self, ref, index: int, view_names: list[str], seed: int | None = None):
        """Decode one frame of `ref` and apply the named views (augmentation
        pipelines), returning a [V,C,H,W] tensor with one row per view.

        With a `seed`, the augmentation randomness is drawn from that seed
        inside a forked RNG scope instead of from the global stream -- the
        contract that makes worker-pool loading reproducible."""
        if self.timer is not None:
            with self.timer.measure("data.load"):
                return self._one_frame(ref, index, view_names, seed)
        return self._one_frame(ref, index, view_names, seed)

    def _one_frame(self, ref, index: int, view_names: list[str], seed: int | None):
        pil = self.stream.open_frame(ref, index)
        if seed is None:
            return self.views.apply(pil, view_names)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            py_state = random.getstate()
            random.seed(seed)
            try:
                return self.views.apply(pil, view_names)
            finally:
                random.setstate(py_state)

    def frame_views_batch(self, items: list[WorkItem]) -> list[torch.Tensor]:
        """Load many frames, one [V,C,H,W] tensor per item, in item order.

        items: (ref, frame_index, view_names, seed) tuples.  workers=0 loads
        in-process; otherwise decode+augment fan out to the process pool.
        Same items, same tensors, either way."""
        if not items:
            return []
        if self.workers <= 0:
            return [self.frame_views(r, i, v, seed=s) for r, i, v, s in items]
        pool = self._ensure_pool()
        chunk = max(1, len(items) // (self.workers * 4))
        if self.timer is not None:
            with self.timer.measure("data.load"):
                return list(pool.map(_pool_task, items, chunksize=chunk))
        return list(pool.map(_pool_task, items, chunksize=chunk))

    def clip_views(
        self, ref, view_names: list[str], clip_len: int, stride: int, rng: random.Random
    ) -> torch.Tensor:
        """Decode a contiguous clip from `ref` and apply the named views,
        returning a [V,T,C,H,W] tensor.

        A clip is T=clip_len frames taken every `stride` frames from a random
        start inside the ref; indices past the end clamp to the last frame,
        so a short ref yields a clip padded by repeating that frame (the
        common video convention).  Augmentation is temporally consistent:
        each clip view pipeline runs once on the whole [T,C,H,W] tensor, so
        all T frames get the same random crop etc. (see views.apply_clip).

        `view_names` must name clip-domain views (clip_simclr,
        clip_center_crop, ...).  Frames are decoded one by one through
        stream.open_frame; batching the decode through
        VideoDecoder.read_window is an optimization left for the Poodle
        integration.
        """

        def _load() -> torch.Tensor:
            n = self.stream.num_frames(ref)
            start = self.pick_clip_start(ref, clip_len, stride, rng)
            idx = [min(start + j * stride, n - 1) for j in range(clip_len)]
            clip = torch.stack([TF.to_tensor(self.stream.open_frame(ref, i)) for i in idx])
            return clip  # [T,C,H,W]

        if self.timer is not None:
            with self.timer.measure("data.load"):
                clip = _load()
                return self.views.apply_clip(clip, view_names)
        return self.views.apply_clip(_load(), view_names)
