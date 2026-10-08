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

Parallelism: `views_batch` can fan decode+augment out to a pool of worker
processes (`loader.workers` in the config; 0 = in-process).  The worker
count is a pure throughput knob and never changes what a run computes,
because every item's augmentation randomness comes from its own derived
seed (Context.derive_seed) rather than from a shared RNG stream -- the same
seeds give the same tensors whether 0 or 8 workers ran them.  Frames and
clips both travel through the pool; a clip item carries its start index
(picked sequentially by the caller) so the worker only decodes and
augments.  Streaming order stays intact: only stage two (refs -> tensors)
runs in parallel; everything that decides WHICH refs to load is still
sequential.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .sources.base import Stream
from .views import ViewBank

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from egolearn.training.profiling import StageTimer

# WorkItem: what one output row of views_batch needs, tagged by kind:
#   ("frame", ref, frame_index, view_names, seed)              -> [V, C, H, W]
#   ("clip", ref, start, view_names, clip_len, stride, seed)   -> [V, T, C, H, W]
# The seed pins the item's augmentation randomness; the frame index / clip
# start were already picked sequentially by the caller.
WorkItem = tuple

_WORKER: dict = {}  # per-process state, filled by _pool_init in each worker


def _pool_init(stream: Stream, view_specs: dict, imports: list[str], cache_frames: int = 0) -> None:
    import importlib
    import signal

    # Only the training process handles checkpoint-pause signals.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    torch.set_num_threads(1)
    # project components (views, ...) register themselves at import time; a
    # spawned worker starts from a bare interpreter, so the config's
    # `_imports_` must run here too before view pipelines can be rebuilt
    for mod in imports:
        importlib.import_module(mod)
    _WORKER["stream"] = stream
    _WORKER["views"] = ViewBank(view_specs)
    _WORKER["cache"] = OrderedDict()
    _WORKER["cache_frames"] = cache_frames


def _open_frame(stream: Stream, ref, index: int, cache: OrderedDict, limit: int):
    """Bounded LRU of decoded pixels, before any random augmentation."""
    if not limit:
        return stream.open_frame(ref, index)
    key = (ref.key, index)
    image = cache.pop(key, None)
    if image is None:
        image = stream.open_frame(ref, index)
    cache[key] = image
    while len(cache) > limit:
        cache.popitem(last=False)
    return image


def _decode_clip(
    stream: Stream,
    ref,
    start: int,
    clip_len: int,
    stride: int,
    cache: OrderedDict | None = None,
    cache_frames: int = 0,
) -> torch.Tensor:
    """[T, C, H, W] float clip; indices past the ref's end clamp to the last
    frame (pad-by-repetition, the common video convention)."""
    n = stream.num_frames(ref)
    idx = [min(start + j * stride, n - 1) for j in range(clip_len)]
    cache = cache if cache is not None else OrderedDict()
    frames = [_open_frame(stream, ref, i, cache, cache_frames) for i in idx]
    if frames and all(
        isinstance(frame, Image.Image) and frame.mode == "RGB" and frame.size == frames[0].size
        for frame in frames
    ):
        width, height = frames[0].size
        clip = torch.empty((clip_len, 3, height, width), dtype=torch.get_default_dtype())
        # Copy uint8 pixels straight into their final float allocation, then
        # normalize once. This equals to_tensor + stack without per-frame float
        # allocations and the subsequent full-clip copy. Cached PILs stay pristine.
        for index, frame in enumerate(frames):
            clip[index].copy_(TF.pil_to_tensor(frame))
        return clip.div_(255)
    return torch.stack([TF.to_tensor(frame) for frame in frames])


def _pool_task(item: WorkItem) -> torch.Tensor:
    # worker processes own their RNGs, so plain seeding is race-free here
    if item[0] == "clip":
        _, ref, start, view_names, clip_len, stride, seed = item
        clip = _decode_clip(
            _WORKER["stream"],
            ref,
            start,
            clip_len,
            stride,
            _WORKER["cache"],
            _WORKER["cache_frames"],
        )
        torch.manual_seed(seed)
        random.seed(seed)
        return _WORKER["views"].apply_clip(clip, view_names)
    _, ref, index, view_names, seed = item
    torch.manual_seed(seed)
    random.seed(seed)
    pil = _open_frame(_WORKER["stream"], ref, index, _WORKER["cache"], _WORKER["cache_frames"])
    if _WORKER["cache_frames"]:
        pil = pil.copy()  # frame views may mutate their input
    return _WORKER["views"].apply(pil, view_names)


def _pool_chunk(items: list[WorkItem]) -> torch.Tensor:
    """Pack homogeneous rows into one shared tensor per worker submission."""
    return torch.stack(_pool_items(items))


def _pool_items(items: list[WorkItem]) -> list[torch.Tensor]:
    return [_pool_task(item) for item in items]


def _work_key(item: WorkItem) -> tuple:
    # Views and seeds vary per update; ownership follows the decoded pixels.
    suffix = (item[4], item[5]) if item[0] == "clip" else ()
    return (item[0], item[1].key, item[2], *suffix)


class Loader:
    def __init__(
        self,
        stream: Stream,
        views: ViewBank,
        workers: int = 0,
        imports: list[str] | None = None,
        pin_memory: bool = False,
        packed: bool = False,
        cache_frames: int = 0,
        affinity: bool = False,
    ):
        self.stream = stream
        self.views = views
        self.workers = int(workers)
        self.imports = list(imports or [])  # modules workers import to register views
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.packed = bool(packed)
        self.cache_frames = int(cache_frames)
        self.affinity = bool(affinity)
        if self.workers < 0 or self.cache_frames < 0:
            raise ValueError("loader workers and cache_frames must be nonnegative")
        if (self.cache_frames or self.affinity) and self.workers <= 0:
            raise ValueError("loader cache_frames and affinity require workers")
        self._pool: ProcessPoolExecutor | None = None
        self._affinity_pools: list[ProcessPoolExecutor] = []
        self._owners: dict[tuple, int] = {}
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
                initargs=(self.stream, self.views.specs, self.imports, self.cache_frames),
            )
        return self._pool

    def _affinity_groups(self, items: list[WorkItem]) -> list[list[int]]:
        """Keep live clips on their previous worker; balance newly admitted ones.

        Only keys in the current batch survive, so ownership metadata is bounded.
        This changes execution placement, never item order or augmentation seeds.
        """
        keys = [_work_key(item) for item in items]
        live = set(keys)
        self._owners = {key: owner for key, owner in self._owners.items() if key in live}
        counts = [0] * self.workers
        for owner in self._owners.values():
            counts[owner] += 1
        groups: list[list[int]] = [[] for _ in range(self.workers)]
        for index, key in enumerate(keys):
            if key not in self._owners:
                owner = min(range(self.workers), key=counts.__getitem__)
                self._owners[key] = owner
                counts[owner] += 1
            groups[self._owners[key]].append(index)
        return groups

    def _submit_affinity(self, items: list[WorkItem], packed: bool):
        if not self._affinity_pools:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._affinity_pools = [
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=mp.get_context("spawn"),
                    initializer=_pool_init,
                    initargs=(self.stream, self.views.specs, self.imports, self.cache_frames),
                )
                for _ in range(self.workers)
            ]
        groups = self._affinity_groups(items)
        fn = _pool_chunk if packed else _pool_items
        return [
            (indices, pool.submit(fn, [items[i] for i in indices]))
            for pool, indices in zip(self._affinity_pools, groups, strict=True)
            if indices
        ]

    def close(self) -> None:
        """Shut the worker pool down (the drivers call this when a run ends).

        Waits for the workers to exit, so none outlive the run; queued decodes
        are cancelled, and the drivers have drained their prefetch by then.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        for pool in self._affinity_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        self._affinity_pools.clear()
        self._owners.clear()

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

    def views_batch(self, items: list[WorkItem]) -> list[torch.Tensor]:
        """Load many items, one view tensor per item, in item order (frames
        give [V,C,H,W], clips [V,T,C,H,W]; see the WorkItem shapes above).

        workers=0 loads in-process; otherwise decode+augment fan out to the
        process pool.  Same items, same tensors, either way."""
        if not items:
            return []
        if self.workers <= 0:
            out = []
            for item in items:
                if item[0] == "clip":
                    _, ref, start, views, clip_len, stride, seed = item
                    out.append(self._one_clip(ref, start, views, clip_len, stride, seed))
                else:
                    _, ref, index, views, seed = item
                    out.append(self.frame_views(ref, index, views, seed=seed))
            return out
        if self.affinity:
            with self.timer.measure("data.load") if self.timer else nullcontext():
                pending = self._submit_affinity(items, packed=False)
                ordered: dict[int, torch.Tensor] = {}
                for indices, future in pending:
                    ordered.update(zip(indices, future.result(), strict=True))
                return [ordered[i] for i in range(len(items))]
        pool = self._ensure_pool()
        chunk = max(1, len(items) // (self.workers * 4))
        if self.timer is not None:
            with self.timer.measure("data.load"):
                return list(pool.map(_pool_task, items, chunksize=chunk))
        return list(pool.map(_pool_task, items, chunksize=chunk))

    def tensor_batch(self, items: list[WorkItem]) -> torch.Tensor:
        """Collate homogeneous rows into one optionally pinned host allocation.

        Ported from the MAE loading path, with stable worker ownership for replay.
        Use views_batch for heterogeneous slots. No tensors enter the decode cache.
        """
        if not items:
            raise ValueError("tensor_batch requires at least one item")
        if not self.packed or self.workers <= 0:
            tensors = self.views_batch(items)
            first = tensors[0]
            out = torch.empty(
                (len(items), *first.shape), dtype=first.dtype, pin_memory=self.pin_memory
            )
            return torch.stack(tensors, out=out)
        with self.timer.measure("data.load") if self.timer else nullcontext():
            if self.affinity:
                pending = self._submit_affinity(items, packed=True)
                packed_out = None
                for indices, future in pending:
                    chunk = future.result()
                    if packed_out is None:
                        packed_out = torch.empty(
                            (len(items), *chunk.shape[1:]),
                            dtype=chunk.dtype,
                            pin_memory=self.pin_memory,
                        )
                    packed_out.index_copy_(0, torch.tensor(indices), chunk)
                assert packed_out is not None
                return packed_out
            size = max(1, (len(items) + 2 * self.workers - 1) // (2 * self.workers))
            chunks = list(
                self._ensure_pool().map(
                    _pool_chunk, [items[i : i + size] for i in range(0, len(items), size)]
                )
            )
            first = chunks[0]
            out = torch.empty(
                (len(items), *first.shape[1:]), dtype=first.dtype, pin_memory=self.pin_memory
            )
            return torch.cat(chunks, out=out)

    def clip_views(
        self,
        ref,
        view_names: list[str],
        clip_len: int,
        stride: int,
        rng: random.Random,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Decode a contiguous clip from `ref` and apply the named views,
        returning a [V,T,C,H,W] tensor.

        A clip is T=clip_len frames taken every `stride` frames from a random
        start inside the ref (drawn from `rng`); indices past the end clamp
        to the last frame, so a short ref yields a clip padded by repeating
        that frame (the common video convention).  Augmentation is temporally
        consistent: each clip view pipeline runs once on the whole [T,C,H,W]
        tensor, so all T frames get the same random crop etc. (see
        views.apply_clip).  With a `seed`, augmentation randomness comes from
        that seed instead of the global stream, like frame_views.

        `view_names` must name clip-domain views (clip_simclr,
        clip_center_crop, ...).  Frames are decoded one by one through
        stream.open_frame; batching the decode through
        VideoDecoder.read_window is an optimization left for the Poodle
        integration.
        """
        start = self.pick_clip_start(ref, clip_len, stride, rng)
        return self._one_clip(ref, start, view_names, clip_len, stride, seed)

    def _one_clip(
        self, ref, start: int, view_names: list[str], clip_len: int, stride: int, seed: int | None
    ) -> torch.Tensor:
        def _load() -> torch.Tensor:
            clip = _decode_clip(self.stream, ref, start, clip_len, stride)
            if seed is None:
                return self.views.apply_clip(clip, view_names)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                py_state = random.getstate()
                random.seed(seed)
                try:
                    return self.views.apply_clip(clip, view_names)
                finally:
                    random.setstate(py_state)

        if self.timer is not None:
            with self.timer.measure("data.load"):
                return _load()
        return _load()
