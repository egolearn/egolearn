"""Loader: turns an ItemRef into tensors, as late as possible.

Everything upstream of the Loader passes around refs (lightweight pointers
into the source data) plus metadata; images are only decoded and augmented
here, when a batch is being built.  Alternative loaders (embedding caches,
generative memory) implement this same small interface, so replacing how
items become tensors -- e.g. a "generative memory" -- means implementing
these few methods and touching nothing else.

If a StageTimer is attached (`loader.timer`, set by the training driver),
decode+augment time is recorded under "data.load" -- this is the number that
answers "how much latency do the data abstractions cost?".
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as TF

from .sources.base import Stream
from .views import ViewBank

if TYPE_CHECKING:
    from egolearn.training.profiling import StageTimer


class Loader:
    def __init__(self, stream: Stream, views: ViewBank):
        self.stream = stream
        self.views = views
        # optional StageTimer (egolearn.training.profiling); the driver attaches it
        self.timer: StageTimer | None = None

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
    def frame_views(self, ref, index: int, view_names: list[str]):
        """Decode one frame of `ref` and apply the named views (augmentation
        pipelines), returning a [V,C,H,W] tensor with one row per view."""
        if self.timer is not None:
            with self.timer.measure("data.load"):
                pil = self.stream.open_frame(ref, index)
                return self.views.apply(pil, view_names)
        pil = self.stream.open_frame(ref, index)
        return self.views.apply(pil, view_names)

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
