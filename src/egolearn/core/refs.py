"""ItemRef: a small reference that points at a piece of stream data.

Everything that makes decisions -- admission components (the pieces that
watch the stream and decide what enters memory), memory tables, samplers --
passes around ItemRefs plus small metadata, never tensors.  Images are only
decoded when a batch is built (`egolearn.data.loading`), as late as possible.

A ref is a *timed interval* on one channel of one session.  A single frame
is the special case t1 == t0.  What the interval loads into (one frame,
K sampled frames, a contiguous clip, an audio window, ...) is decided by the
batch recipe's view spec (the recipe describes how a batch is assembled;
a view is an augmentation pipeline), not by the ref itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ItemRef:
    session: str  # session (video) id
    channel: str  # e.g. "video", "transcript"
    t0: float  # media time, seconds
    t1: float  # media time, seconds; == t0 for point items
    locator: tuple = field(default=())
    # `locator` is an opaque, hashable, adapter-defined address, e.g.
    # ("frames", video_dir, start_idx, end_idx).  Only the Stream adapter
    # that minted the ref interprets it.

    def __post_init__(self):
        if self.t1 < self.t0:
            raise ValueError(f"ItemRef with t1 < t0: {self!r}")

    @property
    def is_point(self) -> bool:
        return self.t1 == self.t0

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def key(self) -> tuple:
        """Stable hashable identity (used by LabelStore, dedup sets)."""
        return (self.session, self.channel, self.locator)
