"""Video decoding backends: (video file, time) -> frames, on the fly.

Design decision from the 2026-08 working-group meeting: the primary data
path is DIRECT VIDEO LOADING -- keep the raw videos on disk and decode the
requested window on demand -- instead of pre-extracting millions of frame
JPEGs.  Pre-extracted frame directories remain supported as one backend
(`decoder:frame_dir`), because live-camera and legacy pipelines produce them.

A decoder must serve two access patterns, and implementations should
optimize them separately:

  * SEQUENTIAL (the streaming pass): frames are requested in time order.
    `iter_frames` should keep an open decode handle and prefetch; never
    reopen/seek per frame.
  * RANDOM (replay): memory holds refs to old intervals and asks for a few
    frames from arbitrary positions.  `read_frame`/`read_window` should
    seek to the nearest keyframe and decode forward, batching requests to
    the same file where possible.

The efficient random-access implementation is expected to come from the
lab's existing video pipeline (see decoding/poodle.py); this protocol is the
integration surface.
"""

from __future__ import annotations

from collections.abc import Iterator


class VideoDecoder:
    """Protocol. All times are media-time seconds within one video file."""

    def probe(self, path: str) -> tuple[float, int]:
        """Return (fps, num_frames) without decoding the whole file."""
        raise NotImplementedError  # pragma: no cover

    def read_frame(self, path: str, index: int):
        """Random access: decode and return frame `index` as a PIL image."""
        raise NotImplementedError  # pragma: no cover

    def read_window(self, path: str, i0: int, i1: int, indices: list[int] | None = None) -> list:
        """Random access: frames [i0, i1) -- or only `indices` (relative to
        i0) within that window.  One seek per window, not per frame."""
        raise NotImplementedError  # pragma: no cover

    def iter_frames(self, path: str, start_index: int = 0) -> Iterator:
        """Sequential access: yield (index, PIL image) from start_index on.
        Backends should hold the handle open and prefetch."""
        raise NotImplementedError  # pragma: no cover

    def close(self) -> None:
        """Release any cached decode handles."""
