"""Video decoding backends: given a video file and a position, return frames
decoded on the fly.  A decoder is the pluggable component VideoStream uses
to read pixels.

Design decision from the 2026-08 working-group meeting: the primary data
path is DIRECT VIDEO LOADING -- keep the raw videos on disk and decode the
requested window on demand -- instead of pre-extracting millions of frame
JPEGs.  Pre-extracted frame directories remain supported as one backend
(`decoder:frame_dir`), because live-camera and legacy pipelines produce them.

A decoder must serve two access patterns, and implementations should
optimize them separately:

  * SEQUENTIAL (the streaming pass): frames are requested in time order.
    `iter_frames` should keep an open decode handle and prefetch; never
    reopen the file or seek for each frame.
  * RANDOM (replay): memory holds refs to old intervals and asks for a few
    frames from arbitrary positions.  `read_frame`/`read_window` should
    seek to the nearest keyframe and decode forward, batching requests to
    the same file where possible.

PyAV supplies a CPU backend for constant-rate recordings. The Poodle
backend remains a placeholder for integration with the lab's video pipeline.
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
        """Random access: return frames [i0, i1) -- or, if `indices` is
        given, only those frames (indices are relative to i0).  Implementations
        should seek once per window, not once per frame."""
        raise NotImplementedError  # pragma: no cover

    def iter_frames(self, path: str, start_index: int = 0) -> Iterator:
        """Sequential access: yield (index, PIL image) from start_index on.
        Backends should hold the handle open and prefetch."""
        raise NotImplementedError  # pragma: no cover

    def close(self) -> None:
        """Release any cached decode handles."""
