"""FrameDirDecoder: a directory of pre-extracted frames, viewed as a video.

This is the legacy/live-camera path (SAYCam 5fps JPEG dumps) expressed as a
decoding backend, so that video-native and frame-dump datasets share the
same VideoStream source.  "Decoding" is just an image open; random and
sequential access cost the same.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from PIL import Image

from egolearn.config.registry import DECODERS

from .base import VideoDecoder

_FRAME_EXTS = (".jpg", ".jpeg", ".png")


@DECODERS.register("frame_dir")
class FrameDirDecoder(VideoDecoder):
    def __init__(self, fps: float = 5.0):
        self.fps = float(fps)
        self._listing_cache: dict[str, list[str]] = {}

    def _frames(self, path: str) -> list[str]:
        if path not in self._listing_cache:
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(_FRAME_EXTS))
            self._listing_cache[path] = files
        return self._listing_cache[path]

    def probe(self, path: str) -> tuple[float, int]:
        return self.fps, len(self._frames(path))

    def read_frame(self, path: str, index: int):
        fn = self._frames(path)[index]
        with Image.open(os.path.join(path, fn)) as f:
            return f.convert("RGB")

    def read_window(self, path: str, i0: int, i1: int, indices: list[int] | None = None) -> list:
        picks = range(i1 - i0) if indices is None else indices
        return [self.read_frame(path, i0 + i) for i in picks]

    def iter_frames(self, path: str, start_index: int = 0) -> Iterator:
        for i in range(start_index, len(self._frames(path))):
            yield i, self.read_frame(path, i)

    def close(self) -> None:
        self._listing_cache.clear()
