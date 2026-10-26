"""Decode constant-rate videos on demand with bounded, process-local handles."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from PIL import Image

from egolearn.config.registry import DECODERS
from egolearn.data.decoding.base import VideoDecoder


@dataclass
class _Handle:
    container: Any
    stream: Any
    frames: Iterator
    index: int = -1
    image: Image.Image | None = None


@DECODERS.register("pyav")
class PyAVDecoder(VideoDecoder):
    """Read RGB frames from constant-rate videos with indexed container metadata.

    ``cache_handles`` bounds open files per process. ``seek_gap`` is the
    largest forward gap, in frames, decoded without seeking (default 64).
    Each codec uses ``threads`` CPU threads (default 1). No pixels are stored
    on disk. Variable-rate timestamps and missing frame counts raise errors.
    """

    def __init__(self, cache_handles: int = 4, seek_gap: int = 64, threads: int = 1):
        if cache_handles < 1 or seek_gap < 1 or threads < 1:
            raise ValueError("cache_handles, seek_gap, and threads must be positive")
        self.cache_handles = cache_handles
        self.seek_gap = seek_gap
        self.threads = threads
        self._handles: OrderedDict[str, _Handle] = OrderedDict()
        self._meta: dict[str, tuple[float, int]] = {}

    def _open(self, path: str) -> _Handle:
        import av

        container = av.open(path)
        try:
            if not container.streams.video:
                raise ValueError(f"no video stream in {path}")
            stream = container.streams.video[0]
            if not stream.average_rate or not stream.time_base or stream.frames <= 0:
                raise ValueError(
                    f"decoder:pyav requires frame rate and frame count metadata: {path}"
                )
            stream.codec_context.thread_count = self.threads
            return _Handle(container, stream, iter(container.decode(stream)))
        except BaseException:
            container.close()
            raise

    def probe(self, path: str) -> tuple[float, int]:
        """Return container frame rate and count without decoding pixels."""
        if path not in self._meta:
            handle = self._open(path)
            try:
                self._meta[path] = (float(handle.stream.average_rate), handle.stream.frames)
            finally:
                handle.container.close()
        return self._meta[path]

    def _handle(self, path: str) -> _Handle:
        if path in self._handles:
            self._handles.move_to_end(path)
        else:
            if len(self._handles) >= self.cache_handles:
                _, old = self._handles.popitem(last=False)
                old.container.close()
            self._handles[path] = self._open(path)
        return self._handles[path]

    def _read(self, path: str, index: int) -> Image.Image:
        handle = self._handle(path)
        stream = handle.stream
        if not 0 <= index < stream.frames:
            raise IndexError(f"frame {index} outside {path} ({stream.frames} frames)")
        if index == handle.index and handle.image is not None:
            return handle.image
        origin = stream.start_time or 0
        tick = stream.time_base * stream.average_rate
        if index < handle.index or index - handle.index > self.seek_gap:
            handle.container.seek(
                origin + int(index / tick), stream=stream, backward=True, any_frame=False
            )
            handle.frames = iter(handle.container.decode(stream))
        for frame in handle.frames:
            if frame.pts is None:
                raise ValueError(f"missing frame timestamp in {path}")
            position = (frame.pts - origin) * tick
            ordinal = round(position)
            # Permit timestamp quantization of at most one container tick.
            if abs(position - ordinal) > max(tick, 0.001):
                raise ValueError(f"decoder:pyav requires constant-rate timestamps: {path}")
            if ordinal > index:
                raise ValueError(f"frame {index} missing from constant-rate timeline: {path}")
            if ordinal == index:
                handle.index = index
                handle.image = frame.to_image()
                return handle.image
        raise IndexError(f"video ended before frame {index}: {path}")

    def read_frame(self, path: str, index: int) -> Image.Image:
        return self._read(path, index)

    def read_window(
        self, path: str, i0: int, i1: int, indices: list[int] | None = None
    ) -> list[Image.Image]:
        """Read distinct frames in order, then restore requested order and repeats."""
        if not 0 <= i0 <= i1:
            raise IndexError(f"invalid frame window [{i0}, {i1})")
        picks = list(range(i1 - i0)) if indices is None else indices
        if any(not 0 <= i < i1 - i0 for i in picks):
            raise IndexError(f"frame indices outside window [{i0}, {i1})")
        decoded = {i: self._read(path, i0 + i) for i in sorted(set(picks))}
        return [decoded[i] for i in picks]

    def iter_frames(self, path: str, start_index: int = 0) -> Iterator:
        _, count = self.probe(path)
        if not 0 <= start_index <= count:
            raise IndexError(f"invalid start frame {start_index}")
        for index in range(start_index, count):
            yield index, self._read(path, index)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.container.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        # Spawned workers open their own handles, even after parent-side reads.
        return {**self.__dict__, "_handles": OrderedDict()}
