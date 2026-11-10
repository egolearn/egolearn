"""Video input for generative models: which frames of a clip a model is shown.

Scorers that ask a generative model about a video (multiple_choice.py,
temporal_grounding.py) read it here, so the frame rate, frame count and
resolution are configuration of the evaluation, recorded with every result,
and every model is shown the same frames.

``VideoSampling`` picks ``n = min(max_frames, max(1, ceil(duration * fps)))``
frames at times ``t_k = k * duration / n`` for ``k = 0 .. n - 1``, measured
from the clip start.  The first frame is the clip's first, and the sampling
rate handed to the model is ``n / duration``, so a model that maps frame k to
time ``k / rate`` (Qwen2.5-VL's absolute time encoding) recovers ``t_k``
exactly.  Long clips are therefore read below ``fps`` once they reach
``max_frames``.  Each frame is the first one presented at or after its time
(within half a native frame period), resized so its longer side is at most
``max_side`` pixels with the aspect ratio kept; ``max_side: null`` keeps the
native resolution.  A file that ends more than a second before the clip end raises
an error instead of repeating its last frame.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from egolearn.model.generative.base import Video

# Decoding forward is cheaper than seeking across gaps shorter than this.
_SEEK_GAP_SEC = 2.0
# A file may end this much before the requested end (rounding of the annotated duration);
# a larger shortfall means a truncated file.
_END_SLACK_SEC = 1.0


@dataclass(frozen=True)
class VideoClip:
    """Where an item's video is: a file, and optionally a window of it in seconds."""

    path: str
    start_sec: float = 0.0
    end_sec: float | None = None  # default: the end of the file


@dataclass(frozen=True)
class VideoSampling:
    """How a clip is read for a generative model (see the module docstring)."""

    fps: float = 1.0
    max_frames: int = 64
    max_side: int | None = 448

    def __post_init__(self) -> None:
        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.max_side is not None and self.max_side < 1:
            raise ValueError(f"max_side must be >= 1 or null, got {self.max_side}")

    def times(self, duration: float) -> list[float]:
        """Frame times in seconds from the clip start."""
        if not (math.isfinite(duration) and duration > 0):
            raise ValueError(f"clip duration must be positive, got {duration}")
        count = min(self.max_frames, max(1, math.ceil(duration * self.fps - 1e-9)))
        return [k * duration / count for k in range(count)]

    def resize(self, image: Image.Image) -> Image.Image:
        if self.max_side is None or max(image.size) <= self.max_side:
            return image
        scale = self.max_side / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.BICUBIC)


def _duration(container, stream, path: str) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / 1_000_000
    raise ValueError(f"{path}: the container reports no duration; give the clip's end_sec")


def read_video(clip: VideoClip, sampling: VideoSampling) -> Video:
    """The frames of ``clip`` that ``sampling`` selects, as a Video."""
    import av  # optional dependency (video / train extras)

    with av.open(os.fspath(clip.path)) as container:
        if not container.streams.video:
            raise ValueError(f"{clip.path}: no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        origin = stream.start_time or 0
        rate = float(stream.average_rate) if stream.average_rate else None
        tolerance = 0.5 / rate if rate else 1e-3
        end = clip.end_sec if clip.end_sec is not None else _duration(container, stream, clip.path)
        duration = end - clip.start_sec
        times = sampling.times(duration)
        frames: list[Image.Image] = []
        decoded = None
        current: Image.Image | None = None
        current_time = -math.inf
        last_frame: Any = None  # the latest frame decoded so far, and its time
        last_time = -math.inf
        for offset in times:
            target = clip.start_sec + offset
            if current is not None and current_time >= target - tolerance:
                frames.append(current)  # the video is sparser than the sampling
                continue
            if decoded is None or target - current_time > _SEEK_GAP_SEC:
                seek = max(0.0, target - tolerance)
                container.seek(origin + int(seek / time_base), stream=stream, backward=True)
                decoded = container.decode(stream)
            for frame in decoded:
                if frame.pts is None:
                    continue
                time = float((frame.pts - origin) * time_base)
                if time > last_time:
                    last_frame, last_time = frame, time
                if time >= target - tolerance:
                    current, current_time = sampling.resize(frame.to_image()), time
                    break
            else:  # past the last frame
                decoded = None
                if last_frame is None or target - last_time > _END_SLACK_SEC:
                    raise ValueError(
                        f"{clip.path} ends at {last_time:.1f} s, before the frame at "
                        f"{target:.1f} s (a truncated file, or a wrong end_sec)"
                    )
                if last_time > current_time:
                    current, current_time = sampling.resize(last_frame.to_image()), last_time
            assert current is not None  # set by the decode loop or the end-of-file branch
            frames.append(current)
    return Video(
        frames=tuple(frames),
        timestamps=tuple(times),
        fps=len(times) / duration,
        sampling=asdict(sampling),
    )


class VideoReader:
    """Reads clips with ``sampling``, in parallel, keeping the most recent ones.

    Benchmarks list items of the same clip together (NLQ has several queries per
    clip), so a small cache avoids decoding a clip once per question.
    """

    def __init__(self, sampling: VideoSampling, workers: int = 8, cache_size: int = 4):
        if workers < 1 or cache_size < 0:
            raise ValueError("workers must be >= 1 and cache_size >= 0")
        self.sampling = sampling
        self.workers = workers
        self.cache_size = cache_size
        self._cache: OrderedDict[VideoClip, Video] = OrderedDict()

    def read(self, clips: Sequence[VideoClip]) -> list[Video]:
        missing = [clip for clip in dict.fromkeys(clips) if clip not in self._cache]
        if len(missing) > 1 and self.workers > 1:
            with ThreadPoolExecutor(min(self.workers, len(missing))) as pool:
                videos = list(pool.map(lambda c: read_video(c, self.sampling), missing))
        else:
            videos = [read_video(clip, self.sampling) for clip in missing]
        fresh = dict(zip(missing, videos, strict=True))
        result = [fresh[clip] if clip in fresh else self._cache[clip] for clip in clips]
        for clip, video in fresh.items():
            self._cache[clip] = video
        for clip in clips:
            if clip in self._cache:
                self._cache.move_to_end(clip)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result


def shown(video: Video) -> dict[str, float]:
    """What the model was shown of one clip, for a prediction row."""
    return {"frames": len(video.frames), "fps": video.fps}


def shown_summary(rows: Sequence[dict]) -> dict[str, float]:
    """Mean frame count and sampling rate over prediction rows, for benchmark metadata.

    With ``max_frames`` reached, the rate is below the configured ``fps``; this records
    what the model actually saw.
    """
    seen = [row for row in rows if "frames" in row]
    if not seen:
        return {}
    return {
        "video_frames_mean": sum(row["frames"] for row in seen) / len(seen),
        "video_fps_mean": sum(row["fps"] for row in seen) / len(seen),
    }
