"""Read prepared WalkingTours frames in an explicit video order."""

from __future__ import annotations

import math
import re
from pathlib import Path

from egolearn.config.registry import STREAMS
from egolearn.data.sources.saycam import SaycamStream


@STREAMS.register("walkingtours")
class WalkingToursStream(SaycamStream):
    """Read numbered frames without resampling or reordering the videos."""

    def __init__(
        self,
        root_dir: str,
        sessions: list[str],
        fps: float = 3.75,
        include: str | None = None,
    ):
        if not sessions or len(set(sessions)) != len(sessions):
            raise ValueError("WalkingTours requires a nonempty ordered list of unique sessions")
        for session in sessions:
            path = Path(session)
            if not session or path.is_absolute() or ".." in path.parts or path == Path("."):
                raise ValueError(
                    "WalkingTours sessions must be relative directories under root_dir"
                )
        if include is not None:
            raise ValueError("WalkingTours uses explicit sessions; set include=null")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("WalkingTours fps must be finite and positive")
        super().__init__(root_dir, fps=fps, naming="frame_index", sessions=list(sessions))

    def _frames(self, video_dir: str) -> list[str]:
        if video_dir in self._frames_cache:
            return self._frames_cache[video_dir]
        files = super()._frames(video_dir)
        try:
            if not files:
                raise ValueError(f"WalkingTours session {video_dir!r} has no frames")
            for index, name in enumerate(files, start=1):
                match = re.fullmatch(r"frame_(\d+)\.(jpg|jpeg|png)", name, flags=re.IGNORECASE)
                if match is None or int(match[1]) != index:
                    raise ValueError(
                        f"WalkingTours session {video_dir!r} needs contiguous frame_<index> images "
                        f"starting at 1; expected frame {index}, found {name!r}"
                    )
        except ValueError:
            self._frames_cache.pop(video_dir, None)
            raise
        return files
