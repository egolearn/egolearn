"""Stream adapters for SAYCam and BabyView, read from directories of
pre-extracted frames.  An adapter is dataset-specific but method-agnostic:
any project training on these datasets uses the same adapter unchanged.

Expected layout: `root_dir/<video_dir>/<frame files>` where the video_dir
names sort chronologically (SAYCam's S_YYYYMMDD_... names do).  Frame naming:

  * "timestamp":   S_20131130_1313_01_259.20.jpg -> t = 259.20 s  (SAYCam)
  * "frame_index": frame_00000123.jpg           -> t = (123-1)/fps (BabyView)

The transcript channel is built from a CVCL-style "expand" json: a list of
{video_filename, utterance, utterance_num, timestamps[, clip_importance]}.
Each entry becomes one sparse interval [min(ts), max(ts)] that is surfaced
only at its end time, so the learner never sees an utterance before it has
finished.  The RAW timestamp list travels in the payload (text.timestamps)
because BabyCL's AN chunk emitter splits non-contiguous runs into separate
chunks.

Dataset-level preprocessing (5fps decoding, orientation fixes) happens in
the prepare scripts before training; adapters only read what is already on
disk.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from PIL import Image

from egolearn.config.registry import STREAMS
from egolearn.core import keys as K
from egolearn.core.refs import ItemRef

from . import base as S

_FRAME_EXTS = (".jpg", ".jpeg", ".png")


def _time_from_name(fn: str, naming: str, fps: float, strict: bool = False) -> float:
    stem = os.path.splitext(os.path.basename(fn))[0]
    if naming == "frame_index":
        m = re.search(r"frame_(\d+)$", stem, flags=re.IGNORECASE)
        if strict and (m is None or int(m.group(1)) < 1):
            raise ValueError(f"invalid frame-index filename: {fn}")
        idx = int(m.group(1)) if m else 1
        return (max(1, idx) - 1) / float(fps)
    # timestamp naming: last underscore-separated field is seconds
    try:
        value = float(stem.split("_")[-1])
        if strict and (not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid timestamp filename: {fn}")
        return value
    except ValueError:
        if strict:
            raise ValueError(f"invalid timestamp filename: {fn}") from None
        return 0.0


@STREAMS.register("saycam")
class SaycamStream(S.Stream):
    def __init__(
        self,
        root_dir: str,
        transcript_json: str | None = None,
        fps: float = 5.0,
        naming: str = "timestamp",
        sessions: list[str] | None = None,
        include: str | None = None,
        session_glob: str = "*",
        natural_order: bool = False,
        strict_filenames: bool = False,
    ):
        """`include` is an fnmatch pattern on session directory names (for
        example "S_*" selects the SAYCam S child when root_dir holds all
        three children).  If `sessions` (an explicit list) is given, it wins
        over both `include` and the default sorted directory scan."""
        self.root = Path(root_dir)
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.strict_filenames = bool(strict_filenames)
        self.naming = naming
        self.fps_source = (
            "declared_frame_index_fps" if naming == "frame_index" else "filename_timestamps"
        )
        self._session_ids = sessions  # explicit ordering wins; else sorted scan
        self._include = include
        self._session_glob = session_glob
        self._natural_order = natural_order
        self._frames_cache: dict[str, list[str]] = {}
        self._utts_by_stem: dict[str, list[dict]] = {}
        self._total: int | None = None
        if transcript_json:
            self._load_transcript(transcript_json)

    # ---- transcript --------------------------------------------------------
    def _load_transcript(self, path: str) -> None:
        with open(path) as f:
            items = json.load(f)
        for it in items:
            ts = it.get("timestamps") or []
            if not ts:
                continue
            stem = os.path.splitext((it.get("video_filename") or "").strip())[0]
            payload = {
                K.TEXT_UTTERANCE: it.get("utterance") or "",
                K.UTTERANCE_NUM: it.get("utterance_num"),
                K.TEXT_TIMESTAMPS: [float(t) for t in ts],
            }
            if it.get("clip_importance") is not None:
                payload[K.WEIGHT_SAMPLE] = float(it["clip_importance"])
            self._utts_by_stem.setdefault(stem, []).append(
                {"t0": float(min(ts)), "t1": float(max(ts)), "payload": payload}
            )

    # ---- session enumeration -----------------------------------------------
    def session_ids(self) -> list[str]:
        if self._session_ids is None:
            import fnmatch

            if not self.root.exists():
                raise FileNotFoundError(self.root)
            if not self.root.is_dir():
                raise NotADirectoryError(self.root)
            names = [
                str(d.relative_to(self.root))
                for d in self.root.glob(self._session_glob)
                if d.is_dir()
            ]
            names = [n for n in names if self._include is None or fnmatch.fnmatch(n, self._include)]
            if self._natural_order:
                # Child/date/part order (part 2 before part 10), with the
                # remaining recording identifier as a deterministic tie-break.
                self._session_ids = sorted(
                    names,
                    key=lambda n: (
                        [(1, int(p)) if p.isdigit() else (0, p) for p in re.split(r"(\d+)", n)],
                        n,
                    ),
                )
            else:
                self._session_ids = sorted(names)
        return self._session_ids

    def __getstate__(self):
        # Spawned decode workers discover filenames lazily. Copying the
        # parent's full multi-million-frame cache into each worker is costly.
        return {**self.__dict__, "_frames_cache": {}}

    def _frames(self, video_dir: str) -> list[str]:
        if video_dir not in self._frames_cache:
            p = self.root / video_dir
            files = [f for f in os.listdir(p) if f.lower().endswith(_FRAME_EXTS)]
            files.sort(
                key=lambda fn: _time_from_name(fn, self.naming, self.fps, self.strict_filenames)
            )
            self._frames_cache[video_dir] = files
        return self._frames_cache[video_dir]

    def sessions(self):
        for sid in self.session_ids():
            files = self._frames(sid)
            if not files:
                continue
            times = [
                _time_from_name(fn, self.naming, self.fps, self.strict_filenames) for fn in files
            ]
            video = S.DenseFrameChannel("video", sid, times, [(sid, i) for i in range(len(files))])
            channels: dict[str, S.Channel] = {"video": video}
            stem = os.path.splitext(sid)[0]
            utts = self._utts_by_stem.get(stem, [])
            if utts:
                channels["transcript"] = S.SparseIntervalChannel("transcript", sid, utts)
            yield S.Session(sid, channels)

    def total_observations(self) -> int | None:
        if self._total is None:
            total = 0
            for sid in self.session_ids():
                total += len(self._frames(sid))
                total += len(self._utts_by_stem.get(os.path.splitext(sid)[0], []))
            self._total = total
        return self._total

    # ---- reading pixels (refs become images only here) ----------------------
    def _locator_span(self, ref: ItemRef):
        loc = ref.locator
        if len(loc) == 2:  # (video_dir, idx) -- point frame
            return loc[0], loc[1], loc[1] + 1
        if len(loc) == 3:  # (video_dir, i0, i1) -- interval
            return loc[0], loc[1], loc[2]
        raise ValueError(f"unrecognized video locator {loc!r}")

    def num_frames(self, ref: ItemRef) -> int:
        _, i0, i1 = self._locator_span(ref)
        return i1 - i0

    def open_frame(self, ref: ItemRef, index: int) -> Image.Image:
        video_dir, i0, i1 = self._locator_span(ref)
        if not (0 <= index < i1 - i0):
            raise IndexError(f"frame index {index} outside ref span {i1 - i0}")
        fn = self._frames(video_dir)[i0 + index]
        with Image.open(self.root / video_dir / fn) as f:
            return f.convert("RGB")


@STREAMS.register("babyview")
class BabyViewStream(SaycamStream):
    """BabyView: same adapter as SAYCam, but frames are named by index
    (frame_%08d.jpg -> t=(i-1)/fps).

    The 9:16 portrait pre-crop (BabyView paper appendix) is not done here:
    it is handled either by a prepare script or by a precrop-enabled view
    (a registered augmentation pipeline), so pair this adapter with one.
    """

    def __init__(
        self,
        root_dir: str,
        transcript_json: str | None = None,
        fps: float = 5.0,
        sessions: list[str] | None = None,
        include: str | None = None,
        session_glob: str = "*",
        natural_order: bool = False,
        strict_filenames: bool = False,
    ):
        super().__init__(
            root_dir,
            transcript_json=transcript_json,
            fps=fps,
            naming="frame_index",
            sessions=sessions,
            include=include,
            session_glob=session_glob,
            natural_order=natural_order,
            strict_filenames=strict_filenames,
        )
