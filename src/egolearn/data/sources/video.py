"""VideoStream: treats a directory of raw video files as a stream source
(a stream is the dataset adapter that yields sessions of observations).

This is the video-first data path decided at the 2026-08 working-group
meeting: each session is a video file itself, and frames are decoded on
demand by a pluggable VideoDecoder backend (egolearn.data.decoding).  There
is no step that pre-extracts frames to disk.

Session metadata (fps, frame counts) comes from `manifest_json` when given
-- {"<file>": {"fps": ..., "num_frames": ...}, ...} -- otherwise from
`decoder.probe()`, called once per file the first time it is needed.

The transcript channel reuses the CVCL-style expand json exactly as
SaycamStream does (utterances do not depend on how video is decoded).

Locators are (video_path, i0[, i1]) frame-index spans, the same shape the
frame-dir adapters use -- so admission components and memory never need to
know which backend created a ref (a lightweight pointer to frames on disk).

Known limitation (fine for the skeleton, must be fixed with a real video
backend): `open_frame` uses the decoder's RANDOM access path even during the
sequential streaming pass.  See decoding/poodle.py for the two ways to close
this (a read-ahead cache inside the decoder, or prefetching via iter_frames).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from egolearn.config.registry import STREAMS
from egolearn.core import keys as K
from egolearn.core.refs import ItemRef

from . import base as S

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@STREAMS.register("video")
class VideoStream(S.Stream):
    def __init__(
        self,
        root_dir: str,
        decoder,
        transcript_json: str | None = None,
        manifest_json: str | None = None,
        sessions: list[str] | None = None,
    ):
        self.root = Path(root_dir)
        self.decoder = decoder
        self._session_ids = sessions  # explicit ordering wins; else sorted scan
        self._meta: dict[str, dict] = {}
        self._utts_by_stem: dict[str, list[dict]] = {}
        self._total: int | None = None
        if manifest_json:
            with open(manifest_json) as f:
                self._meta = {k: dict(v) for k, v in json.load(f).items()}
        if transcript_json:
            self._load_transcript(transcript_json)

    # ---- transcript (same expand-json format as SaycamStream) --------------
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
            self._session_ids = sorted(
                f for f in os.listdir(self.root) if f.lower().endswith(_VIDEO_EXTS)
            )
        return self._session_ids

    def _session_meta(self, sid: str) -> dict:
        if sid not in self._meta:
            fps, n = self.decoder.probe(str(self.root / sid))
            self._meta[sid] = {"fps": float(fps), "num_frames": int(n)}
        return self._meta[sid]

    def sessions(self):
        for sid in self.session_ids():
            meta = self._session_meta(sid)
            fps, n = float(meta["fps"]), int(meta["num_frames"])
            if n <= 0:
                continue
            path = str(self.root / sid)
            times = [i / fps for i in range(n)]
            video = S.DenseFrameChannel("video", sid, times, [(path, i) for i in range(n)])
            channels: dict[str, S.Channel] = {"video": video}
            utts = self._utts_by_stem.get(os.path.splitext(sid)[0], [])
            if utts:
                channels["transcript"] = S.SparseIntervalChannel("transcript", sid, utts)
            yield S.Session(sid, channels)

    def total_observations(self) -> int | None:
        if self._total is None:
            total = 0
            for sid in self.session_ids():
                total += int(self._session_meta(sid)["num_frames"])
                total += len(self._utts_by_stem.get(os.path.splitext(sid)[0], []))
            self._total = total
        return self._total

    # ---- reading pixels (refs become images only here) ----------------------
    def _locator_span(self, ref: ItemRef):
        loc = ref.locator
        if len(loc) == 2:  # (video_path, idx) -- point frame
            return loc[0], loc[1], loc[1] + 1
        if len(loc) == 3:  # (video_path, i0, i1) -- interval
            return loc[0], loc[1], loc[2]
        raise ValueError(f"unrecognized video locator {loc!r}")

    def num_frames(self, ref: ItemRef) -> int:
        _, i0, i1 = self._locator_span(ref)
        return i1 - i0

    def open_frame(self, ref: ItemRef, index: int):
        path, i0, i1 = self._locator_span(ref)
        if not (0 <= index < i1 - i0):
            raise IndexError(f"frame index {index} outside ref span {i1 - i0}")
        return self.decoder.read_frame(path, i0 + index)
