"""Deterministic metadata-only frame/clip inventory, shared with streaming budgets."""

from __future__ import annotations

import math
import statistics
from itertools import pairwise
from typing import cast

from egolearn.core.refs import ItemRef
from egolearn.data.sources.base import DenseFrameChannel, Stream


def index_stream(
    stream: Stream,
    *,
    unit: str = "clip",
    clip_len: int = 16,
    clip_stride: int = 1,
    subsample: int = 1,
    max_frame_gap: float | None = None,
    expected_fps: float | None = None,
):
    """List complete, session-local windows and their source statistics."""
    refs: list[ItemRef] = []
    median_intervals = []
    stats = {
        "sessions": 0,
        "frames": 0,
        "gaps": 0,
        "continuous_runs": 0,
        "clips": 0,
        "discarded_frames": 0,
    }
    span = (clip_len - 1) * clip_stride + 1
    for session in stream.sessions():
        for ch in session.channels.values():
            if getattr(ch, "kind", None) != "dense":
                continue
            dense = cast(DenseFrameChannel, ch)
            stats["sessions"] += 1
            stats["frames"] += len(dense)
            if unit != "clip":
                refs.extend(dense.ref_at(i) for i in range(0, len(dense), subsample))
                continue
            deltas = [b - a for a, b in pairwise(dense.times)]
            if deltas:
                median_intervals.append(statistics.median(deltas))
            if expected_fps is not None and deltas:
                median = median_intervals[-1]
                if not math.isfinite(median) or abs(median * expected_fps - 1) > 0.1:
                    raise ValueError(
                        f"session {session.sid}: median frame interval {median} "
                        f"does not match expected_fps={expected_fps}"
                    )
            breaks = [0]
            if max_frame_gap is not None:
                breaks.extend(
                    i + 1
                    for i, dt in enumerate(deltas)
                    if not math.isfinite(dt) or dt <= 0 or dt > max_frame_gap
                )
            breaks.append(len(dense))
            stats["gaps"] += len(breaks) - 2
            stats["continuous_runs"] += len(breaks) - 1
            selected = 0
            for begin, end in pairwise(breaks):
                for j, i0 in enumerate(range(begin, end - span + 1, span)):
                    if j % subsample == 0:
                        refs.append(dense.interval_ref(i0, i0 + span))
                        selected += 1
            stats["clips"] += selected
            stats["discarded_frames"] += len(dense) - selected * span
    index_stats: dict[str, float | int | str] = dict(stats)
    index_stats["fps_source"] = getattr(stream, "fps_source", "stream_timestamps")
    if median_intervals:
        index_stats["median_interval_s"] = statistics.median(median_intervals)
    if unit == "clip":
        print(f"[clip index] {index_stats}", flush=True)
    return refs, index_stats


def index_digest(refs: list[ItemRef]) -> str:
    """Pin ordered session/channel/locator/timestamp identity, without reading pixels."""
    import hashlib
    import json

    digest = hashlib.sha256()
    for ref in refs:
        digest.update(json.dumps([ref.key, ref.t0, ref.t1]).encode() + b"\n")
    return digest.hexdigest()
