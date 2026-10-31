"""Generic temporal-grounding scorer for generative models.

A benchmark builds ``GroundingItem``s (a query, a clip, the clip's duration
and, unless hidden, the ground-truth window in seconds from the clip start).
This scorer asks the model for the window in one prompt per item, parses the
reply and computes the metrics.  It needs ``text_generation``.

The prompt (``PROMPT_VERSION`` 1)::

    The video is <duration> seconds long.
    Query: <query>
    Give the start and end time, in seconds, of the moment in the video that
    answers the query. Reply in the form "start - end".

The model sees the clip as sampled by ``VideoSampling``, whose frame times
start at 0 at the clip start, so its answer is in the same time frame as the
annotation.  A reply is read as every pair of times joined by "-", "to",
"and", "," or "~", in order of appearance; times are seconds ("12.5", "12.5 s")
or clock times ("1:05", "0:01:05").  Each window is put in order and clipped
to ``[0, duration]``; the first five are kept, best first.  A reply with no
pair counts as a miss with IoU 0 and is reported as ``unparsed``.

The metrics follow the official Ego4D NLQ evaluation (``evaluate_ego4d_nlq.py``
in EGO4D/episodic-memory): an item counts as recalled at (k, t) when one of its
first k windows has IoU strictly above t with the ground truth (``r<k>_iou<t>``),
and ``miou`` is the mean IoU of the first window.  The prompt asks for one
window, so R@5 equals R@1 unless a model offers several.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from egolearn.evaluation.video_input import (
    VideoClip,
    VideoReader,
    VideoSampling,
    shown,
    shown_summary,
)

PROMPT_VERSION = 1
THRESHOLDS = (0.3, 0.5)
TOP_K = (1, 5)
_TIME = r"(\d+(?::\d{1,2}){1,2}(?:\.\d+)?|\d+(?:\.\d+)?)"
_PAIR = re.compile(
    _TIME + r"\s*(?:s|secs?|seconds?)?\s*(?:-|\u2013|\u2014|to|and|,|~)\s*" + _TIME, re.IGNORECASE
)


@dataclass(frozen=True)
class GroundingItem:
    uid: str
    query: str
    video: VideoClip
    duration: float  # seconds; windows are in [0, duration] from the clip start
    window: tuple[float, float] | None = None  # ground truth; None when hidden

    def __post_init__(self) -> None:
        if not self.duration > 0:
            raise ValueError(f"{self.uid}: duration must be positive, got {self.duration}")


def format_prompt(query: str, duration: float) -> str:
    return (
        f"The video is {duration:.1f} seconds long.\n"
        f"Query: {query.strip()}\n"
        "Give the start and end time, in seconds, of the moment in the video that answers "
        'the query. Reply in the form "start - end".'
    )


def _seconds(text: str) -> float:
    value = 0.0
    for part in text.split(":"):
        value = value * 60 + float(part)
    return value


def parse_windows(reply: str, duration: float, limit: int = max(TOP_K)) -> list[list[float]]:
    """Windows ``[start, end]`` named in a reply, in order (see the module docstring)."""
    windows = []
    for match in _PAIR.finditer(reply):
        start, end = sorted((_seconds(match.group(1)), _seconds(match.group(2))))
        windows.append([min(max(start, 0.0), duration), min(max(end, 0.0), duration)])
        if len(windows) == limit:
            break
    return windows


def interval_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two windows; the union is their hull, as in the official evaluation."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


class TemporalGroundingScorer:
    """Scores items with ``model.generate(generative, texts=, videos=)``."""

    def __init__(
        self,
        name: str,
        generative: str = "text",
        sampling: VideoSampling | None = None,
        batch_size: int = 8,
        workers: int = 8,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.name = name
        self.generative = generative
        self.sampling = sampling or VideoSampling()
        self.batch_size = batch_size
        self.reader = VideoReader(self.sampling, workers=workers)
        self.video_summary: dict[str, float] = {}

    def configuration(self) -> dict[str, Any]:
        return {
            "generative": self.generative,
            **asdict(self.sampling),
            "batch_size": self.batch_size,
            "workers": self.reader.workers,
            "prompt_version": PROMPT_VERSION,
        }

    def evaluate(
        self, model, ctx, items: Sequence[GroundingItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """``(metrics, predictions)``; one prediction row per item, in order."""
        if not items:
            raise ValueError(f"{self.name}: no items to score")
        rows: list[dict[str, Any]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            videos = self.reader.read([item.video for item in batch])
            replies = model.generate(
                self.generative,
                texts=[format_prompt(item.query, item.duration) for item in batch],
                videos=[(video,) for video in videos],
            )
            if len(replies) != len(batch):
                raise ValueError(f"{self.name}: {len(replies)} replies for {len(batch)} items")
            for item, reply, video in zip(batch, replies, videos, strict=True):
                windows = parse_windows(str(reply), item.duration)
                row = {"uid": item.uid, "reply": str(reply), "windows": windows}
                rows.append(row | shown(video))
        self.video_summary = shown_summary(rows)
        return self._metrics(items, rows), rows

    def _metrics(self, items, rows) -> dict[str, float]:
        metrics: dict[str, float] = {
            "n_queries": float(len(rows)),
            "unparsed": sum(not row["windows"] for row in rows) / len(rows),
        }
        scored = [
            (item, row) for item, row in zip(items, rows, strict=True) if item.window is not None
        ]
        if not scored:
            return metrics
        ious = [
            [interval_iou(window, item.window) for window in row["windows"]] or [0.0]
            for item, row in scored
        ]
        for k in TOP_K:
            for threshold in THRESHOLDS:
                hits = [any(iou > threshold for iou in values[:k]) for values in ious]
                metrics[f"r{k}_iou{threshold}"] = float(np.mean(hits))
        metrics["miou"] = float(np.mean([values[0] for values in ious]))
        metrics["n_scored"] = float(len(scored))
        return metrics
