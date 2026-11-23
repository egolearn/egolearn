# compute_iou and the recall and mIoU loop in TemporalGroundingScorer._metrics are
# adapted from EGO4D/episodic-memory NLQ/VSLNet/utils/evaluate_ego4d_nlq.py at
# 5275b9570a3d77884e8685bad474309f7778db1f (MIT, Copyright (c) 2021 EGO4D Dataset and
# Benchmark Suite; full notice in THIRD_PARTY_NOTICES.md).
"""Generic temporal-grounding scorer for generative models.

A benchmark builds ``GroundingItem``s (a query, a clip, the clip's duration
and, unless hidden, the ground-truth window in seconds from the clip start).
This scorer asks the model for the window in one prompt per item, parses the
reply and computes the metrics.  It needs ``text_generation``.

The prompt and the reply parser are functions registered by name (registries
``prompt`` and ``parser``) and chosen in the configuration, which records the
names used.  A prompt takes ``(query, duration)`` and returns the text; a parser
takes ``(reply, duration)`` and returns windows ``[start, end]``, best first.
Register others in a module listed in ``_imports_``.

The default prompt, ``start_end_seconds``::

    The video is <duration> seconds long.
    Query: <query>
    Give the start and end time, in seconds, of the moment in the video that
    answers the query. Reply in the form "start - end".

The model sees the clip as sampled by ``VideoSampling``, whose frame times
start at 0 at the clip start, so its answer is in the same time frame as the
annotation.  The default parser, ``time_pairs``, reads every pair of times joined by "-", "to",
"and", "," or "~", in order of appearance; times are seconds ("12.5", "12.5 s")
or clock times ("1:05", "0:01:05").  Each window is put in order and clipped
to ``[0, duration]``; the first five are kept, best first.  A reply with no
pair counts as a miss with IoU 0 and is reported as ``unparsed``.

The metrics are the official Ego4D NLQ evaluation (``compute_IoU`` and
``evaluate_nlq_performance`` in ``evaluate_ego4d_nlq.py``, ported): an item counts
as recalled at (k, t) when one of its first k windows has IoU strictly above t
with the ground truth (``r<k>_iou<t>``), and ``miou`` averages
``np.sort(overlap[0])[-3:]`` per item.  ``overlap`` has one row per window and one
column for the single ground-truth window, so that is the first window's IoU.  A
reply without a window is scored as ``[0, 0]``, as in the prediction file.  The
default prompt asks for one window, so R@5 equals R@1 unless a model offers several.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from egolearn.config.registry import PARSERS, PROMPTS
from egolearn.core import keys as K
from egolearn.evaluation.video_input import (
    VideoClip,
    VideoReader,
    VideoSampling,
    shown,
    shown_summary,
)

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


@PROMPTS.register("start_end_seconds")
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


@PARSERS.register("time_pairs")
def parse_windows(reply: str, duration: float, limit: int = max(TOP_K)) -> list[list[float]]:
    """Windows ``[start, end]`` named in a reply, in order (see the module docstring)."""
    windows = []
    for match in _PAIR.finditer(reply):
        start, end = sorted((_seconds(match.group(1)), _seconds(match.group(2))))
        windows.append([min(max(start, 0.0), duration), min(max(end, 0.0), duration)])
        if len(windows) == limit:
            break
    return windows


def compute_iou(pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]]) -> np.ndarray:
    """IoU of every predicted window with every ground-truth window, ``[P, G]``.

    The official ``compute_IoU`` for lists of windows: the union is the hull of the two.
    """
    pred_arr, gt_arr = np.array(pred, dtype=float), np.array(gt, dtype=float)
    inter_left = np.maximum(pred_arr[:, 0, None], gt_arr[None, :, 0])
    inter_right = np.minimum(pred_arr[:, 1, None], gt_arr[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred_arr[:, 0, None], gt_arr[None, :, 0])
    union_right = np.maximum(pred_arr[:, 1, None], gt_arr[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    return 1.0 * inter / union


class TemporalGroundingScorer:
    """Scores items with ``model.generate(generative, texts=, videos=)``."""

    requires_capabilities = (K.CAP_TEXT_GENERATION,)

    def __init__(
        self,
        name: str,
        generative: str = "text",
        sampling: VideoSampling | None = None,
        batch_size: int = 8,
        workers: int = 8,
        prompt: str = "start_end_seconds",
        parser: str = "time_pairs",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.name = name
        self.prompt, self._prompt = prompt, PROMPTS.get(prompt)
        self.parser, self._parser = parser, PARSERS.get(parser)
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
            "prompt": self.prompt,
            "parser": self.parser,
        }

    def evaluate(
        self, model, ctx, items: Sequence[GroundingItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """``(metrics, predictions)``; one prediction row per item, in order."""
        if not items:
            raise ValueError(f"{self.name}: no items to score")
        if K.CAP_TEXT_GENERATION not in model.capabilities():
            raise ValueError(
                f"{self.name}: the model has no text-generating component "
                f"(needs {K.CAP_TEXT_GENERATION!r}, has {sorted(model.capabilities())})"
            )
        rows: list[dict[str, Any]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            videos = self.reader.read([item.video for item in batch])
            replies = model.generate(
                self.generative,
                texts=[self._prompt(item.query, item.duration) for item in batch],
                videos=[(video,) for video in videos],
            )
            if len(replies) != len(batch):
                raise ValueError(f"{self.name}: {len(replies)} replies for {len(batch)} items")
            for item, reply, video in zip(batch, replies, videos, strict=True):
                windows = self._parser(str(reply), item.duration)
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
        # The loop of the official evaluate_nlq_performance.
        results: list[list[list[bool]]] = [[[] for _ in TOP_K] for _ in THRESHOLDS]
        average_iou = []
        for item, row in scored:
            assert item.window is not None
            overlap = compute_iou(row["windows"] or [[0.0, 0.0]], [list(item.window)])
            average_iou.append(np.mean(np.sort(overlap[0])[-3:]))
            for tt, threshold in enumerate(THRESHOLDS):
                for rr, k in enumerate(TOP_K):
                    results[tt][rr].append(bool((overlap > threshold)[:k].any()))
        recall = np.array(results).mean(axis=-1)
        for tt, threshold in enumerate(THRESHOLDS):
            for rr, k in enumerate(TOP_K):
                metrics[f"r{k}_iou{threshold}"] = float(recall[tt][rr])
        metrics["miou"] = float(np.mean(average_iou))
        metrics["n_scored"] = float(len(scored))
        return metrics
