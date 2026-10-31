"""Ego4D NLQ: the moment of a clip that answers a natural-language query, from a generative model.

The benchmark reads an NLQ annotation file and turns every query into a
``GroundingItem`` over its whole clip (``<clip_root>/<clip_uid>.mp4``, with
the clip duration from the annotation); prompting, reply parsing and the
official metrics are in the generic scorer (evaluation/temporal_grounding.py).

``split: val`` reads ``nlq_val.json`` and reports R@{1,5} at IoU {0.3, 0.5}
and mIoU.  ``split: test`` reads ``nlq_test_unannotated.json``, whose windows
are hidden, and only writes predictions.  ``predictions_dir`` receives
``predictions.json`` in the challenge format (``clip_uid``, ``annotation_uid``,
``query_idx`` and five ``predicted_times`` per query, best first; a reply with
fewer windows repeats its last one, and a reply with none is submitted as
``[0, 0]``) and ``replies.jsonl`` with every reply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from egolearn.config.registry import BENCHMARKS
from egolearn.core import keys as K
from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.benchmarks.ego4d_nlq.data import NLQSplit, first_clips, load_split
from egolearn.evaluation.temporal_grounding import (
    TOP_K,
    GroundingItem,
    TemporalGroundingScorer,
)
from egolearn.evaluation.video_input import VideoClip, VideoSampling

SPLITS = ("val", "test")


@BENCHMARKS.register("ego4d_nlq")
class Ego4DNLQBenchmark(Benchmark):
    """Ego4D NLQ recall of a generative model; see docs/evaluation/ego4d_nlq.md."""

    name = "ego4d_nlq"
    version = "1"
    requires_capabilities = (K.CAP_TEXT_GENERATION,)

    def __init__(
        self,
        name: str,
        annotations_json: str,
        clip_root: str,
        split: str = "val",
        generative: str = "text",
        fps: float = 0.5,
        max_frames: int = 128,
        max_side: int | None = 448,
        batch_size: int = 8,
        workers: int = 8,
        max_clips: int | None = None,
        predictions_dir: str | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if max_clips is not None and max_clips < 1:
            raise ValueError("max_clips must be >= 1")
        self.name = name
        self.annotations_json = annotations_json
        self.clip_root = clip_root
        self.split = split
        self.max_clips = max_clips
        self.predictions_dir = predictions_dir
        self.scorer = TemporalGroundingScorer(
            name,
            generative=generative,
            sampling=VideoSampling(fps=fps, max_frames=max_frames, max_side=max_side),
            batch_size=batch_size,
            workers=workers,
        )
        self.data: NLQSplit | None = None

    def clip_path(self, clip_uid: str) -> Path:
        return Path(self.clip_root) / f"{clip_uid}.mp4"

    def prepare(self) -> None:
        if self.data is not None:
            return
        data = load_split(self.annotations_json, self.split, annotated=self.split == "val")
        if self.max_clips is not None:
            data = first_clips(data, self.max_clips)
        if not data.queries:
            raise ValueError(f"{self.annotations_json}: no queries")
        clips = {query.clip_uid for query in data.queries}
        missing = sorted(uid for uid in clips if not self.clip_path(uid).is_file())
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(clips)} NLQ clips are missing under {self.clip_root}, "
                f"e.g. {[f'{uid}.mp4' for uid in missing[:3]]}"
            )
        self.data = data

    def configuration(self) -> dict[str, Any]:
        return {
            **self.scorer.configuration(),
            "name": self.name,
            "split": self.split,
            "max_clips": self.max_clips,
        }

    def metadata(self) -> dict[str, Any]:
        if self.data is None:
            return {}
        return {**self.data.metadata(), **self.scorer.video_summary}

    def run(self, model, ctx) -> dict[str, float]:
        self.prepare()
        assert self.data is not None
        items = []
        for query in self.data.queries:
            clip = self.data.clips[query.clip_uid]
            window = None
            if query.start_sec is not None and query.end_sec is not None:
                window = (query.start_sec, query.end_sec)
            items.append(
                GroundingItem(
                    uid=f"{query.annotation_uid}/{query.query_idx}",
                    query=query.query,
                    video=VideoClip(str(self.clip_path(clip.clip_uid)), 0.0, clip.duration),
                    duration=clip.duration,
                    window=window,
                )
            )
        metrics, rows = self.scorer.evaluate(model, ctx, items)
        if self.predictions_dir is not None:
            self._write_outputs(rows)
        return metrics

    def _write_outputs(self, rows: list[dict[str, Any]]) -> None:
        assert self.data is not None and self.predictions_dir is not None
        out = Path(self.predictions_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = []
        for query, row in zip(self.data.queries, rows, strict=True):
            windows = row["windows"] or [[0.0, 0.0]]
            windows = windows + [windows[-1]] * (max(TOP_K) - len(windows))
            results.append(
                {
                    "clip_uid": query.clip_uid,
                    "annotation_uid": query.annotation_uid,
                    "query_idx": query.query_idx,
                    "predicted_times": windows,
                }
            )
        submission = {"version": "1.0", "challenge": "ego4d_nlq_challenge", "results": results}
        (out / "predictions.json").write_text(json.dumps(submission) + "\n")
        (out / "replies.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
