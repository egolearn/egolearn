"""Ego4D NLQ annotations: clips and their natural-language queries.

An NLQ annotation file lists videos, each with clips (mostly 8 minutes, from
3.5 to 20); every clip carries annotations whose ``language_queries`` hold a
question and, outside the test split, the clip-relative window
``[clip_start_sec, clip_end_sec]`` that answers it.  As in the official
baseline, a query without text is skipped but still counts in the numbering:
``query_idx`` is the index in ``language_queries``, which is how prediction
files name queries.

The benchmark checks each file against the SHA-256 of the official v2 release
(``ANNOTATION_SHA256``; the ego4d CLI with ``--version v2 --datasets
annotations``), so every run scores the same queries: 415 val clips with 4,552
queries, and 333 test clips with 4,004.  ``verify=False`` admits subsets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ANNOTATION_SHA256 = {
    "val": "20413901b8dbfcb097e7219da6b3a577ebb1c309f8e6667a0f53fdd2ae0d6da7",
    "test": "fbe1452334e7b7b0048c51dcc3457cae4138142faac94467bc293074191f5161",
}


@dataclass(frozen=True)
class NLQClip:
    clip_uid: str
    video_uid: str
    video_start_sec: float  # the clip's range in its parent video
    video_end_sec: float

    @property
    def duration(self) -> float:
        return self.video_end_sec - self.video_start_sec


@dataclass(frozen=True)
class NLQQuery:
    clip_uid: str
    annotation_uid: str
    query_idx: int  # index in the annotation's language_queries (skipped ones included)
    query: str
    start_sec: float | None  # clip-relative answer window; None on the test split
    end_sec: float | None


@dataclass(frozen=True)
class NLQSplit:
    name: str
    clips: dict[str, NLQClip]  # clip_uid -> clip, in file order
    queries: tuple[NLQQuery, ...]
    sha256: str | None  # of the annotation file the split was read from

    def metadata(self) -> dict[str, Any]:
        return {
            "n_clips": len(self.clips),
            "n_videos": len({clip.video_uid for clip in self.clips.values()}),
            "n_queries": len(self.queries),
            "sha256": self.sha256,
        }


def load_split(path: str | Path, name: str, annotated: bool = True) -> NLQSplit:
    """Read an NLQ annotation file (``nlq_{train,val,test_unannotated}.json``).

    ``annotated=False`` reads the test split, whose queries carry text only.
    """
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    clips: dict[str, NLQClip] = {}
    queries: list[NLQQuery] = []
    for video in data["videos"]:
        for clip_datum in video["clips"]:
            clip_uid = str(clip_datum["clip_uid"])
            if clip_uid in clips:
                raise ValueError(f"{path}: clip {clip_uid} appears twice")
            clips[clip_uid] = NLQClip(
                clip_uid=clip_uid,
                video_uid=str(video["video_uid"]),
                video_start_sec=float(clip_datum["video_start_sec"]),
                video_end_sec=float(clip_datum["video_end_sec"]),
            )
            for annotation in clip_datum["annotations"]:
                for index, datum in enumerate(annotation["language_queries"]):
                    if not datum.get("query"):
                        continue
                    start = end = None
                    if annotated:
                        start = float(datum["clip_start_sec"])
                        end = float(datum["clip_end_sec"])
                    queries.append(
                        NLQQuery(
                            clip_uid=clip_uid,
                            annotation_uid=str(annotation["annotation_uid"]),
                            query_idx=index,
                            query=str(datum["query"]),
                            start_sec=start,
                            end_sec=end,
                        )
                    )
    return NLQSplit(
        name=name,
        clips=clips,
        queries=tuple(queries),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _subset(split: NLQSplit, name: str, clip_uids: Iterable[str]) -> NLQSplit:
    keep = set(clip_uids)
    return replace(
        split,
        name=name,
        clips={uid: clip for uid, clip in split.clips.items() if uid in keep},
        queries=tuple(q for q in split.queries if q.clip_uid in keep),
    )


def first_clips(split: NLQSplit, count: int) -> NLQSplit:
    """The first ``count`` clips of a split (file order) with their queries; for smoke runs."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    return _subset(split, split.name, list(split.clips)[:count])
