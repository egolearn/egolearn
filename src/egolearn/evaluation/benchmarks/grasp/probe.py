"""GRASP Level 2 probe backed by the generic VoE scorer.

The upstream release ships no metadata, so `prepare.py` builds `manifest.json`:
the (plausible, implausible) twins of every test, each video's SHA-256 and
frame count, and whether the twins open on the same first frame.  This adapter
reads that manifest, resolves paths against a caller-provided root, optionally
restricts the tests, and labels each pair with its test and concepts so the
scorer reports per-test and per-concept accuracy and AUROC alongside the
overall scores.  docs/evaluation/grasp.md documents the download and protocol.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from egolearn.config.registry import PROBES
from egolearn.evaluation.benchmarks.grasp.manifest import TESTS, pair_groups
from egolearn.evaluation.voe import VoEProbe
from egolearn.hub import sha256_file


@PROBES.register("grasp")
class GRASPProbe(VoEProbe):
    """VoE accuracy and AUROC on the matched pairs of the GRASP Level 2 tests."""

    def __init__(
        self,
        name: str,
        manifest: str,
        data_root: str,
        tests: str | Sequence[str] | None = None,
        verify_videos: bool = False,
        num_frames: int = 64,
        context_frames: int | Sequence[int] = (2, 4, 6, 8, 10, 12, 14, 16),
        distance: str = "l1",
        context: str = "video",
        target: str = "video_target",
        predictor: str = "pred",
        view: str = "resize",
        size: int = 224,
    ):
        super().__init__(
            name,
            data_root,
            num_frames,
            context_frames,
            distance,
            context,
            target,
            predictor,
            view,
            size,
        )
        selected = None if tests is None else [tests] if isinstance(tests, str) else list(tests)
        if selected is not None:
            unknown = [test for test in selected if test not in TESTS]
            if unknown:
                raise ValueError(f"unknown GRASP tests {unknown}; known tests: {sorted(TESTS)}")
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(f"tests must be a non-empty list without duplicates: {selected}")
        self.manifest = manifest
        self.tests = selected
        self.verify_videos = verify_videos
        self.selected_tests: list[str] = []
        self.first_frame_mismatches = 0

    def load_pairs(self) -> list[tuple[str, str, tuple[str, ...]]]:
        root = Path(self.data_root)
        doc = json.loads(Path(self.manifest).read_text())
        if doc.get("benchmark") != "grasp":
            raise ValueError(f"{self.manifest} is not a GRASP manifest")
        wanted = set(self.tests) if self.tests is not None else set(doc["tests"])
        absent = sorted(wanted - set(doc["tests"]))
        if absent:
            raise ValueError(f"tests {absent} are not in the manifest ({sorted(doc['tests'])})")
        pairs = [pair for pair in doc["pairs"] if pair["test"] in wanted]
        paths = sorted({rel for pair in pairs for rel in (pair["possible"], pair["impossible"])})
        missing = [rel for rel in paths if not (root / rel).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(paths)} GRASP videos are missing under {root}, "
                f"e.g. {missing[:3]}"
            )
        if self.verify_videos:
            altered = [
                rel for rel in paths if sha256_file(root / rel) != doc["videos"][rel]["sha256"]
            ]
            if altered:
                raise ValueError(
                    f"{len(altered)} of {len(paths)} GRASP videos differ from the manifest, "
                    f"e.g. {altered[:3]}"
                )
        self.selected_tests = sorted(wanted)
        self.first_frame_mismatches = sum(not pair["first_frame_match"] for pair in pairs)
        return [
            (
                str(root / pair["possible"]),
                str(root / pair["impossible"]),
                pair_groups(pair["test"]),
            )
            for pair in pairs
        ]

    def metadata(self) -> dict:
        if self.pairs is None:
            return {}
        return {
            **super().metadata(),
            "tests": list(self.selected_tests),
            "n_tests": len(self.selected_tests),
            "first_frame_mismatches": self.first_frame_mismatches,
        }
