"""EgoSchema: five-way questions about 3-minute egocentric videos, answered by a generative model.

The benchmark loads and verifies the release files, turns every question into
a ``MultipleChoiceItem`` over its whole video and leaves prompting, reply
parsing and accuracy to the generic scorer (evaluation/multiple_choice.py).

By default it scores the 500 questions with public answers.  With
``subset_only: false`` it scores all 5,031, and ``submission_dir`` receives
``submission.csv`` (Kaggle ``egoschema-public``: header ``q_uid,answer``, the
answer an index 0-4) and ``submission.json`` (the authors' validation server
format); a question whose reply names no option is submitted as 0.
``predictions.jsonl`` there holds every scored question's reply and
prediction.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from egolearn.config.registry import BENCHMARKS
from egolearn.core import keys as K
from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.benchmarks.egoschema.data import (
    QUESTIONS_SHA256,
    EgoSchemaQuestion,
    load_answers,
    load_questions,
)
from egolearn.evaluation.multiple_choice import MultipleChoiceItem, MultipleChoiceScorer
from egolearn.evaluation.video_input import VideoClip, VideoSampling


@BENCHMARKS.register("egoschema")
class EgoSchemaBenchmark(Benchmark):
    """EgoSchema accuracy of a generative model; see docs/evaluation/egoschema.md."""

    name = "egoschema"
    version = "1"
    requires_capabilities = (K.CAP_TEXT_GENERATION,)

    def __init__(
        self,
        name: str,
        questions_json: str,
        answers_json: str,
        video_root: str,
        generative: str = "text",
        fps: float = 1.0,
        max_frames: int = 64,
        max_side: int | None = 448,
        batch_size: int = 8,
        workers: int = 8,
        subset_only: bool = True,
        max_questions: int | None = None,
        submission_dir: str | None = None,
        bootstrap_resamples: int = 10_000,
        verify: bool = True,
    ):
        if max_questions is not None and max_questions < 1:
            raise ValueError("max_questions must be >= 1")
        if max_questions is not None and not subset_only and submission_dir is not None:
            raise ValueError("submission files need all 5,031 questions; unset max_questions")
        self.name = name
        self.questions_json = questions_json
        self.answers_json = answers_json
        self.video_root = video_root
        self.subset_only = subset_only
        self.max_questions = max_questions
        self.submission_dir = submission_dir
        self.verify = verify
        self.scorer = MultipleChoiceScorer(
            name,
            generative=generative,
            sampling=VideoSampling(fps=fps, max_frames=max_frames, max_side=max_side),
            batch_size=batch_size,
            workers=workers,
            bootstrap_resamples=bootstrap_resamples,
        )
        self.questions: list[EgoSchemaQuestion] | None = None
        self.answers: dict[str, int] | None = None

    def video_path(self, q_uid: str) -> Path:
        return Path(self.video_root) / f"{q_uid}.mp4"

    def prepare(self) -> None:
        if self.questions is not None:
            return
        questions = load_questions(self.questions_json, verify=self.verify)
        answers = load_answers(self.answers_json, verify=self.verify)
        unknown = set(answers) - {q.q_uid for q in questions}
        if unknown:
            raise ValueError(
                f"{len(unknown)} answers name unknown questions, e.g. {sorted(unknown)[:3]}"
            )
        if self.subset_only:
            questions = [q for q in questions if q.q_uid in answers]
        if self.max_questions is not None:
            questions = questions[: self.max_questions]
        missing = [q.q_uid for q in questions if not self.video_path(q.q_uid).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(questions)} EgoSchema videos are missing under "
                f"{self.video_root}, e.g. {[f'{uid}.mp4' for uid in missing[:3]]}"
            )
        self.questions = questions
        self.answers = answers

    def configuration(self) -> dict[str, Any]:
        return {
            **self.scorer.configuration(),
            "name": self.name,
            "subset_only": self.subset_only,
            "max_questions": self.max_questions,
            "verify": self.verify,
        }

    def metadata(self) -> dict[str, Any]:
        if self.questions is None or self.answers is None:
            return {}
        return {
            "n_questions": len(self.questions),
            "n_answered": sum(q.q_uid in self.answers for q in self.questions),
            "questions_sha256": QUESTIONS_SHA256 if self.verify else "unverified",
            **self.scorer.video_summary,
        }

    def run(self, model, ctx) -> dict[str, float]:
        self.prepare()
        assert self.questions is not None and self.answers is not None
        items = [
            MultipleChoiceItem(
                uid=q.q_uid,
                question=q.question,
                options=q.options,
                video=VideoClip(str(self.video_path(q.q_uid))),
                answer=self.answers.get(q.q_uid),
            )
            for q in self.questions
        ]
        metrics, rows = self.scorer.evaluate(model, ctx, items)
        if self.submission_dir is not None:
            self._write_outputs(rows)
        return metrics

    def _write_outputs(self, rows: list[dict[str, Any]]) -> None:
        assert self.submission_dir is not None
        out = Path(self.submission_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        if self.subset_only:
            return
        submission = {row["uid"]: 0 if row["pred"] is None else int(row["pred"]) for row in rows}
        (out / "submission.json").write_text(json.dumps(submission, indent=1) + "\n")
        table = io.StringIO()
        writer = csv.writer(table, lineterminator="\n")
        writer.writerow(["q_uid", "answer"])
        writer.writerows(submission.items())
        (out / "submission.csv").write_text(table.getvalue())
