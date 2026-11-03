"""Generic multiple-choice scorer for generative models.

A benchmark builds ``MultipleChoiceItem``s (question, options, optional video,
answer index or None when hidden) and this scorer turns each into one prompt,
calls ``model.generate`` and parses the reply.  It needs ``text_generation``.

The prompt (``PROMPT_VERSION`` 1) is the question, every option behind a
capital letter, and the instruction usual for letter answers in LLaVA-style
evaluations (lmms-eval's EgoSchema task among them)::

    Question: <question>
    Options:
    A. <option 0>
    ...
    Answer with the option's letter from the given choices directly.

A reply is read as the option whose letter it starts with, alone or followed by
".", ":", ")" or "]" (``"B"``, ``"(B)"``, ``"B. ..."``), else the capital letter
after "answer" or "answer is", else the option whose full text it equals.  A
reply that starts with a word, such as "C is cooking", names no letter (in
EgoSchema, C is the camera wearer).  An unreadable reply counts as wrong and is reported as
``unparsed``.  Accuracy counts items with an answer and carries a percentile
bootstrap interval: each resampled mean of n 0/1 outcomes is drawn directly as
Binomial(n, acc) / n, which has exactly the distribution of a resampled mean.
``pred_frac_<letter>`` is each letter's share of all predictions, which
exposes a letter bias.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from egolearn.core import keys as K
from egolearn.evaluation.video_input import (
    VideoClip,
    VideoReader,
    VideoSampling,
    shown,
    shown_summary,
)

PROMPT_VERSION = 1
LETTERS = "ABCDEFGH"
INSTRUCTION = "Answer with the option's letter from the given choices directly."


@dataclass(frozen=True)
class MultipleChoiceItem:
    uid: str
    question: str
    options: tuple[str, ...]
    video: VideoClip | None = None
    answer: int | None = None  # index into options; None when the answer is hidden

    def __post_init__(self) -> None:
        if not 2 <= len(self.options) <= len(LETTERS):
            raise ValueError(f"{self.uid}: need 2 to {len(LETTERS)} options")
        if self.answer is not None and not 0 <= self.answer < len(self.options):
            raise ValueError(f"{self.uid}: answer {self.answer} is not an option index")


def format_prompt(question: str, options: Sequence[str]) -> str:
    lines = [f"Question: {question.strip()}", "Options:"]
    lines += [f"{LETTERS[i]}. {str(option).strip()}" for i, option in enumerate(options)]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def parse_choice(reply: str, options: Sequence[str]) -> int | None:
    """The option index a reply names, or None (see the module docstring)."""
    letters = LETTERS[: len(options)]
    text = reply.strip()
    lead = re.match(rf"^[\s(\[]*([{letters}])(?:[.:)\]]|$)", text)
    if lead:
        return letters.index(lead.group(1))
    stated = re.search(rf"(?i:answer)(?:\s+is)?\s*:?\s*\(?([{letters}])\b", text)
    if stated:
        return letters.index(stated.group(1))
    normalized = text.rstrip(".").strip().lower()
    for index, option in enumerate(options):
        if normalized == str(option).strip().rstrip(".").lower():
            return index
    return None


def bootstrap_interval(
    correct: np.ndarray, resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float]:
    n = int(correct.size)
    rng = np.random.default_rng(seed)
    means = rng.binomial(n, float(correct.mean()), size=resamples) / n
    low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(low), float(high)


class MultipleChoiceScorer:
    """Scores items with ``model.generate(generative, texts=, videos=)``."""

    requires_capabilities = (K.CAP_TEXT_GENERATION,)

    def __init__(
        self,
        name: str,
        generative: str = "text",
        sampling: VideoSampling | None = None,
        batch_size: int = 8,
        workers: int = 8,
        bootstrap_resamples: int = 10_000,
    ):
        if batch_size < 1 or bootstrap_resamples < 1:
            raise ValueError("batch_size and bootstrap_resamples must be >= 1")
        self.name = name
        self.generative = generative
        self.sampling = sampling or VideoSampling()
        self.batch_size = batch_size
        self.bootstrap_resamples = bootstrap_resamples
        self.reader = VideoReader(self.sampling, workers=workers)
        self.video_summary: dict[str, float] = {}

    def configuration(self) -> dict[str, Any]:
        return {
            "generative": self.generative,
            **asdict(self.sampling),
            "batch_size": self.batch_size,
            "workers": self.reader.workers,
            "bootstrap_resamples": self.bootstrap_resamples,
            "prompt_version": PROMPT_VERSION,
        }

    def evaluate(
        self, model, ctx, items: Sequence[MultipleChoiceItem]
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
            inputs: dict[str, Any] = {
                "texts": [format_prompt(item.question, item.options) for item in batch]
            }
            clips = [item.video for item in batch if item.video is not None]
            if clips and len(clips) != len(batch):
                raise ValueError(f"{self.name}: a batch mixes items with and without video")
            videos = self.reader.read(clips) if clips else []
            if videos:
                inputs["videos"] = [(video,) for video in videos]
            replies = model.generate(self.generative, **inputs)
            if len(replies) != len(batch):
                raise ValueError(f"{self.name}: {len(replies)} replies for {len(batch)} items")
            for i, (item, reply) in enumerate(zip(batch, replies, strict=True)):
                choice = parse_choice(str(reply), item.options)
                row = {"uid": item.uid, "reply": str(reply), "pred": choice, "answer": item.answer}
                rows.append(row | (shown(videos[i]) if videos else {}))
        self.video_summary = shown_summary(rows)
        return self._metrics(items, rows, ctx), rows

    def _metrics(self, items, rows, ctx) -> dict[str, float]:
        count = len(rows)
        metrics: dict[str, float] = {
            "n_scored": float(count),
            "unparsed": sum(row["pred"] is None for row in rows) / count,
        }
        answered = [row for row in rows if row["answer"] is not None]
        if answered:
            correct = np.array([row["pred"] == row["answer"] for row in answered], dtype=float)
            low, high = bootstrap_interval(
                correct, self.bootstrap_resamples, ctx.derive_seed(f"{self.name}:bootstrap")
            )
            metrics.update(
                acc=float(correct.mean()),
                acc_ci_low=low,
                acc_ci_high=high,
                n_answered=float(len(answered)),
            )
        width = max(len(item.options) for item in items)
        for index in range(width):
            share = sum(row["pred"] == index for row in rows) / count
            metrics[f"pred_frac_{LETTERS[index]}"] = share
        return metrics
