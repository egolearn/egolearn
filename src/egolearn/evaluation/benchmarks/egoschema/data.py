"""EgoSchema questions and public answers.

The release (github.com/egoschema/EgoSchema at ``EGOSCHEMA_COMMIT``) ships
``questions.json`` -- 5,031 five-way questions, each about one 3-minute video
named by its ``q_uid`` -- and ``subset_answers.json``, the public answers for
500 of them. The other answers are scored only by the benchmark's servers.
Loaders verify both files against their pinned SHA-256 so every run scores the
same questions; ``verify=False`` exists for synthetic test fixtures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

EGOSCHEMA_COMMIT = "505c787376b5e066d0ae406d0e0d41245cebba15"
QUESTIONS_SHA256 = "92c61d06c026455ce5320fa1fcc6cf43b77d874b708076f6b87abfef6fcb08a8"
ANSWERS_SHA256 = "a38d0aefcf8bcd5fb8888c7598b4550dc9ac6f6ad11cb2e1eae7b93e4c72468d"
NUM_QUESTIONS = 5031
NUM_ANSWERS = 500
NUM_OPTIONS = 5


@dataclass(frozen=True)
class EgoSchemaQuestion:
    q_uid: str
    question: str
    options: tuple[str, ...]


def _read_verified(path: str | Path, expected: str, verify: bool, what: str) -> bytes:
    raw = Path(path).read_bytes()
    if verify:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError(
                f"EgoSchema {what} has an unexpected SHA-256: expected {expected}, got "
                f"{digest} (use the files at EgoSchema commit {EGOSCHEMA_COMMIT})"
            )
    return raw


def load_questions(path: str | Path, verify: bool = True) -> list[EgoSchemaQuestion]:
    """All questions of ``questions.json`` in file order, five options each."""
    rows = json.loads(_read_verified(path, QUESTIONS_SHA256, verify, "questions.json"))
    questions = []
    for row in rows:
        options = tuple(str(row[f"option {i}"]).strip() for i in range(NUM_OPTIONS))
        questions.append(
            EgoSchemaQuestion(
                q_uid=str(row["q_uid"]), question=str(row["question"]).strip(), options=options
            )
        )
    uids = [q.q_uid for q in questions]
    if len(set(uids)) != len(uids):
        raise ValueError("EgoSchema questions.json repeats a q_uid")
    if verify and len(questions) != NUM_QUESTIONS:
        raise ValueError(f"expected {NUM_QUESTIONS} EgoSchema questions, got {len(questions)}")
    return questions


def load_answers(path: str | Path, verify: bool = True) -> dict[str, int]:
    """Public answers of ``subset_answers.json``: q_uid -> option index 0-4."""
    raw = json.loads(_read_verified(path, ANSWERS_SHA256, verify, "subset_answers.json"))
    answers = {str(uid): int(answer) for uid, answer in raw.items()}
    bad = {uid: a for uid, a in answers.items() if not 0 <= a < NUM_OPTIONS}
    if bad:
        raise ValueError(f"EgoSchema answers outside 0-{NUM_OPTIONS - 1}: {list(bad.items())[:3]}")
    if verify and len(answers) != NUM_ANSWERS:
        raise ValueError(f"expected {NUM_ANSWERS} EgoSchema answers, got {len(answers)}")
    return answers
