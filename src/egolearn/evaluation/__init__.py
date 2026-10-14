"""Benchmark contracts and lazy standalone evaluation entry points."""

from typing import TYPE_CHECKING

from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.probe import Probe

if TYPE_CHECKING:
    from egolearn.evaluation.checkpoint import evaluate_checkpoint, load_model
    from egolearn.evaluation.evaluator import (
        EvaluationContext,
        EvaluationError,
        EvaluationResult,
        evaluate,
    )

__all__ = [
    "Benchmark",
    "EvaluationContext",
    "EvaluationError",
    "EvaluationResult",
    "Probe",
    "evaluate",
    "evaluate_checkpoint",
    "load_model",
]


def __getattr__(name: str):
    if name in {"EvaluationContext", "EvaluationError", "EvaluationResult", "evaluate"}:
        from egolearn.evaluation import evaluator

        return getattr(evaluator, name)
    if name in {"evaluate_checkpoint", "load_model"}:
        from egolearn.evaluation import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
