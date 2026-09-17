"""Training-independent benchmark contract (no optional dependencies at import)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from egolearn.evaluation.evaluator import EvaluationContext
    from egolearn.model.model import ComposedModel


class Benchmark:
    """An evaluation protocol over a model, with explicit dataset preparation.

    Constructors should validate configuration without enumerating datasets.
    ``prepare`` may cache dataset metadata; learned classifiers/features must
    be recomputed for each model. ``run`` must leave model weights unchanged.
    Version the protocol when scoring or preprocessing semantics change.
    """

    name = "benchmark"
    version = "1"
    requires_capabilities: tuple[str, ...] = ()

    def prepare(self) -> None:
        """Load/validate data before execution; repeated calls should be cheap."""

    def configuration(self) -> dict[str, Any]:
        """Describe configured instances; config-driven calls also record defaults."""
        return {}

    def metadata(self) -> dict[str, Any]:
        """Dataset sizes, split identity, or other protocol-specific provenance."""
        return {}

    def run(self, model: ComposedModel, ctx: EvaluationContext) -> dict[str, float]:
        raise NotImplementedError
