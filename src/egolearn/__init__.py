"""Learning from egocentric video streams.

Streaming SSL, continual learning, replay, memory, and video QA.

The torch-heavy subsystems load on first use, so a base install (without the
`train` extra) can still import the package and read its version.
"""

__version__ = "0.1.0.dev0"


def __getattr__(name: str):
    if name in ("build", "validate", "Pipeline", "PipelineError"):
        # importing the subsystems populates the component registries
        from . import data, evaluation, memory, model, pipeline, training  # noqa: F401

        return getattr(pipeline, name)
    if name == "Context":
        from .core.context import Context

        return Context
    raise AttributeError(f"module 'egolearn' has no attribute {name!r}")
