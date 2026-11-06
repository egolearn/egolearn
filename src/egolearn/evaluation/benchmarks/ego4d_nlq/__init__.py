"""Ego4D Natural Language Queries (benchmark:ego4d_nlq); see docs/evaluation/ego4d_nlq.md."""

from .benchmark import Ego4DNLQBenchmark

__all__ = ["Ego4DNLQBenchmark"]

BENCHMARK_METADATA = {
    "name": "Ego4D NLQ",
    "source": "[Ego4D release](https://ego4d-data.org/docs/start-here/)",
    "license": "Data: Ego4D license agreement; ported evaluation code: MIT",
    "access": "upstream-licensed",
    "prepare": (
        "No preparer; download v2 NLQ annotations and clips with the Ego4D CLI "
        "([data guide](../evaluation/ego4d_nlq.md#downloading-the-data))."
    ),
}
