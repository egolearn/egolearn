"""GRASP Level 2 violation-of-expectation benchmark; see docs/evaluation/grasp.md."""

from egolearn.evaluation.benchmarks.grasp.probe import GRASPProbe

__all__ = ["GRASPProbe"]

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "GRASP Level 2",
    "source": "[GRASP release](https://github.com/i-machine-think/grasp)",
    "license": "Code: MIT; videos: no separate license stated",
    "access": "upstream-licensed",
    "prepare": '`egl data prepare grasp --data_root "$DATA_ROOT" --output "$MANIFEST"`',
}
