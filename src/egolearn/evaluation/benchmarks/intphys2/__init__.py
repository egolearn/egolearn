"""IntPhys 2 violation-of-expectation benchmark; see docs/evaluation/intphys2.md."""

from egolearn.evaluation.benchmarks.intphys2.probe import IntPhys2Probe

__all__ = ["IntPhys2Probe"]

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "IntPhys 2",
    "source": "[IntPhys 2 release](https://github.com/facebookresearch/IntPhys2)",
    "license": (
        "[CC BY-NC 4.0 with additional evaluation-only restrictions]"
        "(https://github.com/facebookresearch/IntPhys2/blob/main/LICENSE.md)"
    ),
    "access": "upstream-licensed",
    "prepare": (
        "No preparer; unpack the Main split and supply `data_root` ([data "
        "guide](../evaluation/intphys2.md))."
    ),
}
