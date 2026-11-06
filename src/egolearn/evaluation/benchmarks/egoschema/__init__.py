"""EgoSchema five-way video QA (benchmark:egoschema); see docs/evaluation/egoschema.md."""

from .benchmark import EgoSchemaBenchmark

__all__ = ["EgoSchemaBenchmark"]

BENCHMARK_METADATA = {
    "name": "EgoSchema",
    "source": "[EgoSchema release](https://github.com/egoschema/EgoSchema)",
    "license": "Annotations: see source; videos: Ego4D terms and Kaggle competition rules",
    "access": "upstream-licensed",
    "prepare": (
        "No preparer; download the pinned annotations and videos "
        "([data guide](../evaluation/egoschema.md#downloading-the-data))."
    ),
}
