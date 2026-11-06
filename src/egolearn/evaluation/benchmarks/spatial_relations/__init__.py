"""Spatial relation categorization with 2D abstract and 3D rendered scenes."""

from egolearn.evaluation.benchmarks.spatial_relations.probe import SpatialRelationsProbe
from egolearn.evaluation.benchmarks.spatial_relations.score import aggregate_scores, score_trials

__all__ = ["SpatialRelationsProbe", "aggregate_scores", "score_trials"]

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "Spatial relations",
    "source": "[Davidson et al. (2024)](https://doi.org/10.1016/j.cognition.2023.105690)",
    "license": "Prepared collection: CC BY 4.0; see source for external assets",
    "access": "public",
    "prepare": (
        '`egl data prepare spatial-relations --output-dir "$OUT_DIR" --generator-root '
        '"$GENERATOR_ROOT" --exp3-image-root "$EXP3_IMAGE_ROOT"` '
        "([inputs](../evaluation/spatial_relations.md#preparing-and-checking-data))"
    ),
}
