"""Spatial relation categorization with 2D abstract and 3D rendered scenes."""

from egolearn.evaluation.benchmarks.spatial_relations.probe import SpatialRelationsProbe
from egolearn.evaluation.benchmarks.spatial_relations.score import aggregate_scores, score_trials

__all__ = ["SpatialRelationsProbe", "aggregate_scores", "score_trials"]
