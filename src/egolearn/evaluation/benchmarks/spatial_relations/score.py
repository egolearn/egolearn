"""Cosine relation matching and balanced spatial categorization accuracy."""

from __future__ import annotations

from itertools import product

import numpy as np

from egolearn.evaluation.benchmarks.spatial_relations.prepare import (
    RELATIONS,
    RENDERERS,
    TARGET_CONDITIONS,
)

DOMAINS = {
    "abstract_2d": {"label": "2D abstract scenes", "experiment": "exp1", "renderers": RENDERERS},
    "rendered_3d": {"label": "3D rendered scenes", "experiment": "exp3", "renderers": ("blender",)},
}
AGGREGATION = "equal_domains_equal_relation_target_renderer_cells_v1"


def score_trials(features, records, trials, tie_tolerance=1e-6):
    """Cosine margins and half-credit ties, independent of candidate order."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(records) or not trials:
        raise ValueError("expected nonempty trials and [n_images, hidden_dim] embeddings")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms == 0):
        raise ValueError("expected finite nonzero embeddings")
    features = features / norms
    lookup = {record["path"]: i for i, record in enumerate(records)}
    indices = np.array(
        [
            [
                lookup[row["anchor"]],
                lookup[row["candidates"][row["correct_index"]]],
                lookup[row["candidates"][1 - row["correct_index"]]],
            ]
            for row in trials
        ]
    )
    anchor, same, other = (features[indices[:, i]] for i in range(3))
    margins = (anchor * (same - other)).sum(axis=1)
    ties = np.abs(margins) <= tie_tolerance
    scores = np.where(ties, 0.5, (margins > 0).astype(float))
    return scores, margins


def aggregate_scores(domains):
    """Equal domain weight, then equal renderer x relation x target cell weight.

    Input maps each descriptive domain ID to (trials, per-trial scores).
    Scores are fractions with half-credit ties. Missing cells fail closed;
    neither trial count nor renderer count can change a domain's weight.
    """
    if set(domains) != set(DOMAINS):
        raise ValueError("both abstract_2d and rendered_3d domains are required")
    metrics = {}
    for domain, (trials, values) in domains.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(trials),) or not np.isin(values, [0, 0.5, 1]).all():
            raise ValueError("expected one score of 0, 0.5, or 1 per trial")
        groups: dict[tuple[str, str, str], list[int]] = {}
        for i, row in enumerate(trials):
            key = row["renderer"], row["relation"], row["target_condition"]
            groups.setdefault(key, []).append(i)
        expected = set(product(DOMAINS[domain]["renderers"], RELATIONS, TARGET_CONDITIONS))
        if set(groups) != expected:
            raise ValueError(f"incomplete or unknown condition cells in {domain}")
        cells = {key: float(values[indices].mean()) for key, indices in groups.items()}
        metrics[f"{domain}/accuracy"] = float(np.mean(list(cells.values())))
        metrics[f"{domain}/n_trials"] = float(len(trials))
        for (renderer, relation, target), accuracy in cells.items():
            target_label = "same_object" if target == "identical" else "changed_object"
            metrics[f"{domain}/cells/{renderer}/{relation}/{target_label}/accuracy"] = accuracy
        for relation in RELATIONS:
            metrics[f"{domain}/{relation}/accuracy"] = float(
                np.mean([v for k, v in cells.items() if k[1] == relation])
            )
        for target, label in (("identical", "same_object"), ("different", "changed_object")):
            metrics[f"{domain}/{label}/accuracy"] = float(
                np.mean([v for k, v in cells.items() if k[2] == target])
            )
        metrics[f"{domain}/object_change_gap"] = (
            metrics[f"{domain}/same_object/accuracy"] - metrics[f"{domain}/changed_object/accuracy"]
        )
    metrics["accuracy"] = float(np.mean([metrics[f"{d}/accuracy"] for d in DOMAINS]))
    for label in (*RELATIONS, "same_object", "changed_object"):
        metrics[f"{label}/accuracy"] = float(
            np.mean([metrics[f"{d}/{label}/accuracy"] for d in DOMAINS])
        )
    metrics["n_trials"] = sum(metrics[f"{d}/n_trials"] for d in DOMAINS)
    return metrics
