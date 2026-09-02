"""Pure scoring functions for the Tartaglini et al. shape-bias benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShapeBiasResult:
    """Aggregate metrics and one compact diagnostic row per anchor."""

    metrics: dict[str, Any]
    anchors: list[dict[str, Any]]


def _normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("expected embeddings with shape [n_shapes, n_textures, hidden_dim]")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cosine similarity is undefined for zero embeddings")
    return array / norms


def score_shape_bias(
    embeddings: np.ndarray,
    *,
    shape_names: list[str] | None = None,
    texture_names: list[str] | None = None,
) -> ShapeBiasResult:
    """Score every valid novel-stimulus triplet by cosine similarity.

    For anchor ``(s, t)``, every ``(s, t')`` with ``t' != t`` is compared
    against every ``(s', t)`` with ``s' != s``. A decision is shape-based
    only when the shape-match cosine is strictly greater. Ties therefore
    count as texture decisions, matching Tartaglini et al. (2022).
    """
    normalized = _normalized_embeddings(embeddings)
    n_shapes, n_textures, _ = normalized.shape
    if n_shapes < 2 or n_textures < 2:
        raise ValueError("at least two shapes and two textures are required")
    shape_names = shape_names or [str(index) for index in range(n_shapes)]
    texture_names = texture_names or [str(index) for index in range(n_textures)]
    if len(shape_names) != n_shapes or len(texture_names) != n_textures:
        raise ValueError("embedding axes do not match the supplied names")

    # For each anchor (s, t), compare its same-shape candidates (s, u) and
    # same-texture candidates (v, t). The masks remove the anchor itself.
    shape_similarities = np.einsum("std,sud->stu", normalized, normalized)
    texture_similarities = np.einsum("std,vtd->stv", normalized, normalized)
    shape_mask = np.broadcast_to(
        ~np.eye(n_textures, dtype=bool),
        (n_shapes, n_textures, n_textures),
    )
    texture_mask = np.broadcast_to(
        ~np.eye(n_shapes, dtype=bool)[:, None, :],
        (n_shapes, n_textures, n_shapes),
    )
    shape_candidates = shape_similarities[shape_mask].reshape(n_shapes, n_textures, n_textures - 1)
    texture_candidates = texture_similarities[texture_mask].reshape(
        n_shapes, n_textures, n_shapes - 1
    )
    margins = shape_candidates[:, :, :, None] - texture_candidates[:, :, None, :]
    shape_decisions_by_anchor = np.sum(margins > 0, axis=(2, 3))
    ties_by_anchor = np.sum(margins == 0, axis=(2, 3))

    trials_per_anchor = (n_shapes - 1) * (n_textures - 1)
    shape_decisions = int(shape_decisions_by_anchor.sum())
    ties = int(ties_by_anchor.sum())
    n_triplets = n_shapes * n_textures * trials_per_anchor
    texture_decisions = n_triplets - shape_decisions

    anchors: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(shape_names):
        for texture_index, texture in enumerate(texture_names):
            anchor_shape_decisions = int(shape_decisions_by_anchor[shape_index, texture_index])
            anchor_ties = int(ties_by_anchor[shape_index, texture_index])
            anchor_margins = margins[shape_index, texture_index]
            anchors.append(
                {
                    "shape": shape,
                    "texture": texture,
                    "n_triplets": trials_per_anchor,
                    "shape_decisions": anchor_shape_decisions,
                    "texture_decisions": trials_per_anchor - anchor_shape_decisions,
                    "ties": anchor_ties,
                    "shape_bias": anchor_shape_decisions / trials_per_anchor,
                    "mean_cosine_margin": float(anchor_margins.mean()),
                }
            )

    metrics = {
        "protocol": "tartaglini_exp3_novel_80pct_unaligned",
        "decision_rule": "shape_cosine_gt_texture_cosine; ties_are_texture",
        "n_shapes": n_shapes,
        "n_textures": n_textures,
        "n_stimuli": n_shapes * n_textures,
        "n_triplets": n_triplets,
        "shape_decisions": shape_decisions,
        "texture_decisions": texture_decisions,
        "ties": ties,
        "shape_bias": shape_decisions / n_triplets,
        "texture_bias": texture_decisions / n_triplets,
        "mean_cosine_margin": float(margins.mean()),
    }
    return ShapeBiasResult(metrics=metrics, anchors=anchors)
