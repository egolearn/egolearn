"""EgoLearn probe for the novel-object shape-bias benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from egolearn.config.registry import PROBES, VIEWS
from egolearn.core import keys as K
from egolearn.data import views as core_views
from egolearn.evaluation.probe import Probe

from .score import ShapeBiasResult, score_shape_bias
from .stimuli import (
    load_stimulus_manifest,
    ordered_stimulus_records,
    sha256_file,
    validate_stimulus_manifest,
    write_json,
)


@PROBES.register("shape_bias")
class ShapeBiasProbe(Probe):
    """Measure shape-vs-texture preference in one image representation.

    The 256 prepared stimuli are encoded once per probe run. All 57,600
    triplet decisions are then derived from those cached in-memory features.
    """

    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        manifest: str,
        batch_size: int = 32,
        view: str = "resize",
        size: int = 224,
        verify_hashes: bool = True,
        output_json: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.name = name
        self.manifest_path = Path(manifest)
        self.stimulus_root = self.manifest_path.parent
        self.manifest = load_stimulus_manifest(self.manifest_path)
        validate_stimulus_manifest(
            self.manifest,
            self.stimulus_root,
            verify_hashes=verify_hashes,
        )
        self.records = ordered_stimulus_records(self.manifest)
        self.batch_size = int(batch_size)
        view_builder = getattr(core_views, view, None) or VIEWS.get(view)
        self.transform = view_builder(size=size)
        self.output_json = output_json

    @torch.no_grad()
    def evaluate(self, model, device) -> ShapeBiasResult:
        """Return the complete result, including per-anchor diagnostics."""
        batches: list[np.ndarray] = []
        for start in range(0, len(self.records), self.batch_size):
            tensors = []
            for record in self.records[start : start + self.batch_size]:
                with Image.open(self.stimulus_root / record["path"]) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            inputs = torch.stack(tensors).to(device)
            features = model.encode("image", inputs).pooled
            if not isinstance(features, torch.Tensor):
                raise ValueError("pooled image encoder output is not a tensor")
            if features.ndim != 2 or features.shape[0] != inputs.shape[0]:
                raise ValueError(
                    f"shape-bias features must be [batch, hidden_dim], got {features.shape}"
                )
            batches.append(features.float().cpu().numpy())

        embeddings = np.concatenate(batches, axis=0)
        if embeddings.shape[0] != 256 or not np.isfinite(embeddings).all():
            raise ValueError("expected 256 finite stimulus embeddings")
        grid = embeddings.reshape(
            len(self.manifest["shapes"]),
            len(self.manifest["textures"]),
            -1,
        )
        return score_shape_bias(
            grid,
            shape_names=self.manifest["shapes"],
            texture_names=self.manifest["textures"],
        )

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        result = self.evaluate(model, ctx.device)
        if self.output_json:
            step = getattr(getattr(ctx, "clock", None), "step", 0)
            output_path = self.output_json.format(step=step)
            write_json(
                output_path,
                {
                    **result.metrics,
                    "manifest": str(self.manifest_path),
                    "manifest_sha256": sha256_file(self.manifest_path),
                    "anchors": result.anchors,
                },
            )
        return {
            "shape_bias": float(result.metrics["shape_bias"]),
            "shape_bias_texture": float(result.metrics["texture_bias"]),
            "shape_bias_margin": float(result.metrics["mean_cosine_margin"]),
            "shape_bias_ties": float(result.metrics["ties"]),
            "shape_bias_n": float(result.metrics["n_triplets"]),
        }
