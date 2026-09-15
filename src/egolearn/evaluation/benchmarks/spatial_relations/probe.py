"""Frozen-image spatial relation categorization for training-time evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from egolearn.config.registry import PROBES
from egolearn.core import keys as K
from egolearn.evaluation.benchmarks.shape_bias.stimuli import sha256_file, write_json
from egolearn.evaluation.benchmarks.spatial_relations.prepare import validate_dataset
from egolearn.evaluation.benchmarks.spatial_relations.preprocessing import spatial_transform
from egolearn.evaluation.benchmarks.spatial_relations.score import (
    AGGREGATION,
    DOMAINS,
    aggregate_scores,
    score_trials,
)
from egolearn.evaluation.probe import Probe


@PROBES.register("spatial_relations")
class SpatialRelationsProbe(Probe):
    """Two-choice relation matching, balanced across 2D and 3D scenes.

    Like other probes, the training evaluation hook supplies a model in eval
    mode. The encoder must provide pooled image features; no classifier is fit.
    """

    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        abstract_2d_manifest: str,
        rendered_3d_manifest: str,
        batch_size: int = 32,
        tie_tolerance: float = 1e-6,
        output_json: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError("tie_tolerance must be finite and nonnegative")
        self.name = name
        self.batch_size = batch_size
        self.tie_tolerance = tie_tolerance
        self.output_json = output_json
        self.transform = spatial_transform()
        self.datasets = {}
        profiles = set()
        for domain, path in (
            ("abstract_2d", abstract_2d_manifest),
            ("rendered_3d", rendered_3d_manifest),
        ):
            manifest_path = Path(path)
            if manifest_path.name != "manifest.json":
                raise ValueError("expected a prepared manifest.json")
            root = manifest_path.parent
            manifest = validate_dataset(root)
            if manifest["experiment"] != DOMAINS[domain]["experiment"]:
                raise ValueError(f"wrong dataset for {domain}")
            monitoring = manifest["source"].get("kind") == "monitoring_subset"
            profiles.add("monitoring" if monitoring else "full")
            trials = [
                json.loads(line)
                for line in (root / manifest["trials"]["path"]).read_text().splitlines()
            ]
            self.datasets[domain] = (root, manifest, trials, sha256_file(manifest_path))
        if len(profiles) != 1:
            raise ValueError("both domains must use the same full or monitoring profile")
        self.profile = profiles.pop()

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        scored = {}
        diagnostics = {}
        device = torch.device(ctx.device)
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device.type, enabled=False):
                for domain, (root, manifest, trials, digest) in self.datasets.items():
                    records = manifest["images"]
                    batches = []
                    for start in range(0, len(records), self.batch_size):
                        tensors = []
                        for record in records[start : start + self.batch_size]:
                            with Image.open(root / record["path"]) as image:
                                tensors.append(self.transform(image.convert("RGB")))
                        inputs = torch.stack(tensors).to(device)
                        features = model.encode("image", inputs).pooled
                        if (
                            not isinstance(features, torch.Tensor)
                            or features.ndim != 2
                            or features.shape[0] != len(inputs)
                        ):
                            raise ValueError("expected pooled image features [batch, hidden_dim]")
                        batches.append(features.float().cpu().numpy())
                    scores, margins = score_trials(
                        np.concatenate(batches), records, trials, self.tie_tolerance
                    )
                    scored[domain] = (trials, scores)
                    diagnostics[domain] = {
                        "label": DOMAINS[domain]["label"],
                        "manifest_sha256": digest,
                        "protocol": manifest["protocol"],
                        # The manifest hash identifies the complete image inventory.
                        "source": {
                            key: value
                            for key, value in manifest["source"].items()
                            if key != "images"
                        },
                        "n_images": len(records),
                        "mean_cosine_margin": float(margins.mean()),
                        "tie_rate": float((np.abs(margins) <= self.tie_tolerance).mean()),
                    }
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn
        metrics = aggregate_scores(scored)
        for domain, diagnostic in diagnostics.items():
            for key in ("n_images", "mean_cosine_margin", "tie_rate"):
                metrics[f"{domain}/{key}"] = float(diagnostic[key])
        metrics = {f"{self.name}/{key}": value for key, value in metrics.items()}
        if self.output_json:
            step = getattr(getattr(ctx, "clock", None), "step", 0)
            write_json(
                self.output_json.format(step=step),
                {
                    "benchmark": "spatial_relations",
                    "label": "Spatial relation categorization",
                    "profile": self.profile,
                    "aggregation": AGGREGATION,
                    "tie_tolerance": self.tie_tolerance,
                    "preprocessing": "RGB; bilinear resize 224; ImageNet normalization",
                    "feature": "model.encode('image').pooled",
                    "precision": "autocast/TF32 disabled; float64 cosine scoring",
                    "step": step,
                    "metrics": metrics,
                    "domains": diagnostics,
                },
            )
        return metrics
