"""EgoLearn probe for the Konkle objects 4AFC benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from egolearn.config.registry import PROBES, VIEWS
from egolearn.core import keys as K
from egolearn.data.text import VocabTokenizer
from egolearn.evaluation.probe import Probe

CHANCE = 0.25


def sample_trials(
    pool_sizes: dict[str, int], n_trials: int = 100, seed: int = 0
) -> dict[str, list[list[tuple[str, int]]]]:
    """Draw the trials of every category in ``pool_sizes``.

    Returns ``{category: [[(category, image index), ...4 candidates...], ...]}``
    with the target first. The order of random draws reproduces
    ``eval_4afc_region`` in mcfrank/vlm-headcam: categories are visited in
    sorted order, and each trial draws the target image, then three foil
    categories without replacement, then one image from each foil category.
    """
    if len(pool_sizes) < 4:
        raise ValueError("4AFC needs at least four categories")
    rng = np.random.default_rng(seed)
    categories = sorted(pool_sizes)
    trials: dict[str, list[list[tuple[str, int]]]] = {}
    for category in categories:
        others = [c for c in categories if c != category]
        trials[category] = []
        for _ in range(n_trials):
            target = int(rng.choice(pool_sizes[category]))
            foils = [others[int(i)] for i in rng.choice(len(others), 3, replace=False)]
            candidates = [(category, target)]
            candidates += [(foil, int(rng.choice(pool_sizes[foil]))) for foil in foils]
            trials[category].append(candidates)
    return trials


@PROBES.register("konkle_4afc")
class Konkle4AFCProbe(Probe):
    """Match each category word to one of four object photographs.

    Images are encoded once per run. Categories whose label is not a single
    token of the model vocabulary score chance and are excluded from the foil
    pool; ``four_afc_acc`` averages over every category of the manifest.
    """

    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION, K.CAP_TEXT_REPRESENTATION)

    def __init__(
        self,
        name: str,
        manifest: str,
        vocab_json: str,
        vision_head: str,
        size: int = 224,
        view: str = "resize",
        n_trials: int = 100,
        seed: int = 0,
        batch_size: int = 64,
        output_json: str | None = None,
    ) -> None:
        if n_trials <= 0 or batch_size <= 0:
            raise ValueError("n_trials and batch_size must be positive")
        self.name = name
        raw = Path(manifest).read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.categories: dict[str, list[dict]] = json.loads(raw)["categories"]
        self.tokenizer = VocabTokenizer.from_json(vocab_json, add_sos_eos=False)
        self.vision_head = vision_head
        self.transform = VIEWS.get(view)(size=size)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.output_json = output_json

    def metadata(self) -> dict:
        return {
            "manifest_sha256": self.manifest_sha256,
            "n_categories": len(self.categories),
            "n_images": sum(len(images) for images in self.categories.values()),
            "n_trials_per_category": self.n_trials,
            "trial_seed": self.seed,
        }

    def _in_vocabulary(self, category: str) -> bool:
        return category in self.tokenizer.vocab and len(category.split()) == 1

    @torch.no_grad()
    def _image_features(self, model, device, paths: list[str]) -> torch.Tensor:
        features = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as handle:
                    images.append(self.transform(handle.convert("RGB")))
            pooled = model.encode("image", torch.stack(images).to(device)).pooled
            features.append(F.normalize(model.apply_head(self.vision_head, pooled), dim=1))
        return torch.cat(features)

    @torch.no_grad()
    def per_category(self, model, device) -> dict[str, float]:
        """Per-category accuracy; out-of-vocabulary categories are set to chance."""
        scored = sorted(c for c in self.categories if self._in_vocabulary(c))
        result = dict.fromkeys(self.categories, CHANCE)
        if len(scored) < 4:
            return result
        offsets: dict[str, int] = {}
        paths: list[str] = []
        for category in scored:
            offsets[category] = len(paths)
            paths += [image["path"] for image in self.categories[category]]
        image_features = self._image_features(model, device, paths)
        ids, lengths = self.tokenizer.batch(scored)
        text = F.normalize(model.encode("text", ids.to(device), lengths.to(device)).pooled, dim=1)
        trials = sample_trials(
            {c: len(self.categories[c]) for c in scored}, self.n_trials, self.seed
        )
        for row, category in enumerate(scored):
            rows = torch.tensor(
                [[offsets[c] + i for c, i in trial] for trial in trials[category]], device=device
            )
            scores = image_features[rows] @ text[row]
            result[category] = int((scores.argmax(dim=1) == 0).sum()) / self.n_trials
        return result

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        accuracy = self.per_category(model, ctx.device)
        scored = [c for c in accuracy if self._in_vocabulary(c)]
        if self.output_json is not None:
            step = getattr(getattr(ctx, "clock", None), "step", 0)
            output = Path(self.output_json.format(step=step))
            output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "per_category": accuracy,
                "out_of_vocabulary": sorted(set(accuracy) - set(scored)),
            }
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return {
            "four_afc_acc": sum(accuracy.values()) / len(accuracy),
            "four_afc_acc_in_vocab": (
                sum(accuracy[c] for c in scored) / len(scored) if len(scored) >= 4 else CHANCE
            ),
            "four_afc_categories": float(len(accuracy)),
            "four_afc_oov_categories": float(len(accuracy) - len(scored)),
        }
