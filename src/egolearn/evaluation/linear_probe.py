"""Fixed-feature L-BFGS linear probing: fit a classifier on frozen features, report
held-out top-1 -- the standard readout for comparing self-supervised
encoders (the roadmap's "one implementation, no apex dependency").

Protocol decisions, made once here so every model is probed identically:

* features are the named encoder's POOLED output on eval-preprocessed
  images; the preprocessing (`view`/`size`) is explicit config, like
  four_afc.  An optional `head:` probes a head's output instead; the
  default probes the raw representation.
* features are standardized by the train split's mean/std by default;
  ``feature_norm: l2`` instead scales each feature vector to unit length
  (the BabyCL paper's probes).
* the classifier is multinomial logistic regression, ZERO-initialized and
  fitted with full-batch L-BFGS + L2 penalty on the weights (never the
  bias).  ``l2`` weights ``||W||^2`` against the MEAN cross-entropy;
  ``c`` instead states the penalty as scikit-learn's inverse strength,
  ``C * sum(CE) + ||W||^2 / 2``, i.e. ``l2 = 1 / (2 * C * n_train)``, so a
  protocol written against scikit-learn keeps its constant.  A convex problem with a
  deterministic solver: the probe consumes NO randomness at all, so probing
  mid-training cannot perturb the run (covered by a test) and the same
  checkpoint always scores the same number.
* the train/test split is FROZEN by the benchmark adapter or JSON manifest,
  never drawn at probe time.

JSON remains available as a loader for custom datasets and existing configs::

    {"classes": ["apple", ...],
     "train": [{"image": "/abs/path.jpg", "label": 0}, ...],
     "test":  [{"image": "/abs/path.jpg", "label": 3}, ...]}
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from egolearn.config.registry import PROBES, VIEWS
from egolearn.core import keys as K
from egolearn.evaluation.probe import Probe

FEATURE_NORMS = ("standardize", "l2")


@dataclass(frozen=True)
class LabeledImage:
    image: str
    label: int


@dataclass(frozen=True)
class ClassificationSplit:
    """In-memory dataset contract consumed by the generic linear scorer.

    Benchmark adapters construct this directly.  JSON manifests are only one
    loader for the contract, retained for existing configs and custom datasets.
    """

    classes: tuple[str, ...]
    train: tuple[LabeledImage, ...]
    test: tuple[LabeledImage, ...]

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("linear-probe class list is empty")
        for split_name, items in (("train", self.train), ("test", self.test)):
            if not items:
                raise ValueError(f"linear-probe {split_name} split is empty")
            for item in items:
                if not 0 <= item.label < len(self.classes):
                    raise ValueError(
                        f"linear-probe {split_name} label {item.label} is outside "
                        f"[0, {len(self.classes) - 1}]"
                    )

    def metadata(self) -> dict:
        """Identify ordered split membership without recording machine-specific roots.

        The digest covers class names, relative image paths and labels, not image bytes.
        """
        # Stream the identity so ImageNet-sized splits need no second full manifest.
        parents = (Path(item.image).absolute().parent for item in chain(self.train, self.test))
        root = next(parents)
        for parent in parents:
            if not parent.is_relative_to(root):
                root = Path(os.path.commonpath((root, parent)))
        digest = hashlib.sha256()
        digest.update(json.dumps(self.classes).encode() + b"\n")
        for name, items in (("train", self.train), ("test", self.test)):
            digest.update(name.encode() + b"\n")
            for item in items:
                relative = Path(item.image).absolute().relative_to(root).as_posix()
                digest.update(json.dumps((relative, int(item.label))).encode() + b"\n")
        return {
            "classes": list(self.classes),
            "n_train": len(self.train),
            "n_test": len(self.test),
            "split_sha256": digest.hexdigest(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping) -> ClassificationSplit:
        def read_items(items: Sequence[Mapping]) -> tuple[LabeledImage, ...]:
            return tuple(
                LabeledImage(image=str(item["image"]), label=int(item["label"])) for item in items
            )

        return cls(
            classes=tuple(str(name) for name in value["classes"]),
            train=read_items(value["train"]),
            test=read_items(value["test"]),
        )


@PROBES.register("linear")
class LinearProbe(Probe):
    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        manifest: str | None = None,
        encoder: str = "image",
        head: str | None = None,
        view: str = "center_crop",
        size: int = 224,
        batch_size: int = 256,
        l2: float = 1e-3,
        max_iter: int = 500,
        split: ClassificationSplit | None = None,
        feature_norm: str = "standardize",
        c: float | None = None,
    ):
        self.name = name
        if (manifest is None) == (split is None):
            raise ValueError("linear probe needs exactly one of manifest or split")
        if feature_norm not in FEATURE_NORMS:
            raise ValueError(f"feature_norm must be one of {FEATURE_NORMS}, got {feature_norm!r}")
        if c is not None and not c > 0:
            raise ValueError(f"c must be positive, got {c}")
        self.manifest = str(manifest) if manifest is not None else None
        self.manifest_sha256 = None
        if split is None:
            assert manifest is not None
            raw = Path(manifest).read_bytes()
            self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
            split = ClassificationSplit.from_mapping(json.loads(raw))
        self.split = split
        self.classes = list(split.classes)
        self.train_set = split.train
        self.test_set = split.test
        self.encoder = encoder
        self.head = head
        self.view = view
        self.size = int(size)
        self.transform = VIEWS.get(view)(size=self.size)
        self.batch_size = int(batch_size)
        self.feature_norm = feature_norm
        self.c = float(c) if c is not None else None
        # scikit-learn's C * sum(CE) + ||W||^2 / 2, divided by C * n_train
        self.l2 = 1.0 / (2.0 * self.c * len(self.train_set)) if self.c is not None else float(l2)
        self.max_iter = int(max_iter)

    def configuration(self) -> dict:
        return {
            "name": self.name,
            "manifest": Path(self.manifest).name if self.manifest is not None else None,
            "encoder": self.encoder,
            "head": self.head,
            "view": self.view,
            "size": self.size,
            "batch_size": self.batch_size,
            "feature_norm": self.feature_norm,
            "c": self.c,
            "l2": self.l2,
            "max_iter": self.max_iter,
            "split": self.split.metadata(),
        }

    def metadata(self) -> dict:
        metadata = self.split.metadata()
        if self.manifest_sha256 is not None:
            metadata["manifest_sha256"] = self.manifest_sha256
        return metadata

    def _extract(
        self, model, ctx, items: Sequence[LabeledImage]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats: list[torch.Tensor] = []
        labels: list[int] = []
        with torch.no_grad():
            for i in range(0, len(items), self.batch_size):
                chunk = items[i : i + self.batch_size]
                imgs = []
                for it in chunk:
                    with Image.open(it.image) as f:
                        imgs.append(self.transform(f.convert("RGB")))
                x = torch.stack(imgs, dim=0).to(ctx.device)
                out = model.encode(self.encoder, x).pooled
                if self.head is not None:
                    out = model.apply_head(self.head, out)
                feats.append(out.float())
                labels.extend(it.label for it in chunk)
        return torch.cat(feats, dim=0), torch.tensor(labels, device=ctx.device)

    def run(self, model, ctx) -> dict[str, float]:
        x_tr, y_tr = self._extract(model, ctx, self.train_set)
        x_te, y_te = self._extract(model, ctx, self.test_set)
        if self.feature_norm == "l2":
            x_tr = F.normalize(x_tr, dim=1)
            x_te = F.normalize(x_te, dim=1)
        else:
            mu = x_tr.mean(dim=0)
            sd = x_tr.std(dim=0).clamp_min(1e-6)
            x_tr = (x_tr - mu) / sd
            x_te = (x_te - mu) / sd

        n_cls, dim = len(self.classes), x_tr.size(1)
        w = torch.zeros(n_cls, dim, device=x_tr.device, requires_grad=True)
        b = torch.zeros(n_cls, device=x_tr.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")

        # ProbeHook calls probes under no_grad; the classifier fit needs its
        # own (tiny, model-free) graph -- the features stay detached
        with torch.enable_grad():

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(x_tr @ w.t() + b, y_tr) + self.l2 * w.square().sum()
                loss.backward()
                return loss

            opt.step(closure)

        with torch.no_grad():
            top1 = float(((x_te @ w.t() + b).argmax(dim=1) == y_te).float().mean())
            train_top1 = float(((x_tr @ w.t() + b).argmax(dim=1) == y_tr).float().mean())
        return {"top1": top1, "train_top1": train_top1}
