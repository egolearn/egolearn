"""Linear probing: fit a linear classifier on FROZEN features, report
held-out top-1 -- the standard readout for comparing self-supervised
encoders (the roadmap's "one implementation, no apex dependency").

Protocol decisions, made once here so every model is probed identically:

* features are the named encoder's POOLED output on eval-preprocessed
  images; the preprocessing (`view`/`size`) is explicit config, like
  four_afc.  An optional `head:` probes a head's output instead; the
  default probes the raw representation.
* features are standardized by the train split's mean/std.
* the classifier is multinomial logistic regression, ZERO-initialized and
  fitted with full-batch L-BFGS + L2 penalty.  A convex problem with a
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

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from egolearn.config.registry import PROBES, VIEWS
from egolearn.core import keys as K
from egolearn.evaluation.probe import Probe


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
    ):
        self.name = name
        if (manifest is None) == (split is None):
            raise ValueError("linear probe needs exactly one of manifest or split")
        if split is None:
            assert manifest is not None
            with open(manifest) as f:
                split = ClassificationSplit.from_mapping(json.load(f))
        self.split = split
        self.classes = list(split.classes)
        self.train_set = split.train
        self.test_set = split.test
        self.encoder = encoder
        self.head = head
        self.transform = VIEWS.get(view)(size=size)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)

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
