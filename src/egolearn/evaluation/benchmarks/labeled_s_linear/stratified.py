"""Labeled-S linear probe under the BabyCL paper's stratified 80/20 protocol.

The paper probes every Labeled-S image instead of the canonical Memory
Storyboard split: the dataset root is read as an ImageFolder (26 category
directories, sorted classes and files), each category's sample indices are
shuffled with ``numpy.random.RandomState(seed)`` in label order, and the first
``max(1, int(train_fraction * n))`` of each category train the classifier.
The paper probes the 512-d ``embed512`` head output, scales each feature vector
to unit length, and fits scikit-learn's ``LogisticRegression(C=0.316,
max_iter=1000)``; the defaults below state the same objective for the shared
L-BFGS scorer.

The split is rebuilt in memory from the caller's dataset root, so no
machine-specific manifest is written.  Labeled-S itself is licensed separately
and is never downloaded here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder

from egolearn.config.registry import PROBES
from egolearn.evaluation.benchmarks.labeled_s_linear.probe import N_CLASSES
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage, LinearProbe


def build_stratified_split(
    data_root: str | Path, train_fraction: float = 0.8, seed: int = 42
) -> ClassificationSplit:
    """Per-category shuffled split over an ImageFolder root, in ImageFolder order."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    dataset = ImageFolder(Path(data_root).resolve())
    if len(dataset.classes) != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} Labeled-S categories, got {len(dataset.classes)}")
    by_label: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(dataset.samples):
        by_label.setdefault(label, []).append(index)
    rng = np.random.RandomState(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in by_label.values():
        order = np.array(indices)
        rng.shuffle(order)
        n_train = max(1, int(train_fraction * len(order)))
        train_indices.extend(order[:n_train].tolist())
        test_indices.extend(order[n_train:].tolist())

    def items(indices: list[int]) -> tuple[LabeledImage, ...]:
        return tuple(
            LabeledImage(image=dataset.samples[i][0], label=dataset.samples[i][1]) for i in indices
        )

    return ClassificationSplit(
        classes=tuple(dataset.classes), train=items(train_indices), test=items(test_indices)
    )


@PROBES.register("labeled_s_stratified_linear")
class LabeledSStratifiedLinearProbe(LinearProbe):
    """All Labeled-S images, stratified 80/20, unit-length features, C=0.316."""

    def __init__(
        self,
        name: str,
        data_root: str,
        train_fraction: float = 0.8,
        split_seed: int = 42,
        encoder: str = "image",
        head: str | None = None,
        view: str = "center_crop",
        size: int = 224,
        batch_size: int = 256,
        feature_norm: str = "l2",
        c: float | None = 0.316,
        l2: float = 1e-3,
        max_iter: int = 1000,
    ):
        self.data_root = str(data_root)
        self.train_fraction = float(train_fraction)
        self.split_seed = int(split_seed)
        split = build_stratified_split(data_root, self.train_fraction, self.split_seed)
        super().__init__(
            name=name,
            split=split,
            encoder=encoder,
            head=head,
            view=view,
            size=size,
            batch_size=batch_size,
            l2=l2,
            max_iter=max_iter,
            feature_norm=feature_norm,
            c=c,
        )

    def configuration(self) -> dict:
        config = super().configuration()
        config.pop("manifest")
        config.pop("split")
        return {
            **config,
            "data_root": Path(self.data_root).name,
            "train_fraction": self.train_fraction,
            "split_seed": self.split_seed,
        }
