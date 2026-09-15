"""Standalone ImageNet benchmark using the shared stochastic-view SGD protocol."""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder

from egolearn.config.registry import BENCHMARKS
from egolearn.core import keys as K
from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage
from egolearn.evaluation.linear_sgd import LinearSGDEvaluator


def build_split(data_root: str | Path, expected_classes: int | None = 1000) -> ClassificationSplit:
    root = Path(data_root).resolve()
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"ImageNet root must contain train/ and val/ directories, got {root}"
        )

    train_dataset = ImageFolder(train_root)
    val_dataset = ImageFolder(val_root)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        train_only = sorted(set(train_dataset.classes) - set(val_dataset.classes))
        val_only = sorted(set(val_dataset.classes) - set(train_dataset.classes))
        raise ValueError(
            "ImageNet train/val class mappings differ: "
            f"train-only={train_only[:10]}, val-only={val_only[:10]}"
        )
    if expected_classes is not None and len(train_dataset.classes) != expected_classes:
        raise ValueError(
            f"expected {expected_classes} ImageNet classes, got {len(train_dataset.classes)}"
        )

    def convert(samples: list[tuple[str, int]]) -> tuple[LabeledImage, ...]:
        # ImageFolder joins samples to its root, which is already absolute above.
        # Resolving every sample would add one filesystem metadata lookup per
        # image -- over a million redundant lstat calls for ImageNet-1k.
        return tuple(LabeledImage(image=path, label=label) for path, label in samples)

    return ClassificationSplit(
        classes=tuple(train_dataset.classes),
        train=convert(train_dataset.samples),
        test=convert(val_dataset.samples),
    )


@BENCHMARKS.register("imagenet_linear_sgd")
class ImageNetLinearBenchmark(Benchmark):
    """Full ImageNet train/val adapter, defaulting to 224-pixel inputs."""

    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        data_root: str,
        expected_classes: int | None = 1000,
        encoder: str = "image",
        head: str | None = None,
        size: int = 224,
        batch_size: int = 1024,
        workers: int = 16,
        epochs: int = 10,
        lr: float = 3.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        larc_trust_coefficient: float | None = 0.001,
        l2_normalize: bool = True,
        seed: int = 0,
        view: str = "imagenet_center_crop",
    ):
        self.name = name
        self.data_root = data_root
        self.expected_classes = expected_classes
        self.split: ClassificationSplit | None = None
        self.scorer = LinearSGDEvaluator(
            name=name,
            encoder=encoder,
            head=head,
            view=view,
            size=size,
            batch_size=batch_size,
            workers=workers,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            larc_trust_coefficient=larc_trust_coefficient,
            l2_normalize=l2_normalize,
            seed=seed,
        )

    def prepare(self) -> None:
        if self.split is None:
            self.split = build_split(self.data_root, expected_classes=self.expected_classes)

    def configuration(self) -> dict:
        return {
            "name": self.name,
            "data_root": self.data_root,
            "expected_classes": self.expected_classes,
            **{
                key: getattr(self.scorer, key)
                for key in (
                    "encoder",
                    "head",
                    "size",
                    "batch_size",
                    "workers",
                    "epochs",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "larc_trust_coefficient",
                    "l2_normalize",
                    "seed",
                    "view",
                )
            },
        }

    def metadata(self) -> dict:
        if self.split is None:
            return {}
        return {
            "classes": list(self.split.classes),
            "n_train": len(self.split.train),
            "n_test": len(self.split.test),
        }

    def run(self, model, ctx) -> dict[str, float]:
        self.prepare()
        assert self.split is not None
        return self.scorer.evaluate(model, ctx, self.split)
