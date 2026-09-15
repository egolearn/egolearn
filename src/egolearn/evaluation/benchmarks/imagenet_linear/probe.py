"""ImageNet probe backed by the shared stochastic-view SGD linear-decoding protocol."""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder

from egolearn.config.registry import PROBES
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage
from egolearn.evaluation.linear_sgd_probe import LinearSGDProbe


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


@PROBES.register("imagenet_linear_sgd")
class ImageNetLinearSGDProbe(LinearSGDProbe):
    """Full ImageNet train/val adapter, defaulting to 224-pixel inputs."""

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
        super().__init__(
            name=name,
            split=build_split(data_root, expected_classes=expected_classes),
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
