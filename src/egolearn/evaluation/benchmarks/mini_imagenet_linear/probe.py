"""mini-ImageNet (100 classes) fixed-feature linear probe, BabyCL paper protocol."""

from __future__ import annotations

from pathlib import Path

from egolearn.config.registry import PROBES
from egolearn.evaluation.benchmarks.imagenet_linear.benchmark import build_split
from egolearn.evaluation.linear_probe import LinearProbe

N_CLASSES = 100


@PROBES.register("mini_imagenet_linear")
class MiniImageNetLinearProbe(LinearProbe):
    """ImageFolder ``train/`` and ``val/`` split; unit-length features, C=0.316."""

    def __init__(
        self,
        name: str,
        data_root: str,
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
        super().__init__(
            name=name,
            split=build_split(data_root, expected_classes=N_CLASSES),
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
        return {**config, "data_root": Path(self.data_root).name}
