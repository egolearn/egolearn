"""Evaluate an image encoder by adapting a private copy on ImageNet."""

from __future__ import annotations

from egolearn.config.registry import BENCHMARKS
from egolearn.core import keys as K
from egolearn.evaluation import full_finetune
from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.benchmarks.imagenet_linear.benchmark import build_split
from egolearn.evaluation.evaluator import EvaluationContext
from egolearn.evaluation.linear_probe import ClassificationSplit
from egolearn.model.model import ComposedModel


@BENCHMARKS.register("imagenet_full_finetune")
class ImageNetFullFinetuneBenchmark(Benchmark):
    """Fit an encoder and classification head, then score final-epoch accuracy.

    The input model remains unchanged. ``output_dir`` optionally saves epoch
    checkpoints for resume; use a distinct directory for each model and seed.
    ``seed`` controls initialization, shuffling and augmentation. Worker count,
    physical batch size and accumulation are recorded parts of the protocol.
    """

    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        data_root: str,
        expected_classes: int | None = 1000,
        encoder: str = "image",
        pooling: str = "default",
        drop_path: float = 0.0,
        size: int = 224,
        batch_size: int = 64,
        accumulation: int = 4,
        workers: int = 16,
        epochs: int = 100,
        warmup_epochs: int = 5,
        lr: float = 0.001,
        weight_decay: float = 0.05,
        layer_decay: float = 1.0,
        crop_min: float = 0.08,
        randaugment: bool = True,
        erase_probability: float = 0.25,
        mixup: float = 0.8,
        cutmix: float = 1.0,
        smoothing: float = 0.1,
        clip_grad: float = 1.0,
        eval_view: str = "imagenet_center_crop",
        eval_every: int = 1,
        seed: int = 0,
        amp: bool = False,
        defer_metrics: bool = False,
        channels_last: bool = False,
        cudnn_benchmark: bool = False,
        compile_mode: str | None = None,
        output_dir: str | None = None,
    ):
        self.name = name
        self.data_root = data_root
        self.expected_classes = expected_classes
        self.output_dir = output_dir
        self.settings = {
            "encoder": encoder,
            "pooling": pooling,
            "drop_path": drop_path,
            "size": size,
            "batch_size": batch_size,
            "accumulation": accumulation,
            "workers": workers,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "layer_decay": layer_decay,
            "crop_min": crop_min,
            "randaugment": randaugment,
            "erase_probability": erase_probability,
            "mixup": mixup,
            "cutmix": cutmix,
            "smoothing": smoothing,
            "clip_grad": clip_grad,
            "eval_view": eval_view,
            "eval_every": eval_every,
            "seed": seed,
            "amp": amp,
            "defer_metrics": defer_metrics,
            "channels_last": channels_last,
            "cudnn_benchmark": cudnn_benchmark,
            "compile_mode": compile_mode,
        }
        full_finetune.validate_config(self.settings)
        self.split: ClassificationSplit | None = None

    def prepare(self) -> None:
        if self.split is None:
            split = build_split(self.data_root, expected_classes=self.expected_classes)
            train = {item.image for item in split.train}
            if any(item.image in train for item in split.test):
                raise ValueError("training and held-out images overlap")
            self.split = split

    def configuration(self) -> dict:
        return {
            "name": self.name,
            "data_root": self.data_root,
            "expected_classes": self.expected_classes,
            "output_dir": self.output_dir,
            **self.settings,
        }

    def metadata(self) -> dict:
        return self.split.metadata() if self.split is not None else {}

    def run(self, model: ComposedModel, ctx: EvaluationContext) -> dict[str, float]:
        self.prepare()
        assert self.split is not None
        return full_finetune.fit(
            model,
            self.split,
            {**self.settings, "device": str(ctx.device)},
            output_dir=self.output_dir,
        )
