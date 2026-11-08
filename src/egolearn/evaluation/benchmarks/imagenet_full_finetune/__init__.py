"""Register the ImageNet full-network fine-tuning benchmark."""

from egolearn.evaluation.benchmarks.imagenet_full_finetune.benchmark import (
    ImageNetFullFinetuneBenchmark,
)

__all__ = ["ImageNetFullFinetuneBenchmark"]

BENCHMARK_METADATA = {
    "name": "ImageNet full-network fine-tuning",
    "source": "[ImageNet](https://www.image-net.org/)",
    "license": "ImageNet access and use terms",
    "access": "upstream-licensed",
    "prepare": (
        "Obtain images upstream and arrange `train/<class>/` and `val/<class>/` under `data_root`."
    ),
}
