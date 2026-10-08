"""Run benchmarks on a loaded model without constructing a training pipeline."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from egolearn import __version__
from egolearn.config.instantiate import instantiate
from egolearn.core.context import _stable_seed
from egolearn.evaluation.benchmark import Benchmark
from egolearn.model.model import ComposedModel
            try:
                # Each benchmark starts from the same seed, independent of suite order.
                with (
                    _isolated_evaluation_rng(ctx.seed),
                    torch.inference_mode(False),
                    torch.no_grad(),
                ):
                    benchmark.prepare()
                    metrics = {
                        key: float(value) for key, value in benchmark.run(model, ctx).items()
                    }
                    for key, value in metrics.items():
                        if not math.isfinite(value):
                            raise ValueError(f"metric {key!r} must be finite, got {value}")
                    metadata = deepcopy(benchmark.metadata())
                result.results[benchmark.name] = metrics
                result.configs[benchmark.name] = config
                result.versions[benchmark.name] = str(benchmark.version)
                result.metadata[benchmark.name] = metadata
                if record is not None:
                    record.evaluation(result)
            except Exception as exc:
                result.status = "partial"
                result.failed_benchmark = benchmark.name
                if record is not None:
                    record.evaluation(result)
                raise EvaluationError(benchmark.name, result) from exc
