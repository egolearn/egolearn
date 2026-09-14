"""Run benchmarks on a loaded model without constructing a training pipeline."""

from __future__ import annotations

import importlib
import inspect
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from egolearn import __version__
from egolearn.config.instantiate import instantiate
from egolearn.core.context import _stable_seed
from egolearn.evaluation.benchmark import Benchmark
from egolearn.model.model import ComposedModel


@dataclass(frozen=True)
class EvaluationContext:
    """Evaluation device and seeds, independent of training clocks and state."""

    seed: int
    device: torch.device

    def derive_seed(self, name: str) -> int:
        return _stable_seed(self.seed, name)


@dataclass
class EvaluationResult:
    """JSON-serializable metrics and provenance from one model evaluation."""

    results: dict[str, dict[str, float]]
    configs: dict[str, dict[str, Any]]
    versions: dict[str, str]
    metadata: dict[str, dict[str, Any]]
    seed: int
    device: str
    checkpoint: dict[str, Any] | None = None
    model_config: dict[str, Any] | None = None
    package_version: str = __version__
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


@contextmanager
def _isolated_evaluation_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    # Preserve the legacy global RNG used by third-party benchmark code.
    numpy_state = np.random.get_state()  # noqa: NPY002
    # fork_rng preserves CPU and every CUDA device, including classifier init.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))  # noqa: NPY002
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)  # noqa: NPY002


@contextmanager
def _frozen_model(model: ComposedModel) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval()
        model.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)
        # Preserve mixed modes (e.g. frozen EMA targets), even on failure.
        for module, training in modes:
            module.training = training


def _configured_benchmark(spec: Benchmark | Mapping) -> tuple[Benchmark, dict]:
    if isinstance(spec, Benchmark):
        benchmark = spec
        config = benchmark.configuration()
    else:
        benchmark = instantiate(dict(spec))
        if not isinstance(benchmark, Benchmark):
            raise TypeError("evaluation entries must implement Benchmark or Probe")
        # Include constructor defaults, so changes to defaults remain visible
        # when comparing results produced by different versions of EgoLearn.
        signature = inspect.signature(type(benchmark))
        arguments = signature.bind(**{k: v for k, v in spec.items() if not k.startswith("_")})
        arguments.apply_defaults()
        config = {**arguments.arguments, **benchmark.configuration()}
    return benchmark, {
        "_target_": f"{type(benchmark).__module__}.{type(benchmark).__qualname__}",
        **deepcopy(config),
    }


def evaluate(
    model: ComposedModel,
    *,
    benchmarks: Sequence[Benchmark | Mapping],
    device: str | torch.device | None = None,
    seed: int = 0,
    imports: Sequence[str] = (),
) -> EvaluationResult:
    """Evaluate a loaded model, preserving its modes, gradient flags, and RNGs.

    The model must already be on ``device`` (inferred by default); this avoids
    moving a live training model behind its optimizer's back. Both benchmark
    configs and existing probe configs are accepted. Dataset metadata may be
    reused by passing benchmark instances to subsequent calls.

    Benchmarks must not mutate model weights or buffers. Inference runs under
    no_grad, but a benchmark may enable gradients for a fresh linear decoder.
    """
    if not benchmarks:
        raise ValueError("evaluation needs at least one benchmark")
    tensors = [*model.parameters(), *model.buffers()]
    model_device = tensors[0].device if tensors else torch.device("cpu")
    selected_device = torch.device(device) if device is not None else model_device
    if selected_device.type == "cuda" and selected_device.index is None:
        selected_device = torch.device("cuda", torch.cuda.current_device())
    if any(tensor.device != selected_device for tensor in tensors):
        raise ValueError(f"move the model to {selected_device} before calling evaluate()")
    ctx = EvaluationContext(seed=int(seed), device=selected_device)
    result = EvaluationResult(
        results={},
        configs={},
        versions={},
        metadata={},
        seed=ctx.seed,
        device=str(ctx.device),
        model_config=deepcopy(model.construction_config),
    )
    with _isolated_evaluation_rng(ctx.seed), _frozen_model(model):
        for module in imports:
            importlib.import_module(module)
        configured = [_configured_benchmark(spec) for spec in benchmarks]
        names = [benchmark.name for benchmark, _ in configured]
        if len(names) != len(set(names)):
            raise ValueError("evaluation benchmark names must be unique")
        for benchmark, _ in configured:
            missing = set(benchmark.requires_capabilities) - set(model.capabilities())
            if missing:
                raise ValueError(
                    f"benchmark {benchmark.name!r} needs capabilities {sorted(missing)}"
                )
        for benchmark, config in configured:
            # Each benchmark starts from the same seed, independent of suite order.
            with _isolated_evaluation_rng(ctx.seed), torch.inference_mode(False), torch.no_grad():
                benchmark.prepare()
                metrics = benchmark.run(model, ctx)
            result.results[benchmark.name] = {key: float(value) for key, value in metrics.items()}
            result.configs[benchmark.name] = config
            result.versions[benchmark.name] = str(benchmark.version)
            result.metadata[benchmark.name] = benchmark.metadata()
    return result
