"""Load model weights for evaluation without resuming training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.evaluator import (
<<<<<<< HEAD
    EvaluationResult,
    _evaluate,
    _isolated_evaluation_rng,
    benchmark_configs,
    descriptive_benchmarks,
    prepared_benchmarks,
=======
    EvaluationError,
    EvaluationResult,
    _isolated_evaluation_rng,
    evaluate,
>>>>>>> origin/fix/evaluation-provenance
)
from egolearn.hub import resolve, resolve_with_metadata
from egolearn.model.build import build_model
from egolearn.model.model import ComposedModel
<<<<<<< HEAD
from egolearn.run_record import record_run
=======
from egolearn.utils.checkpoint import read_checkpoint
>>>>>>> origin/fix/evaluation-provenance


def _construction_config(source: str | Path | Mapping) -> tuple[dict, list[str], int | None]:
    cfg = (
        OmegaConf.load(resolve(source, repo_type="model"))
        if isinstance(source, (str, Path))
        else OmegaConf.create(dict(source))
    )
    if not isinstance(cfg, DictConfig):
        raise ValueError("model_config must contain a model mapping")
    if cfg.get("format_version", 1) != 1:
        raise ValueError("unsupported model construction config version")
    # Resolve only model-related nodes: unrelated training dataset paths may no
    # longer exist, and their environment variables need not be set for evaluation.
    model_config = OmegaConf.to_container(cfg.get("model", cfg), resolve=True)
    if not isinstance(model_config, dict):
        raise ValueError("model_config must contain a model mapping")
    imports = list(cfg.get("_imports_", []))
    vocab_size = None
    if any(
        spec.get("vocab_size") == "auto" for spec in (model_config.get("encoders") or {}).values()
    ):
        from egolearn.data.text import VocabTokenizer

        if cfg.get("tokenizer") is None:
            raise ValueError("vocab_size=auto needs a tokenizer or a resolved vocabulary size")
        raw_tokenizer = OmegaConf.to_container(cfg["tokenizer"], resolve=True)
        assert isinstance(raw_tokenizer, dict)
        tokenizer_config = {str(key): value for key, value in raw_tokenizer.items()}
        vocab_size = VocabTokenizer.from_json(
            tokenizer_config.pop("vocab_json"), **tokenizer_config
        ).vocab_size
    return model_config, imports, vocab_size


def _load_model(
    checkpoint: str | Path,
    *,
    model_config: str | Path | Mapping | None,
    device: str | torch.device,
    imports: Sequence[str],
    trusted_snapshot: bool,
) -> tuple[ComposedModel, dict[str, Any]]:
    local_path, provenance = resolve_with_metadata(checkpoint, repo_type="model")
    state, metadata = read_checkpoint(local_path, trusted_snapshot=trusted_snapshot)
    metadata = {**provenance, **metadata}
    source = model_config if model_config is not None else state.get("model_config")
    if source is None:
        raise ValueError(
            "checkpoint has no model construction config; supply model_config "
            "(the saved .hydra/config.yaml or an explicit model mapping)"
        )
    config, model_imports, vocab_size = _construction_config(source)
    model = build_model(
        config,
        vocab_size=vocab_size,
        imports=list(dict.fromkeys([*model_imports, *imports])),
        initialize_pretrained=False,
    )
    if (
        state.get("export_contract") is not None
        and model.export_contract() != state["export_contract"]
    ):
        raise ValueError("model_config does not match the checkpoint's export contract")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
<<<<<<< HEAD
    metadata = {
        **provenance,
        "path": str(path),
        "sha256": digest,
        "step": state.get("step", (state.get("context") or {}).get("clock", {}).get("step")),
        "training_provenance": state.get("run_provenance") or "unavailable",
    }
=======
>>>>>>> origin/fix/evaluation-provenance
    # The caller retains only the model and provenance; optimizer/replay state
    # from legacy checkpoints is released before benchmark preparation begins.
    return model, metadata


def load_model(
    checkpoint: str | Path,
    *,
    model_config: str | Path | Mapping | None = None,
    device: str | torch.device = "cpu",
    imports: Sequence[str] = (),
    trusted_snapshot: bool = False,
) -> ComposedModel:
    """Reconstruct an EgoLearn checkpoint's model in eval mode.

    Existing files need an explicit model config. New pipeline-built checkpoints
    carry their resolved construction config. Training state is never restored.
    Snapshots require trusted_snapshot=True; see SECURITY.md.
    """
    with _isolated_evaluation_rng(0):
        model, _ = _load_model(
            checkpoint,
            model_config=model_config,
            device=device,
            imports=imports,
            trusted_snapshot=trusted_snapshot,
        )
    return model


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    benchmarks: Sequence[Benchmark | Mapping],
    model_config: str | Path | Mapping | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    imports: Sequence[str] = (),
<<<<<<< HEAD
    record_json: str | Path | None = "auto",
=======
    trusted_snapshot: bool = False,
>>>>>>> origin/fix/evaluation-provenance
) -> EvaluationResult:
    """Load a trained model and evaluate it independently of its training data."""
    config: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "model_config": model_config,
        "benchmarks": benchmark_configs(benchmarks),
        "device": str(device),
        "seed": int(seed),
        "_imports_": list(imports),
        "record_json": record_json,
    }
    with (
        record_run(
            config, "eval", reuse=True, descriptive_sections=descriptive_benchmarks(benchmarks)
        ) as record,
        _isolated_evaluation_rng(seed),
    ):
        plain = record.config if record is not None else config
        model, metadata = _load_model(
            plain["checkpoint"],
            model_config=plain.get("model_config"),
            device=device,
            imports=imports,
            trusted_snapshot=trusted_snapshot,
        )
<<<<<<< HEAD
        if record is not None:
            # Hub resolution was frozen to a local path before loading. Keep its origin.
            origin = record.value["input_provenance"].get("checkpoint", {})
            metadata = {**origin, **metadata}
            record.value["checkpoint"] = metadata
        result = _evaluate(
            model,
            benchmarks=prepared_benchmarks(benchmarks, record),
            device=device,
            seed=seed,
            imports=imports,
            record=record,
        )
        result.checkpoint = metadata
        if record is not None:
            record.evaluation(result)
        return result
=======
        try:
            result = evaluate(
                model, benchmarks=benchmarks, device=device, seed=seed, imports=imports
            )
        except EvaluationError as exc:
            exc.partial_result.checkpoint = metadata
            raise
    result.checkpoint = metadata
    return result
>>>>>>> origin/fix/evaluation-provenance
