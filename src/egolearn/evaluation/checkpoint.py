"""Load model weights for evaluation without resuming training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.evaluator import EvaluationResult, _isolated_evaluation_rng, evaluate
from egolearn.hub import resolve
from egolearn.model.build import build_model
from egolearn.model.model import ComposedModel
from egolearn.utils.checkpoint import read_checkpoint


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
    state, metadata = read_checkpoint(checkpoint, trusted_snapshot=trusted_snapshot)
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
    """Reconstruct a trusted EgoLearn checkpoint/snapshot's model in eval mode.

    Existing files need an explicit model config. New pipeline-built checkpoints
    carry their resolved construction config. Training state is never restored.
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
    trusted_snapshot: bool = False,
) -> EvaluationResult:
    """Load a trained model and evaluate it independently of its training data."""
    with _isolated_evaluation_rng(seed):
        model, metadata = _load_model(
            checkpoint,
            model_config=model_config,
            device=device,
            imports=imports,
            trusted_snapshot=trusted_snapshot,
        )
        result = evaluate(model, benchmarks=benchmarks, device=device, seed=seed, imports=imports)
    result.checkpoint = metadata
    return result
