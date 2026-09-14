"""Construct models independently of streams, optimizers, and training drivers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from copy import deepcopy

from egolearn.config.instantiate import instantiate
from egolearn.config.registry import ENCODERS
from egolearn.model.model import ComposedModel


def build_model(
    config: Mapping,
    *,
    vocab_size: int | None = None,
    imports: Sequence[str] = (),
    initialize_pretrained: bool = True,
) -> ComposedModel:
    """Build a model and retain its resolved construction config for checkpoints.

    ``config`` is the model section of a training config. ``vocab_size`` resolves
    text encoders' ``auto`` setting. Checkpoint loading disables pretrained
    initialization because the checkpoint supplies all weights.
    """
    for module in imports:
        importlib.import_module(module)
    resolved = deepcopy(dict(config))
    encoders = {}
    frozen = []
    for name, original in (resolved.get("encoders") or {}).items():
        spec = dict(original)
        if spec.pop("frozen", False):
            frozen.append(name)
        if spec.get("vocab_size") == "auto":
            kind, _, reg_name = str(spec.get("_target_", "")).partition(":")
            cls = ENCODERS.get(reg_name) if kind == "encoder" else None
            if cls is None or getattr(cls, "modality", None) != "text":
                raise ValueError(
                    f"encoder {name!r}: vocab_size=auto fills in the text "
                    "tokenizer's vocab size and only applies to text encoders"
                )
            if vocab_size is None:
                raise ValueError(f"encoder {name!r}: vocab_size=auto needs a tokenizer")
            spec["vocab_size"] = vocab_size
            original["vocab_size"] = vocab_size
        if not initialize_pretrained and "pretrained" in spec:
            spec["pretrained"] = False
        encoders[name] = instantiate(spec)
    if not encoders:
        raise ValueError("model needs at least one entry under model.encoders")

    heads = {}
    for name, original in (resolved.get("heads") or {}).items():
        spec = dict(original)
        encoder = spec.pop("encoder")
        field = spec.pop("field", "pooled")
        heads[name] = {"encoder": encoder, "field": field, "module": instantiate(spec)}

    predictors = {}
    for name, original in (resolved.get("predictors") or {}).items():
        spec = dict(original)
        inputs = list(spec.pop("inputs", []))
        predictors[name] = {"inputs": inputs, "module": instantiate(spec)}

    model = ComposedModel(encoders, heads=heads, predictors=predictors, frozen=frozen)
    for name, head in heads.items():
        if head["encoder"] not in encoders:
            raise ValueError(
                f"head {name!r} reads encoder {head['encoder']!r}; model has {sorted(encoders)}"
            )
    for name, predictor in predictors.items():
        for encoder in predictor["inputs"]:
            if encoder not in encoders:
                raise ValueError(
                    f"predictor {name!r} reads encoder {encoder!r}; model has {sorted(encoders)}"
                )
    model.construction_config = {
        "format_version": 1,
        "model": resolved,
        "_imports_": list(imports),
    }
    return model
