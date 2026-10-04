"""Turn config dicts into live Python objects.

Hydra owns YAML composition (defaults lists, CLI overrides, multirun) and
writes the resolved config of every run to `.hydra/config.yaml`.  This module
handles what happens after composition: a config node with a "_target_" key
is turned into an object by calling the named component with the node's
remaining keys as arguments (children are built first).  A target is either
a registry address like "admission:babycl_online_segmenter" (a short name
looked up in a component registry, so it works no matter where the code
lives, and the kind prefix is checked) or a dotted import path inside the
egolearn package ("egolearn.memory.table.Table").

Norm (design review): no logic in configs.  YAML only; a config describes an
object graph and scalar hyperparameters, nothing else.
"""

from __future__ import annotations

import importlib
from typing import Any

from egolearn.hub import resolve

from .registry import lookup

TARGET_KEY = "_target_"


def _resolve(target: str):
    if ":" in target:
        return lookup(target)
    if not target.startswith("egolearn."):
        raise KeyError(
            f"dotted _target_ must live inside the egolearn package, got {target!r} "
            "(external code is wired through registries)"
        )
    module, _, attr = target.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def instantiate(node: Any, **overrides):
    """Recursively instantiate a config node."""
    if isinstance(node, list):
        return [instantiate(x) for x in node]
    if isinstance(node, dict):
        if TARGET_KEY in node:
            kwargs = {}
            for k, v in node.items():
                if k == TARGET_KEY or k.startswith("_"):
                    continue
                kwargs[k] = instantiate(v)
            kwargs.update(overrides)
            return _resolve(node[TARGET_KEY])(**kwargs)
        return {k: instantiate(v) for k, v in node.items()}
    if isinstance(node, str) and node.startswith("hf://"):
        return str(resolve(node))
    return node


def load_config(path: str) -> dict:
    """Load one YAML file as a plain dict, without hydra composition.

    For tools that read a self-contained config (benchmarks, notebooks).
    The train entry point (egolearn.train) composes through hydra instead;
    hydra-only keys are dropped here so both kinds of file load.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"config root must be a mapping, got {type(cfg).__name__}")
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    return cfg
