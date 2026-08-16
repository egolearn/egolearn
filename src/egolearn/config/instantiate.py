"""Recursive component instantiation (the registry side of the config system).

Hydra owns YAML composition (defaults lists, CLI overrides, multirun) and
writes the resolved config of every run to `.hydra/config.yaml`.  This module
owns what happens after composition: a config node with a "_target_" key is
instantiated by calling the resolved component with the remaining keys
(children instantiated first).  Targets are either registry addresses
("admission:babycl_online_segmenter") -- short names decoupled from code
location, kind-checked -- or dotted paths inside the egolearn package
("egolearn.memory.table.Table").

Norm (design review): no logic in configs.  YAML only; a config describes an
object graph and scalar hyperparameters, nothing else.
"""

from __future__ import annotations

import importlib
from typing import Any

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
