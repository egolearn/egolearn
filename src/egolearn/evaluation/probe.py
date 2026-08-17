"""Probe protocol: evaluations that can run at any point during training.

A probe (an evaluation routine) only touches the model through its taps and
heads, never through internals.  Every checkpoint records the tap and head
names, so any checkpoint -- from any method -- can be evaluated the same
way.

Benchmark-specific probes live with their benchmark or project and register
into the PROBES registry (e.g. probe:four_afc in
benchmarks/labeled_s_4afc/__init__.py).  Core ships only this protocol.
"""

from __future__ import annotations


class Probe:
    name = "probe"

    def run(self, model, ctx) -> dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError
