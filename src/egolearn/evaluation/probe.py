"""Probe protocol: anytime evaluation against the export contract.

A probe only touches the model through taps/heads (never internals), so any
checkpoint satisfying the export contract is evaluable.

Benchmark-specific probes are project- or benchmarks/-owned and register into
the PROBES registry (e.g. probe:four_afc in
benchmarks/labeled_s_4afc/__init__.py).  Core ships only the protocol.
"""

from __future__ import annotations


class Probe:
    name = "probe"

    def run(self, model, ctx) -> dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError
