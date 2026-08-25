"""StageTimer: records how long each stage of the training pipeline takes.

Working-group requirement (2026-08): "measure how much latency is caused by
the abstractions" and keep GPU utilization visible (reference point: BabyCL
default run, 2x A100, ~48 h, ~70% util).  Every run therefore carries a
per-stage timing report based on wall-clock time, with near-zero overhead
(one perf_counter pair per measurement).

Stages recorded by the LockstepDriver:
    stream.advance   pull observations + run admission components
    data.load        decode + augment (recorded inside the Loader)
    batch.compose    sampling + collation, includes data.load
    model.forward    loss computation (forward passes)
    model.backward   backward pass
    optim.step       optimizer update

`summary()` gives a one-line report; `report()` returns the raw numbers.
GPU work runs asynchronously, so wall-clock time is only a lower bound per
stage -- for exact kernel timing add torch.profiler on top.  This report is
for spotting when the data pipeline, not the GPU, is the bottleneck.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self._stages: dict[str, dict] = {}

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            s = self._stages.setdefault(stage, {"total": 0.0, "count": 0, "max": 0.0})
            s["total"] += dt
            s["count"] += 1
            if dt > s["max"]:
                s["max"] = dt

    def report(self) -> dict[str, dict]:
        """{stage: {total_s, count, mean_ms, max_ms, share}} -- share is the
        fraction of the summed stage time (stages may nest, so shares are a
        guide, not a partition)."""
        grand = sum(s["total"] for s in self._stages.values()) or 1.0
        out = {}
        for name, s in self._stages.items():
            out[name] = {
                "total_s": s["total"],
                "count": s["count"],
                "mean_ms": 1000.0 * s["total"] / max(1, s["count"]),
                "max_ms": 1000.0 * s["max"],
                "share": s["total"] / grand,
            }
        return out

    def summary(self) -> str:
        rep = self.report()
        if not rep:
            return "[timing] no measurements"
        parts = [
            f"{name} {r['mean_ms']:.1f}ms x{r['count']} ({100 * r['share']:.0f}%)"
            for name, r in sorted(rep.items(), key=lambda kv: -kv[1]["total_s"])
        ]
        return "[timing] " + "  ".join(parts)

    def reset(self) -> None:
        self._stages.clear()
