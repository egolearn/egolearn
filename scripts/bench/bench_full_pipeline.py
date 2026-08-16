#!/usr/bin/env python
"""Measure what the abstractions cost: run N real steps of a config and
report per-stage latency + throughput.

    python scripts/bench/bench_full_pipeline.py \\
        --config projects/babycl/configs/saycam_default.yaml --steps 200

Data-plane-only measurement (no model, much faster to iterate on):
scripts/bench/bench_dataloading.py.

Working-group reference point: the BabyCL default run holds ~70% GPU
utilization on 2x A100 over ~48 h.  A framework port of the same config
should be compared against that number BEFORE merging (real experiments
validate merges).  The per-stage breakdown tells you whether a gap is the
data plane (data.load / batch.compose dominate) or the model side.
"""

from __future__ import annotations

import argparse
import json
import time

import egolearn
from egolearn.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--steps", type=int, default=200, help="gradient steps to run (after warmup)")
    ap.add_argument("--output-json", default=None, help="write the raw per-stage report here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("driver", {})["max_steps"] = args.steps

    pipe = egolearn.build(cfg)
    print(pipe.summary(), flush=True)

    t0 = time.perf_counter()
    pipe.driver.fit()
    wall = time.perf_counter() - t0

    steps = pipe.ctx.clock.step
    frames = sum(r["count"] for n, r in pipe.driver.timer.report().items() if n == "data.load")
    print("\n=== throughput ===")
    print(f"wall             : {wall:.1f} s (incl. warmup)")
    print(f"gradient steps   : {steps}  ({steps / max(wall, 1e-9):.2f} steps/s)")
    print(f"frames decoded   : {frames}  ({frames / max(wall, 1e-9):.1f} frames/s)")
    print(pipe.driver.timer.summary())

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {"wall_s": wall, "steps": steps, "stages": pipe.driver.timer.report()}, f, indent=2
            )
        print(f"[report] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
