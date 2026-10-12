#!/usr/bin/env python
"""Full-pipeline benchmark: run N real training steps of a config and report the
time spent in each stage plus overall throughput.  This shows what the framework's
abstractions actually cost.

    python scripts/bench/bench_full_pipeline.py \\
        --config projects/babycl/configs/saycam_default.yaml --steps 200

To measure only the data side (no model, much faster to iterate on), use
scripts/bench/bench_dataloading.py.

Working-group reference point: the BabyCL default run holds ~70% GPU utilization
on 2x A100 over ~48 h.  A framework port of the same config should be compared
against that number BEFORE merging (merges are validated by real experiments).
The per-stage breakdown tells you whether a gap comes from the data pipeline
(data.load / batch.compose dominate) or from the model side.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import egolearn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="primary Hydra .yaml config path (declare hydra.searchpath for packaged defaults)",
    )
    ap.add_argument("--steps", type=int, default=200, help="gradient steps to run (after warmup)")
    ap.add_argument("--output-json", default=None, help="write the raw per-stage report here")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if config_path.suffix != ".yaml":
        ap.error("--config must name a primary Hydra .yaml file")
    with initialize_config_dir(config_dir=str(config_path.parent), version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name=config_path.name), resolve=True)
    assert isinstance(cfg, dict)
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
