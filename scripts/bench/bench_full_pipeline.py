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
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
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
    ap.add_argument("--gpu-log", help="sample nvidia-smi once per second with the training step")
    ap.add_argument(
        "--override", action="append", default=[], help="one Hydra override; repeatable"
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if config_path.suffix != ".yaml":
        ap.error("--config must name a primary Hydra .yaml file")
    with initialize_config_dir(config_dir=str(config_path.parent), version_base="1.3"):
        cfg = OmegaConf.to_container(
            compose(config_name=config_path.name, overrides=args.override), resolve=True
        )
    assert isinstance(cfg, dict)
    cfg.setdefault("driver", {})["max_steps"] = args.steps

    pipe = egolearn.build(cfg)
    print(pipe.summary(), flush=True)

    cuda = torch.device(pipe.ctx.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(pipe.ctx.device)
        torch.cuda.reset_peak_memory_stats(pipe.ctx.device)
    t0 = time.perf_counter()
    monitor = None
    sampler = None
    if args.gpu_log:
        if not cuda:
            raise ValueError("--gpu-log requires CUDA training")
        Path(args.gpu_log).parent.mkdir(parents=True, exist_ok=True)
        uuid = str(torch.cuda.get_device_properties(pipe.ctx.device).uuid)
        uuid = "GPU-" + uuid.removeprefix("GPU-")
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used",
                "--format=csv,noheader,nounits",
                "--loop=1",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

        def sample():
            assert monitor is not None and monitor.stdout is not None
            with open(args.gpu_log, "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "elapsed_s",
                        "step",
                        "timestamp",
                        "gpu_pct",
                        "memory_pct",
                        "watts",
                        "memory_mib",
                    ]
                )
                for row in csv.reader(monitor.stdout):
                    writer.writerow([time.perf_counter() - t0, pipe.ctx.clock.step, *row])
                    f.flush()

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
    try:
        pipe.driver.fit()
        if cuda:
            torch.cuda.synchronize(pipe.ctx.device)
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if sampler is not None:
            sampler.join(timeout=10)
        if monitor is not None and monitor.returncode not in (0, -15):
            raise RuntimeError(f"nvidia-smi sampling failed with exit code {monitor.returncode}")
    wall = time.perf_counter() - t0

    steps = pipe.ctx.clock.step
    frames = pipe.loader.frame_requests
    print("\n=== throughput ===")
    print(f"wall             : {wall:.1f} s (incl. warmup)")
    print(f"gradient steps   : {steps}  ({steps / max(wall, 1e-9):.2f} steps/s)")
    print(f"frame requests   : {frames}  ({frames / max(wall, 1e-9):.1f} frames/s)")
    print(pipe.driver.timer.summary())

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "wall_s": wall,
                    "steps": steps,
                    "steps_per_s": steps / max(wall, 1e-9),
                    "frame_requests": frames,
                    "frame_requests_per_s": frames / max(wall, 1e-9),
                    "observations": pipe.ctx.clock.admitted,
                    "events": pipe.ctx.clock.events_emitted,
                    "nonfinite_steps": pipe.driver.nonfinite_steps,
                    "device": str(pipe.ctx.device),
                    "gpu": torch.cuda.get_device_name(pipe.ctx.device) if cuda else None,
                    "peak_gpu_bytes": (
                        torch.cuda.max_memory_allocated(pipe.ctx.device) if cuda else None
                    ),
                    "config": cfg,
                    "stages": pipe.driver.timer.report(),
                    "stage_timing": "CPU wall time; stages overlap and GPU work is asynchronous",
                },
                f,
                indent=2,
            )
        print(f"[report] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
