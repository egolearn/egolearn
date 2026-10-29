"""Compare fine-tuning runners on equal image counts after GPU warm-up."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch


def load_runner(path: Path):
    """Load a runner supplied explicitly on the command line."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variants() -> dict[str, dict]:
    """Return independent kernel ablations and cumulative batch-size trials."""
    deferred = {"defer_metrics": True}
    kernels = {**deferred, "channels_last": True, "cudnn_benchmark": True}
    compiled = {**kernels, "compile_mode": "default"}
    return {
        "baseline": {},
        "deferred": deferred,
        "channels_last": {**deferred, "channels_last": True},
        "cudnn": {**deferred, "cudnn_benchmark": True},
        "kernels": kernels,
        "compile": {**deferred, "compile_mode": "default"},
        "kernels_compile": compiled,
        "batch128": {**kernels, "batch_size": 128, "accumulation": 2},
        "batch256": {**kernels, "batch_size": 256, "accumulation": 1},
        "batch256_compile": {**compiled, "batch_size": 256, "accumulation": 1},
        "baseline_repeat": {},
    }


def write_summary(output: Path, report: dict) -> None:
    """Write the completed comparisons as a compact table."""
    fields = [
        "model",
        "variant",
        "median_images_per_second",
        "speedup",
        "estimated_100epoch_training_hours",
        "warmup_seconds",
        "peak_gpu_gb",
        "max_state_difference_from_baseline",
    ]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


class TimedLoader:
    """Measure complete update windows, including waits for the next batch."""

    def __init__(self, loader, *, warmup: int, window: int, accumulation: int):
        self.loader = loader
        self.batch_size = loader.batch_size
        self.dataset = loader.dataset
        self.warmup = warmup * accumulation
        self.window = window * accumulation
        self.windows: list[dict] = []
        self.warmup_seconds = 0.0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        torch.cuda.synchronize()
        start = time.perf_counter()
        iterator = iter(self.loader)
        wait_seconds = 0.0
        for index in range(len(self) + 1):
            boundary = index == self.warmup or (
                index > self.warmup and (index - self.warmup) % self.window == 0
            )
            if boundary:
                torch.cuda.synchronize()
                now = time.perf_counter()
                if index == self.warmup:
                    self.warmup_seconds = now - start
                else:
                    samples = self.window * self.batch_size
                    self.windows.append(
                        {
                            "seconds": now - start,
                            "images": samples,
                            "images_per_second": samples / (now - start),
                            "loader_next_seconds": wait_seconds,
                        }
                    )
                start = now
                wait_seconds = 0.0
            if index == len(self):
                return
            wait_start = time.perf_counter()
            batch = next(iterator)
            wait_seconds += time.perf_counter() - wait_start
            yield batch


def run_variant(
    runner,
    baseline,
    current,
    items,
    val_items,
    split,
    args,
    config_path,
    name,
    reference_state,
    reference_ips,
):
    """Measure one fresh model and return its metrics and CPU state."""
    runner.seed_all(current["seed"])
    torch.backends.cudnn.benchmark = False
    model, provenance = runner.load_classifier(current, len(split.classes))
    if runner is baseline:
        model.to(current["device"])
    else:
        runner.prepare_model(model, current)
    optimizer = torch.optim.AdamW(
        runner.parameter_groups(model, current["weight_decay"], current["layer_decay"]),
        lr=current["lr"],
    )
    loader = runner.make_loader(items, runner.train_transform(current), current, training=True)
    timed = TimedLoader(
        loader,
        warmup=args.warmup_updates,
        window=args.window_updates,
        accumulation=current["accumulation"],
    )

    def progress(batch, batches, loss, runner=runner, config_path=config_path, name=name):
        runner.write_json(
            args.output / "progress.json",
            {
                "model": config_path.stem,
                "variant": name,
                "batch": batch,
                "batches": batches,
                "loss": loss,
            },
        )

    torch.cuda.reset_peak_memory_stats()
    metrics = runner.train_epoch(model, timed, optimizer, current, 0, progress)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    validation = runner.make_loader(
        val_items,
        runner.VIEWS.get(current["eval_view"])(size=current["size"]),
        current,
        training=False,
    )
    metrics.update(runner.accuracy(model, validation, current))
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    ips = statistics.median(window["images_per_second"] for window in timed.windows)
    if name == "baseline":
        reference_state, reference_ips = state, ips
    differences = None
    if reference_state is not None:
        differences = max(
            (state[key].double() - value.double()).abs().max().item()
            for key, value in reference_state.items()
            if value.numel() and value.is_floating_point()
        )
    row = {
        "model": config_path.stem,
        "variant": name,
        "config": current,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "windows": timed.windows,
        "warmup_seconds": timed.warmup_seconds,
        "median_images_per_second": ips,
        "speedup": ips / reference_ips if reference_ips else None,
        "estimated_100epoch_training_hours": len(split.train) * 100 / ips / 3600,
        "peak_gpu_gb": peak_gb,
        "max_state_difference_from_baseline": differences,
        "metrics": metrics,
    }
    return row, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--baseline-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=16)
    parser.add_argument("--window-updates", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=list(variants()), default=list(variants()))
    args = parser.parse_args()
    if min(args.warmup_updates, args.window_updates, args.repeats) < 1:
        parser.error("warm-up, window, and repeat counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    optimized, baseline = load_runner(args.runner), load_runner(args.baseline_runner)
    report = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "benchmark_sha256": optimized.digest(__file__),
        "runner_sha256": optimized.digest(args.runner),
        "baseline_sha256": baseline.digest(args.baseline_runner),
        "warmup_updates": args.warmup_updates,
        "window_updates": args.window_updates,
        "repeats": args.repeats,
        "results": [],
        "errors": [],
        "data_warmup": [],
    }
    torch.set_num_threads(1)
    split = None
    dataset_config = None
    for config_path in args.configs:
        config = json.loads(config_path.read_text())
        effective_batch = config["batch_size"] * config["accumulation"]
        if effective_batch != 256:
            raise ValueError("batch-size ablations require a baseline effective batch of 256")
        if split is None or dataset_config != config["dataset"]:
            split = baseline.load_split(config)
            dataset_config = config["dataset"]
        count = effective_batch * (args.warmup_updates + args.window_updates * args.repeats)
        indices = random.Random(config["seed"]).sample(range(len(split.train)), count)
        items = tuple(split.train[index] for index in indices)
        val_indices = random.Random(config["seed"]).sample(range(len(split.test)), 512)
        val_items = tuple(split.test[index] for index in val_indices)
        # All variants revisit this subset. The first baseline must use the same
        # filesystem cache state as later variants.
        baseline.seed_all(config["seed"])
        print(json.dumps({"model": config_path.stem, "state": "warming_data"}), flush=True)
        warm_started = time.perf_counter()
        warm_loader = baseline.make_loader(
            items, baseline.train_transform(config), config, training=True
        )
        for batch in warm_loader:
            del batch
        del warm_loader
        report["data_warmup"].append(
            {
                "model": config_path.stem,
                "images": len(items),
                "seconds": time.perf_counter() - warm_started,
            }
        )
        optimized.write_json(args.output / "results.json", report)
        reference_state = None
        reference_ips = None
        for name in args.variants:
            current = {**config, **variants()[name]}
            runner = baseline if name.startswith("baseline") else optimized
            print(
                json.dumps({"model": config_path.stem, "variant": name, "state": "starting"}),
                flush=True,
            )
            try:
                row, state = run_variant(
                    runner,
                    baseline,
                    current,
                    items,
                    val_items,
                    split,
                    args,
                    config_path,
                    name,
                    reference_state,
                    reference_ips,
                )
                if name == "baseline":
                    reference_state = state
                    reference_ips = row["median_images_per_second"]
                report["results"].append(row)
                print(
                    json.dumps({key: value for key, value in row.items() if key != "config"}),
                    flush=True,
                )
                del state
            except Exception as error:
                report["errors"].append(
                    {
                        "model": config_path.stem,
                        "variant": name,
                        "type": type(error).__name__,
                        "error": str(error),
                    }
                )
                optimized.write_json(args.output / "results.json", report)
                print(json.dumps(report["errors"][-1]), flush=True)
                if name.startswith("baseline"):
                    raise
            optimized.write_json(args.output / "results.json", report)
            write_summary(args.output, report)
            torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()
    report["status"] = "complete_with_errors" if report["errors"] else "complete"
    report["baseline_repeat_ratio"] = {
        row["model"]: row["speedup"]
        for row in report["results"]
        if row["variant"] == "baseline_repeat"
    }
    optimized.write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
