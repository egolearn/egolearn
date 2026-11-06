"""Resumable full-pass MAE runs, with IID matched to streaming update counts.

Invoke with the full_* project configs. SIGUSR1 or the wall-clock budget
pauses at an optimizer boundary, writes an atomic snapshot, and exits 75.
A scheduler may requeue only that exit code; other errors must remain failures.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import time
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import egolearn
from egolearn.training.hooks import Hook, MetricsHook, SnapshotHook, WandbHook
from egolearn.training.offline import OfflineDriver
from egolearn.utils.snapshot import load_snapshot, save_snapshot

PAUSED = 75
PREFLIGHT_STOP = 76


class PauseTraining(Exception):
    """A requested stop after a completed optimizer update."""

    def __init__(self, exit_code: int):
        self.exit_code = exit_code
        super().__init__(exit_code)


def prepare(cfg: dict):
    """Build a full streaming pass and match IID to its exact update count."""
    cfg = deepcopy(cfg)
    driver = cfg["driver"]
    control = cfg["run_control"]
    batch = int(control["batch_size"])
    window = int(control.get("window_size", batch))
    reference_window = int(control.get("reference_window_size", window))
    stride = int(control.get("stride", 1))
    if not 1 <= stride <= window or not 2 <= batch <= window:
        raise ValueError("need 2 <= batch_size <= window_size and 1 <= stride <= window_size")
    if reference_window < window:
        raise ValueError("reference_window_size must be at least window_size")
    if batch < 2 or int(driver.get("grad_accumulation", 1)) != 1:
        raise ValueError("full-pass MAE requires batch_size >= 2 and grad_accumulation=1")
    offline = driver.get("mode", "streaming") == "offline"
    if driver.get("prefetch") and not offline:
        raise ValueError("full-pass streaming requires prefetch=false (one update per advance)")
    if driver.get("max_steps") is not None or driver.get("epochs") is not None:
        raise ValueError("full-pass budget is derived from the dataset; omit max_steps and epochs")
    if cfg["stream"]["_target_"] not in ("stream:saycam", "stream:babyview", "stream:walkingtours"):
        raise ValueError(
            "full-pass runner supports SAYCam, BabyView and WalkingTours frame sources"
        )
    if cfg["stream"].get("transcript_json"):
        raise ValueError("full-pass MAE counts frames only; omit transcript_json")
    if offline:
        if window != batch:
            raise ValueError("IID has no sliding window; window_size must equal batch_size")
        if driver.get("subsample", 1) != 1 or driver["batch"] != {
            "slot": "frames",
            "n": batch,
            "views": ["reconstruction"],
        }:
            raise ValueError("IID must use every frame and the matched reconstruction batch")
        # OfflineDriver requires a budget at construction; replace this
        # placeholder after build has counted the source, before any training.
        driver["max_steps"] = 1
    else:
        expected = {
            "advance": {"kind": "events", "schema": "RawFrame", "n": stride},
            "rate": {"steps_per_admit": 1, "resample": True},
            "warmup": {"admitted": reference_window - stride},
        }
        if any(driver.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "streaming advance must equal stride; warmup must equal reference window - stride"
            )
        if cfg["memories"]["frames"]["tiers"]["window"] != {"capacity": window, "update": "fifo"}:
            raise ValueError("streaming FIFO capacity must equal window_size")
        slots = cfg["recipe"]["slots"]
        if slots != [
            {
                "name": "frames",
                "source": {"kind": "tier", "memory": "frames", "tier": "window"},
                "n": batch,
                "unit": "frame",
                "views": ["reconstruction"],
                "strict": True,
            }
        ] or cfg["routes"] != {"RawFrame": "frames"}:
            raise ValueError("streaming must select batch_size frames from the filled FIFO window")
    if str(cfg.get("device", "cuda")).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is visible")
    if (
        str(cfg.get("device", "cuda")).startswith("cuda")
        and any(loss.get("precision") == "bfloat16" for loss in cfg["losses"])
        and torch.cuda.get_device_capability(torch.device(cfg.get("device", "cuda")))[0] < 8
    ):
        raise RuntimeError("BF16 training requires a GPU with native BF16 support")
    pipe = egolearn.build(cfg)
    if not offline and (
        len(pipe.admission) != 1
        or not getattr(pipe.admission[0], "emits_each_frame", False)
        or pipe.admission[0].emits != ("RawFrame",)
    ):
        pipe.loader.close()
        raise ValueError("full-pass streaming requires a frame emitter without subsampling")
    frames = pipe.ctx.clock.total_observations
    if frames is None or frames < reference_window:
        pipe.loader.close()
        raise ValueError("full-pass MAE needs a finite dataset with at least one reference window")
    if control.get("expected_frames") is not None and frames != int(control["expected_frames"]):
        pipe.loader.close()
        raise ValueError(f"expected {control['expected_frames']} frames, found {frames}")
    # DataDrop changes selection, never admission or the stream advance.
    # The final partial advance still trains from a complete FIFO window.
    updates = math.ceil((frames - reference_window + stride) / stride)
    driver["max_steps"] = updates
    pipe.driver.max_steps = updates
    # A step-indexed schedule gives matched runs the same LR at every update,
    # including the final partial stream advance and IID epoch boundaries.
    if pipe.driver.schedule is not None:
        configured_steps = cfg["solver"]["schedule"].get("total_steps")
        if configured_steps is not None and configured_steps != updates:
            pipe.loader.close()
            raise ValueError("schedule.total_steps must equal the derived update budget")
        pipe.driver.schedule.total_steps = updates
        cfg["solver"]["schedule"]["total_steps"] = updates
    pipe.driver.run_config = deepcopy(cfg)
    for hook in pipe.driver.hooks:
        if isinstance(hook, WandbHook):
            hook._init_kw["config"] = deepcopy(cfg)
    return (
        pipe,
        cfg,
        {
            "frames": frames,
            "updates": updates,
            "batch_size": batch,
            "stride": stride,
            "updates_per_epoch": frames // batch,
            "window_size": 0 if offline else window,
            "reference_window_size": reference_window,
            "drop_fraction": 0 if offline else 1 - batch / window,
            "examples": updates * batch,
        },
    )


class UpdateSnapshots(Hook):
    """Milestones at the same optimizer update in both training modes."""

    def __init__(self, run_dir: Path, updates: int, options: dict, epoch_updates: int = 0):
        self.run_dir = run_dir
        self.every = int(options.get("snapshot_every", 0))
        self.pending = []
        if options.get("snapshot_each_epoch"):
            if epoch_updates < 1:
                raise ValueError("epoch checkpointing needs a positive epoch update count")
            for epoch in range(1, math.ceil(updates / epoch_updates) + 1):
                path = run_dir / f"snapshot_epoch_{epoch:03d}.pth"
                if not path.exists():
                    self.pending.append((min(epoch * epoch_updates, updates), path))
        for fraction in options.get("snapshot_milestones", []):
            if not 0 < fraction <= 1:
                raise ValueError("snapshot milestones must be in (0, 1]")
            path = run_dir / f"snapshot_{int(fraction * 100):03d}pct.pth"
            if not path.exists():
                self.pending.append((math.ceil(fraction * updates), path))

    def on_step(self, driver, logs):
        if driver.ctx.clock.compose_seq != driver.ctx.clock.step:
            raise RuntimeError("an update was skipped; preserving the last valid snapshot")
        step = driver.ctx.clock.step
        if self.every > 0 and step % self.every == 0:
            save_snapshot(str(self.run_dir / "snapshot_latest.pth"), driver)
        for threshold, path in self.pending[:]:
            if step >= threshold:
                save_snapshot(str(path), driver)
                self.pending.remove((threshold, path))


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class BudgetMetrics(Hook):
    """Log source consumption and model work separately, including after resume."""

    def __init__(self, budget: dict, driver):
        self.budget = budget
        self.started = time.monotonic()
        self.start_step = driver.ctx.clock.step
        self.start_frames = self._consumed_frames(driver)

    def _consumed_frames(self, driver) -> int:
        if isinstance(driver, OfflineDriver):
            return driver.ctx.clock.step * self.budget["batch_size"]
        return driver.ctx.clock.admitted

    def on_step(self, driver, logs) -> None:
        if driver.ctx.device.type == "cuda":
            torch.cuda.synchronize(driver.ctx.device)
        elapsed = max(time.monotonic() - self.started, 1e-9)
        step = driver.ctx.clock.step
        consumed = self._consumed_frames(driver)
        batch = self.budget["batch_size"]
        logs.update(
            {
                "train/consumed_frames": consumed,
                "train/window_size": self.budget["window_size"],
                "train/retained_examples": batch,
                "train/optimizer_updates": step,
                "train/examples_seen": step * batch,
                "lr": float(driver.optimizer.param_groups[0]["lr"]),
                "throughput/frames_per_second": (consumed - self.start_frames) / elapsed,
                "throughput/examples_per_second": (step - self.start_step) * batch / elapsed,
                "throughput/updates_per_second": (step - self.start_step) / elapsed,
            }
        )


class RunControl(Hook):
    def __init__(self, run_dir: Path, budget: dict, options: dict, *, log_every: int = 100):
        self.run_dir = run_dir
        self.budget = budget
        self.max_seconds = float(options.get("max_seconds", 0))
        self.stop_after_steps = int(options.get("stop_after_steps", 0))
        self.log_every = max(1, log_every)
        self.started = time.monotonic()
        self.requested = False
        self.last_logs: dict = {}

    def request_pause(self, signum=None, frame=None) -> None:
        # Signal handlers only set a flag; checkpointing waits for a safe boundary.
        self.requested = True

    def status(self, driver, state: str) -> None:
        _write_json(
            self.run_dir / "status.json",
            {
                **self.budget,
                "state": state,
                "step": driver.ctx.clock.step,
                "admitted": driver.ctx.clock.admitted,
                "seconds_this_attempt": time.monotonic() - self.started,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                **self.last_logs,
            },
        )

    def on_step(self, driver, logs):
        step = driver.ctx.clock.step
        # A skipped/nonfinite update invalidates the matched-update protocol.
        if driver.ctx.clock.compose_seq != step:
            raise RuntimeError("an update was skipped; refusing to continue an unmatched run")
        self.last_logs = {**logs, "lr": driver.optimizer.param_groups[0]["lr"]}
        if step == 1 or step % self.log_every == 0:
            self.status(driver, "running")
        elapsed = time.monotonic() - self.started
        preflight_stop = self.stop_after_steps > 0 and step >= self.stop_after_steps
        stop = (
            self.requested
            or (self.max_seconds > 0 and elapsed >= self.max_seconds)
            or preflight_stop
        )
        if stop and step < self.budget["updates"]:
            save_snapshot(str(self.run_dir / "snapshot_latest.pth"), driver)
            self.status(driver, "paused")
            raise PauseTraining(PREFLIGHT_STOP if preflight_stop else PAUSED)


def runtime_identity() -> dict:
    """Identify installed code and numerical dependencies, independent of location."""
    package = Path(egolearn.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode() + b"\0")
        digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        **{name: importlib.metadata.version(name) for name in ("torchvision", "numpy", "pillow")},
    }


def run(cfg: dict) -> int:
    """Run/resume a recipe; return 75 for requeue, 76 for a preflight stop."""
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep concurrent submissions from writing into the same run directory.
    with (run_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipe, resolved, budget = prepare(cfg)
        print(pipe.summary(), flush=True)
        print(f"[budget] {budget}", flush=True)
        if cfg.get("dry_run"):
            pipe.loader.close()
            return 0
        # Timing/stop controls can differ between attempts; scientific settings cannot.
        recipe = deepcopy(resolved)
        for key in ("resume", "dry_run", "run_control"):
            recipe.pop(key, None)
        # These affect throughput only; seeded batch selection and transforms
        # remain identical. Record them per attempt without blocking resume.
        recipe.pop("loader", None)
        recipe["driver"].pop("prefetch", None)
        manifest = {"config": recipe, "budget": budget, "runtime": runtime_identity()}
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("run manifest differs; use a new run_dir for a different recipe")
        else:
            _write_json(manifest_path, manifest)
        attempts = run_dir / "attempts"
        attempts.mkdir(exist_ok=True)
        _write_json(
            attempts / f"{time.time_ns()}.json",
            {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "loader": resolved.get("loader", {}),
                "prefetch": resolved["driver"].get("prefetch", False),
            },
        )
        snapshot = run_dir / "snapshot_latest.pth"
        if snapshot.exists():
            load_snapshot(str(snapshot), pipe.driver)
        elif (run_dir / "status.json").exists():
            raise RuntimeError("existing run has no resumable snapshot; use a new run_dir")
        else:
            # Recover even if the first allocation fails before update 5,000.
            save_snapshot(str(snapshot), pipe.driver)
        control = RunControl(
            run_dir,
            budget,
            cfg["run_control"],
            log_every=cfg.get("hooks", {}).get("log_every", 100),
        )
        pipe.driver.hooks = [
            BudgetMetrics(budget, pipe.driver),
            MetricsHook(str(run_dir), every=cfg.get("hooks", {}).get("log_every", 100)),
            *(h for h in pipe.driver.hooks if not isinstance(h, (SnapshotHook, MetricsHook))),
        ]
        pipe.driver.hooks.extend(
            [
                UpdateSnapshots(
                    run_dir, budget["updates"], cfg["hooks"], budget["updates_per_epoch"]
                ),
                control,
            ]
        )
        old_handler = signal.signal(signal.SIGUSR1, control.request_pause)
        control.status(pipe.driver, "running")
        try:
            pipe.driver.fit()
            if pipe.ctx.clock.step != budget["updates"]:
                raise RuntimeError("stream ended before the matched update budget was reached")
            if pipe.mode == "streaming" and pipe.ctx.clock.admitted != budget["frames"]:
                raise RuntimeError("streaming ended before all frames were consumed")
            save_snapshot(str(snapshot), pipe.driver)
            save_snapshot(str(run_dir / "final.pth"), pipe.driver)
            control.status(pipe.driver, "complete")
            return 0
        except PauseTraining as pause:
            return pause.exit_code
        except BaseException:
            control.status(pipe.driver, "failed")
            raise
        finally:
            signal.signal(signal.SIGUSR1, old_handler)
            pipe.loader.close()
            print(pipe.driver.timer.summary(), flush=True)


@hydra.main(version_base="1.3", config_path="pkg://egolearn.configs", config_name=None)
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    raise SystemExit(run(plain))


if __name__ == "__main__":
    main()
