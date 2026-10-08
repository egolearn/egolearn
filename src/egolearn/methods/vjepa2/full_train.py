"""Full chronological pass and update-matched IID V-JEPA 2 experiments.

Exit 75 requests a new allocation; 76 is a deliberately short preflight.
Neither pause writes a final evaluation checkpoint or changes the full schedule.
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
from egolearn.core import keys as K
from egolearn.core.events import make_event
from egolearn.data.index import index_digest, index_stream
from egolearn.methods.vjepa2.loss import VJEPA2
from egolearn.methods.vjepa2.train import check_batch
from egolearn.run_record import record_run
from egolearn.training.hooks import CheckpointHook, Hook, SnapshotHook, WandbHook
from egolearn.training.offline import OfflineDriver
from egolearn.utils.snapshot import load_snapshot, save_snapshot

PAUSED, PREFLIGHT_STOP = 75, 76


class PauseTraining(Exception):
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"checkpointed pause (exit {code})")


class UpdateSchedule:
    """Identical LR by completed update, independent of the stream clock."""

    def __init__(self, updates: int, warmup_frac: float, warmup_min_steps: int, floor: float):
        if not 0 <= warmup_frac < 1 or not 0 <= floor <= 1 or warmup_min_steps < 0:
            raise ValueError("invalid full-run schedule")
        self.updates = updates
        self.warmup = max(1, warmup_min_steps, math.ceil(updates * warmup_frac))
        self.floor = floor

    def multiplier(self, ctx) -> float:
        step = ctx.clock.step
        if step < self.warmup:
            return (step + 1) / self.warmup
        p = min(1.0, (step - self.warmup) / max(1, self.updates - self.warmup))
        return self.floor + (1 - self.floor) * (1 + math.cos(math.pi * p)) / 2

    def state_dict(self) -> dict:
        return dict(updates=self.updates, warmup=self.warmup, floor=self.floor)

    def load_state_dict(self, state: dict) -> None:
        if state != self.state_dict():
            raise ValueError("snapshot schedule differs from the full-run budget")


def prepare(config: dict):
    """Count the same complete clips used by both drivers and freeze the budget."""
    cfg = deepcopy(config)
    control, driver = cfg["run_control"], cfg["driver"]
    schedule = cfg["solver"]["schedule"]
    if set(schedule) - {"warmup_frac", "warmup_min_steps", "floor", "anchor"} or schedule.get(
        "anchor", False
    ):
        raise ValueError("full-run schedule supports only unanchored update warmup/cosine")
    batch, advance = int(control["batch_size"]), int(control["advance_clips"])
    if batch < 2 or not 1 <= advance <= batch:
        raise ValueError("require batch_size >= 2 and 1 <= advance_clips <= batch_size")
    if cfg.get("init_from") or cfg.get("resume"):
        raise ValueError("full runs initialize from scratch and resume their own latest snapshot")
    if driver.get("max_steps") is not None or driver.get("epochs") is not None:
        raise ValueError("full-run budget is derived; omit max_steps and epochs")
    if cfg["stream"].get("transcript_json"):
        raise ValueError("full clip runs do not accept transcript_json")
    if driver.get("grad_accumulation", 1) != 1 or driver.get("subsample", 1) != 1:
        raise ValueError("full clip runs require grad_accumulation=1 and subsample=1")
    clip = driver["batch"]
    if clip["n"] != batch or clip["unit"] != "clip" or clip["slot"] != "clips":
        raise ValueError("full-run batch must contain batch_size clips in slot clips")
    offline = driver.get("mode") == "offline"
    if not offline:
        emitter = {
            "_target_": "admission:dense_clip",
            **{k: clip[k] for k in ("clip_len", "clip_stride", "max_frame_gap")},
        }
        slot = {
            "name": "clips",
            "source": {"kind": "tier", "memory": "clips", "tier": "window"},
            "n": batch,
            "unit": "clip",
            "views": clip["views"],
            "strict": True,
            "clip_len": clip["clip_len"],
            "clip_stride": clip["clip_stride"],
        }
        expected = {
            "admission": [emitter],
            "memories": {
                "clips": {
                    "tiers": {"window": {"capacity": batch, "update": "fifo"}},
                    "admit_to": ["window"],
                    "overflow": {},
                }
            },
            "routes": {"VisualSegment": "clips"},
            "recipe": {"slots": [slot]},
        }
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError("streaming requires the complete matched dense-clip FIFO recipe")
        if (
            driver.get("advance")
            != {"kind": "events", "schema": "VisualSegment", "n": advance, "require_events": True}
            or driver.get("warmup") != {"events_emitted": batch - advance}
            or driver.get("rate") != {"steps_per_admit": 1, "resample": True}
            or driver.get("prefetch")
            or not driver.get("strict_batches")
        ):
            raise ValueError("streaming advance/warmup/rate must match the full-run budget")
    if str(cfg.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no GPU is visible")
    driver["max_steps"] = 1  # constructor placeholder only
    pipe = egolearn.build(cfg)
    try:
        refs, stats = index_stream(
            pipe.stream,
            **{k: clip[k] for k in ("clip_len", "clip_stride", "max_frame_gap", "expected_fps")},
        )
        n = len(refs)
        if n < batch:
            raise ValueError("dataset needs at least one complete clip batch")
        if pipe.stream.total_observations() != stats["frames"]:
            raise ValueError("full-run source must contain video frames only")
        for key, actual in (("expected_frames", stats["frames"]), ("expected_clips", n)):
            if control.get(key) is not None and control[key] != actual:
                raise ValueError(f"{key}={control[key]}, found {actual}")
        updates = math.ceil((n - batch + advance) / advance)
        budget = {
            **stats,
            "updates": updates,
            "batch_size": batch,
            "advance_clips": advance,
            "updates_per_epoch": n // batch,
            "inventory_sha256": index_digest(refs),
        }
        driver["max_steps"] = pipe.driver.max_steps = updates
        pipe.driver.run_config = cfg
        pipe.driver.index_stats = stats
        if offline:
            pipe.driver.expected_index_digest = budget["inventory_sha256"]
        schedule = cfg["solver"]["schedule"]
        pipe.driver.schedule = UpdateSchedule(
            updates, schedule["warmup_frac"], schedule["warmup_min_steps"], schedule["floor"]
        )
        if not pipe.losses or any(not isinstance(loss, VJEPA2) for loss in pipe.losses):
            raise ValueError("full runner requires V-JEPA 2 losses")
        if any(loss.ema[0] != loss.ema[1] for loss in pipe.losses):
            raise ValueError("matched full runs currently require constant EMA")
        return pipe, cfg, budget, refs
    except BaseException:
        pipe.loader.close()
        raise


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def runtime_identity() -> dict:
    package = Path(egolearn.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        **{k: importlib.metadata.version(k) for k in ("torchvision", "numpy", "pillow")},
    }


def manifest_for(cfg: dict, budget: dict) -> dict:
    scientific = deepcopy(cfg)
    for key in ("run_dir", "resume", "dry_run", "preflight", "record_json", "wandb", "hooks"):
        scientific.pop(key, None)
    scientific.pop("loader", None)
    scientific["driver"].pop("prefetch", None)
    scientific["run_control"] = {k: cfg["run_control"][k] for k in ("batch_size", "advance_clips")}
    return {"config": scientific, "budget": budget, "runtime": runtime_identity()}


class DeferredWandb(WandbHook):
    def on_end(self, driver):
        pass  # finish only after full source checks and exports


class RunControl(Hook):
    def __init__(self, root: Path, options: dict):
        self.root, self.options = root, options
        self.started = time.monotonic()
        self.requested = False
        self.budget: dict = {}
        self.last_logs: dict = {}
        self.state = "failed"

    def request_pause(self, signum=None, frame=None):
        self.requested = True

    def status(self, driver, state: str) -> None:
        self.state = state
        write_json(
            self.root / "status.json",
            {
                **self.budget,
                "state": state,
                "exports": (
                    {name: file_digest(self.root / name) for name in ("random.pth", "final.pth")}
                    if state == "complete"
                    else {}
                ),
                "step": driver.ctx.clock.step,
                "admitted": driver.ctx.clock.admitted,
                "events_emitted": driver.ctx.clock.events_emitted,
                "wandb_runs": [
                    {"id": getattr(h._run, "id", None), "url": getattr(h._run, "url", None)}
                    for h in driver.hooks
                    if isinstance(h, WandbHook) and h._run is not None
                ],
                "seconds_this_attempt": time.monotonic() - self.started,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                **self.last_logs,
            },
        )

    def on_skip(self, driver, *, nonfinite: bool) -> None:
        raise RuntimeError("full-run optimizer update skipped; refusing unmatched training")

    def on_step(self, driver, logs):
        step = driver.ctx.clock.step
        if driver.ctx.clock.compose_seq != step:
            raise RuntimeError("composed batches and optimizer updates differ")
        self.last_logs = {**logs, "lr": driver.optimizer.param_groups[0]["lr"]}
        every = int(self.options.get("snapshot_every", 5000))
        epoch = self.budget["updates_per_epoch"]
        if every > 0 and step % every == 0:
            save_snapshot(str(self.root / "snapshot_latest.pth"), driver)
        if self.options.get("snapshot_each_epoch", True) and step % epoch == 0:
            save_snapshot(str(self.root / f"snapshot_epoch_{step // epoch:03d}.pth"), driver)
        if step == 1 or step % int(self.options.get("log_every", 100)) == 0:
            self.status(driver, "running")
        cap = int(self.options.get("stop_after_steps", 0))
        seconds = float(self.options.get("max_seconds", 0))
        stop = cap > 0 and step >= cap
        if step < self.budget["updates"] and (
            stop or self.requested or (seconds > 0 and time.monotonic() - self.started >= seconds)
        ):
            save_snapshot(str(self.root / "snapshot_latest.pth"), driver)
            self.status(driver, "paused")
            raise PauseTraining(PREFLIGHT_STOP if stop else PAUSED)


def preflight_batch(pipe, refs):
    """Compose the exact next batch, without admission or clock/RNG changes."""
    driver, ctx = pipe.driver, pipe.ctx
    if isinstance(driver, OfflineDriver):
        if index_digest(refs) != driver.expected_index_digest:
            raise ValueError("preflight clip inventory changed")
        epoch, offset = divmod(ctx.clock.step, len(refs) // driver.n)
        order = list(range(len(refs)))
        ctx.fresh_rng(f"offline.shuffle.epoch{epoch}").shuffle(order)
        return driver._compose(driver._pick(refs, order, offset), seq=ctx.clock.compose_seq)
    n = pipe.recipe.slots[0].n
    start = min(ctx.clock.step * driver.advance["n"], len(refs) - n)
    # Composer already supports an explicit event head and sequence number.
    recipe = deepcopy(pipe.recipe)
    recipe.slots[0].source = {"kind": "stream_head", "schema": "VisualSegment"}
    events = [
        make_event(
            "VisualSegment",
            **{K.VIDEO_REF: r, K.TIME_SESSION: r.session, K.TIME_T0: r.t0, K.TIME_T1: r.t1},
        )
        for r in refs[start : start + n]
    ]
    return pipe.composer.compose(
        recipe, {}, {"VisualSegment": events}, ctx, seq=ctx.clock.compose_seq
    )


def _run_locked(cfg: dict, control: RunControl) -> int:
    root = control.root
    exit_code, pipe = 1, None
    manifest_valid = False
    attempt_path: Path | None = None
    try:
        with record_run(cfg, "train") as record:
            pipe, resolved, budget, refs = prepare(cfg)
            control.budget = budget
            if record is not None:
                record.value["config"] = resolved
                record.value["budget"] = budget
            print(pipe.summary(), flush=True)
            print(f"[budget] {budget}", flush=True)
            if cfg.get("dry_run"):
                exit_code = 0
                return 0
            manifest = manifest_for(resolved, budget)
            manifest_path = root / "manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text()) != manifest:
                    raise ValueError("scientific manifest differs; use a new run_dir")
            else:
                if any(root.glob("*.pth")):
                    raise ValueError("checkpoint directory has no scientific manifest")
                write_json(manifest_path, manifest)
            manifest_valid = True
            status_path = root / "status.json"
            if status_path.exists():
                previous = json.loads(status_path.read_text())
                if previous.get("state") == "complete":
                    exports = previous.get("exports", {})
                    if (
                        previous.get("step") != budget["updates"]
                        or set(exports) != {"random.pth", "final.pth"}
                        or any(
                            not (root / name).exists() or file_digest(root / name) != digest
                            for name, digest in exports.items()
                        )
                    ):
                        raise ValueError("completed run export verification failed")
                    if record is not None:
                        record.value["metrics"]["train/steps"] = budget["updates"]
                        record.value["already_complete"] = True
                    exit_code = 0
                    return 0
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            attempt_path = attempts / f"{time.time_ns()}.json"
            write_json(
                attempt_path,
                {"config": resolved, "job_id": os.environ.get("SLURM_JOB_ID")},
            )
            # Delay W&B completion; shared driver hooks must not export a paused final.
            hooks = []
            for hook in pipe.driver.hooks:
                if isinstance(hook, (CheckpointHook, SnapshotHook)):
                    continue
                if isinstance(hook, WandbHook):
                    hook.__class__ = DeferredWandb
                    hook._init_kw["config"] = resolved
                    hook._init_kw["group"] = f"vjepa2-{root.name}"
                hooks.append(hook)
            pipe.driver.hooks = [*hooks, *([record] if record is not None else []), control]
            snapshot, baseline = root / "snapshot_latest.pth", root / "random.pth"
            if snapshot.exists():
                if not baseline.exists():
                    raise ValueError("resumable run is missing its random baseline")
                load_snapshot(str(snapshot), pipe.driver)
                for loss in pipe.losses:
                    loss._synced = True
            else:
                for loss in pipe.losses:
                    loss.initialize_target(pipe.model)
                if baseline.exists():
                    saved = torch.load(baseline, map_location="cpu", weights_only=True)["model"]
                    current = pipe.model.state_dict()
                    if saved.keys() != current.keys() or any(
                        not torch.equal(saved[k], current[k].cpu()) for k in current
                    ):
                        raise ValueError(
                            "existing baseline differs from deterministic initialization"
                        )
                    old_status = (
                        json.loads((root / "status.json").read_text())
                        if (root / "status.json").exists()
                        else {}
                    )
                    if (
                        old_status.get("step", 0) > 0
                        or old_status.get("state") == "complete"
                        or (root / "final.pth").exists()
                    ):
                        raise ValueError("established run has no resumable snapshot")
                else:
                    CheckpointHook(str(root))._save(pipe.driver, "random.pth")
                save_snapshot(str(snapshot), pipe.driver)
            if cfg.get("preflight") and pipe.ctx.clock.step < budget["updates"]:
                diagnostics = check_batch(pipe, preflight_batch(pipe, refs))
                if record is not None:
                    record.value["preflight"] = diagnostics
            del refs
            control.status(pipe.driver, "running")
            pipe.driver.fit()
            if pipe.ctx.clock.step != budget["updates"]:
                raise RuntimeError("stream ended before the full update budget")
            if not isinstance(pipe.driver, OfflineDriver):
                while pipe.driver._step_stream() is not None:
                    if pipe.ctx.clock.events_emitted > budget["clips"]:
                        raise RuntimeError("stream emitted more clips than its inventory")
                if (
                    pipe.ctx.clock.admitted != budget["frames"]
                    or pipe.ctx.clock.events_emitted != budget["clips"]
                ):
                    raise RuntimeError("full stream counts differ from the inventory")
            save_snapshot(str(snapshot), pipe.driver)
            if control.options.get("snapshot_each_epoch", True):
                epoch = math.ceil(budget["updates"] / budget["updates_per_epoch"])
                save_snapshot(str(root / f"snapshot_epoch_{epoch:03d}.pth"), pipe.driver)
            CheckpointHook(str(root))._save(pipe.driver, "final.pth")
            control.status(pipe.driver, "complete")
            if record is not None:
                record.on_end(pipe.driver)
            exit_code = 0
            return 0
    except PauseTraining as pause:
        exit_code = 0  # intentional pause, not a failed W&B attempt
        return pause.code
    except BaseException:
        if pipe is not None and manifest_valid:
            control.status(pipe.driver, "failed")
        raise
    finally:
        if pipe is not None:
            try:
                pipe.loader.close()
            finally:
                for hook in pipe.driver.hooks:
                    if isinstance(hook, WandbHook) and hook._run is not None:
                        hook._run.summary["training_state"] = control.state
                        hook._run.finish(exit_code=exit_code)
                if attempt_path is not None:
                    attempt = json.loads(attempt_path.read_text())
                    attempt.update(
                        state=control.state,
                        exit_code=exit_code,
                        wandb_runs=[
                            {"id": getattr(h._run, "id", None), "url": getattr(h._run, "url", None)}
                            for h in pipe.driver.hooks
                            if isinstance(h, WandbHook) and h._run is not None
                        ],
                    )
                    write_json(attempt_path, attempt)


def run(cfg: dict) -> int:
    root = Path(cfg["run_dir"])
    if os.environ.get("SLURM_JOB_ID") and not root.is_absolute():
        raise ValueError("Slurm full runs require an absolute run_dir for safe resubmission")
    root.mkdir(parents=True, exist_ok=True)
    control = RunControl(root, {**cfg.get("hooks", {}), **cfg["run_control"]})
    old = signal.signal(signal.SIGUSR1, control.request_pause)
    try:
        with (root / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _run_locked(cfg, control)
    finally:
        signal.signal(signal.SIGUSR1, old)


@hydra.main(version_base="1.3", config_path="pkg://egolearn.configs", config_name=None)
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    raise SystemExit(run(plain))


if __name__ == "__main__":
    main()
