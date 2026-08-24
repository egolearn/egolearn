"""Hooks the driver calls at fixed points in its loop: logging, checkpointing, probes,
snapshots.

A hook is a small object the driver calls after every training step, after every stream
advance, and once at the end of the run.  The driver's loop is deliberately tiny; everything
periodic lives in a hook.  A project that needs a genuinely different loop should subclass the
driver instead of stacking more hooks.
"""

from __future__ import annotations

import os
from typing import Any

import torch


class Hook:
    def on_step(self, driver, logs: dict[str, float]) -> None:
        pass

    def on_admit(self, driver, n_events: int) -> None:
        pass

    def on_end(self, driver) -> None:
        pass


class LogHook(Hook):
    def __init__(self, every: int = 50, timing: bool = True):
        self.every = int(every)
        self.timing = bool(timing)

    def on_step(self, driver, logs):
        step = driver.ctx.clock.step
        if step % self.every != 0 and step != 1:
            return
        prog = driver.ctx.clock.progress
        parts = [f"[step {step:6d}]", f"progress={prog:.3f}" if prog is not None else "progress=?"]
        parts += [f"{k}={v:.4f}" for k, v in sorted(logs.items())]
        parts += [m.stats() for m in driver.memories.values()]
        print("  ".join(parts), flush=True)
        if self.timing and getattr(driver, "timer", None) is not None:
            print(driver.timer.summary(), flush=True)


class CheckpointHook(Hook):
    def __init__(self, run_dir: str, every_steps: int = 500):
        self.run_dir = run_dir
        self.every = int(every_steps)

    def on_step(self, driver, logs):
        if driver.ctx.clock.step % self.every:
            return
        self._save(driver, f"step_{driver.ctx.clock.step}.pth")

    def on_end(self, driver):
        self._save(driver, "final.pth")

    def _save(self, driver, fname: str):
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, fname)
        torch.save(
            {
                "model": driver.model.state_dict(),
                "optimizer": driver.optimizer.state_dict(),
                "step": driver.ctx.clock.step,
                "export_contract": driver.model.export_contract(),
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)


class ProbeHook(Hook):
    """Runs evaluation probes (small held-out evaluations of the current model) every
    `every_steps` gradient steps and once at the end, so the model can be scored at any
    point during training rather than only after it finishes."""

    def __init__(self, probes: list, every_steps: int = 500):
        self.probes = probes
        self.every = int(every_steps)
        self.history: list[dict] = []

    def on_step(self, driver, logs):
        if driver.ctx.clock.step % self.every:
            return
        self._run(driver)

    def on_end(self, driver):
        self._run(driver)

    def _run(self, driver):
        driver.model.eval()
        with torch.no_grad():
            for probe in self.probes:
                metrics = probe.run(driver.model, driver.ctx)
                record = {
                    "step": driver.ctx.clock.step,
                    "progress": driver.ctx.clock.progress,
                    **metrics,
                }
                self.history.append(record)
                print(f"[probe {probe.name}] {record}", flush=True)
        driver.model.train()


class SnapshotHook(Hook):
    """Saves snapshots -- the full resumable state of a run (model, optimizer, memories,
    clocks).  Triggered at progress milestones (BabyCL's 5/25/50/100% of the stream)
    and/or every `every_steps` steps for crash recovery, in which case
    `snapshot_latest.pth` is overwritten in place."""

    def __init__(self, run_dir: str, milestones=(0.05, 0.25, 0.5, 1.0), every_steps: int = 0):
        self.run_dir = run_dir
        self.milestones = list(milestones or [])
        self.every_steps = int(every_steps)
        self._done: set[float] = set()

    def on_step(self, driver, logs):
        from ..utils.snapshot import save_snapshot

        step = driver.ctx.clock.step
        if self.every_steps > 0 and step % self.every_steps == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            save_snapshot(os.path.join(self.run_dir, "snapshot_latest.pth"), driver)
        prog = driver.ctx.clock.progress
        if prog is None:
            return
        for m in self.milestones:
            if m not in self._done and prog + 1e-9 >= m:
                self._done.add(m)
                path = os.path.join(self.run_dir, f"snapshot_{int(m * 100):03d}pct.pth")
                if os.path.exists(path):
                    # `_done` is per-process, so a RESUMED run re-crosses
                    # milestones the original already recorded; milestone
                    # snapshots are write-once archives, never overwritten
                    continue
                os.makedirs(self.run_dir, exist_ok=True)
                save_snapshot(path, driver)


class WandbHook(Hook):
    """Config-driven Weights & Biases logging (`wandb.*` keys in the root
    config). Constructed only when `wandb.mode` != "disabled", so the wandb
    package (train extra) is imported lazily and never required otherwise."""

    def __init__(
        self, entity=None, project=None, mode="online", run_name=None, config=None, every: int = 1
    ):
        self.every = max(1, int(every))
        self._init_kw = {
            "entity": entity,
            "project": project,
            "mode": mode,
            "name": run_name,
            "config": config,
        }
        self._run: Any = None  # started on first use: dry runs never open a W&B run

    def _ensure_run(self):
        if self._run is None:
            import wandb

            self._run = wandb.init(**self._init_kw)
        return self._run

    def on_step(self, driver, logs):
        run = self._ensure_run()  # first training step opens the run, even
        step = driver.ctx.clock.step  # when every=N would skip its logging
        if step % self.every:
            return
        payload = dict(logs)
        prog = driver.ctx.clock.progress
        if prog is not None:
            payload["progress"] = prog
        run.log(payload, step=step)

    def on_end(self, driver):
        if self._run is not None:
            self._run.finish()
