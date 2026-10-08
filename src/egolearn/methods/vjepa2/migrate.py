"""Explicit, audited continuation after a reviewed throughput-only source change.

The regular runner deliberately refuses source mismatches. This helper creates a
new directory, preserving the old run and recording both runtime identities. Use
only on trusted local snapshots, after verifying that the code change preserves
training semantics; config equality cannot establish that by itself.
"""

from __future__ import annotations

import fcntl
import json
import shutil
from pathlib import Path

import torch

from egolearn.methods.vjepa2.full_train import file_digest, manifest_for, prepare, write_json
from egolearn.utils.snapshot import load_snapshot


def fork_run(config: dict, source: str, *, source_sha256: str, reason: str) -> dict:
    """Copy the most advanced valid snapshot into a fresh, verified run directory.

    Scientific config, inventory, schedule, dependency versions, deterministic
    baseline and snapshot state must agree. Only the package source digest may
    differ, explicitly acknowledged by the caller with its old digest and reason.
    Metrics are copied intact; MetricsHook archives and trims any abandoned tail
    when training resumes. No status, locks or attempt records are inherited.
    """
    old, new = Path(source).resolve(), Path(config["run_dir"]).resolve()
    if new.exists() or old == new or old in new.parents:
        raise ValueError("migration needs a fresh run_dir outside the source run")
    if not reason.strip():
        raise ValueError("migration requires a reviewed reason")
    # Same lock used by full_train.run. The source must be inactive throughout.
    with (old / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((old / "manifest.json").read_text())
        if previous["runtime"]["source_sha256"] != source_sha256:
            raise ValueError("source digest does not match the reviewed runtime")
        pipe, resolved, budget, _ = prepare(config)
        try:
            manifest = manifest_for(resolved, budget)
            if any(previous[k] != manifest[k] for k in ("config", "budget")):
                raise ValueError("migration changes the scientific config or inventory")
            old_versions = {k: v for k, v in previous["runtime"].items() if k != "source_sha256"}
            new_versions = {k: v for k, v in manifest["runtime"].items() if k != "source_sha256"}
            if old_versions != new_versions:
                raise ValueError("migration changes dependency versions")
            for loss in pipe.losses:
                loss.initialize_target(pipe.model)
            baseline = torch.load(old / "random.pth", map_location="cpu", weights_only=True)
            current = pipe.model.state_dict()
            if baseline["model"].keys() != current.keys() or any(
                not torch.equal(baseline["model"][key], current[key].cpu()) for key in current
            ):
                raise ValueError("baseline differs from deterministic initialization")
            del baseline, current
            candidates = []
            for path in sorted(old.glob("snapshot_*.pth")):
                state = torch.load(path, map_location="cpu", weights_only=False)
                step = state["context"]["clock"]["step"]
                if (
                    state["context"]["clock"]["compose_seq"] != step
                    or not 0 <= step <= budget["updates"]
                    or state["schedule"] != pipe.driver.schedule.state_dict()
                    or state.get("driver", {}).get("skipped_steps", 0)
                    or state.get("driver", {}).get("nonfinite_steps", 0)
                ):
                    raise ValueError(f"invalid continuation snapshot: {path.name}")
                candidates.append((step, path))
                del state
            if not candidates:
                raise ValueError("source has no continuation snapshot")
            step, selected = max(candidates, key=lambda row: (row[0], row[1].name))
            # Exercises strict model/optimizer/schedule restore and stream replay.
            load_snapshot(str(selected), pipe.driver)
            if pipe.ctx.clock.step != step:
                raise ValueError("restored clock differs from the selected snapshot")
            copies = {"random.pth": old / "random.pth", "snapshot_latest.pth": selected}
            copies.update(
                {
                    path.name: path
                    for saved_step, path in candidates
                    if saved_step <= step and path.name.startswith("snapshot_epoch_")
                }
            )
            if (old / "metrics.jsonl").exists():
                copies["metrics.jsonl"] = old / "metrics.jsonl"
            record = {
                "source": str(old),
                "destination": str(new),
                "reason": reason,
                "source_manifest": previous,
                "destination_manifest": manifest,
                "selected_snapshot": selected.name,
                "step": step,
                "sha256": {name: file_digest(path) for name, path in copies.items()},
            }
            new.mkdir(parents=True, exist_ok=False)
            try:
                for name, path in copies.items():
                    shutil.copy2(path, new / name)
                    if file_digest(new / name) != record["sha256"][name]:
                        raise OSError(f"copy verification failed: {name}")
                write_json(new / "migration.json", record)
                # Publish the manifest last; an interrupted copy cannot be resumed.
                write_json(new / "manifest.json", manifest)
            except BaseException:
                shutil.rmtree(new)
                raise
            return record
        finally:
            pipe.loader.close()
