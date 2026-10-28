"""Supervise independent full V-JEPA 2 runs sharing one Slurm allocation.

Only standard-library imports: install group signal handlers before launching
any training interpreter. GPU sharing is configured by the cluster, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO

BOOTSTRAP = (
    "import signal, runpy; signal.signal(signal.SIGUSR1, signal.SIG_IGN); "
    "runpy.run_module('egolearn.methods.vjepa2.full_train', run_name='__main__', alter_sys=True)"
)


@dataclass
class Launch:
    argv: list[str]
    root: Path
    log: Path
    threads: int


def parse_specs(value: list[dict]) -> list[Launch]:
    """Build commands from unique roots; readiness and training use the same path."""
    launches = []
    for spec in value:
        allowed = {"config_dir", "config_name", "overrides", "run_dir", "log", "threads"}
        if set(spec) - allowed:
            raise ValueError(f"unknown spec keys: {sorted(set(spec) - allowed)}")
        overrides = spec.get("overrides", [])
        if not isinstance(overrides, list) or any(not isinstance(x, str) for x in overrides):
            raise ValueError("overrides must be a list of strings")
        root = Path(spec["run_dir"]).resolve()
        threads = int(spec["threads"])
        if threads < 1:
            raise ValueError("each run requires a positive thread count")
        for override in overrides:
            key = override.split("=", 1)[0].lstrip("+~")
            if key in {"run_dir", "dry_run", "run_control"} or key.endswith("stop_after_steps"):
                raise ValueError(f"unsupported production override: {key}")
        launches.append(
            Launch(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    BOOTSTRAP,
                    "--config-dir",
                    str(Path(spec["config_dir"]).resolve()),
                    "--config-name",
                    spec["config_name"],
                    f"run_dir={root}",
                    f"hydra.run.dir={root}/hydra/${{now:%Y-%m-%d_%H-%M-%S}}",
                    *overrides,
                ],
                root,
                Path(spec["log"]).resolve(),
                threads,
            )
        )
    if not launches:
        raise ValueError("group requires at least one run")
    for i, entry in enumerate(launches):
        for other in launches[:i]:
            if (
                entry.root == other.root
                or entry.root in other.root.parents
                or other.root in entry.root.parents
            ):
                raise ValueError("run directories must be distinct and not nested")
            if entry.log == other.log:
                raise ValueError("child logs must be distinct")
    return launches


def _kill_group(process: subprocess.Popen, signum: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signum)


def supervise(launches: list[Launch], record: Path, grace_seconds: float = 480) -> int:
    """Pause peers on USR1 or exit 75; fail without requeue on any other failure."""
    if grace_seconds <= 0:
        raise ValueError("shutdown grace must be positive")
    if os.environ.get("SLURM_JOB_ID"):
        suffix = f"-{os.environ['SLURM_JOB_ID']}-r{os.environ.get('SLURM_RESTART_COUNT', '0')}"
        record = record.with_name(record.stem + suffix + record.suffix)
    requested = False
    stopped_at = None
    failed = False
    children: list[subprocess.Popen] = []
    handles: list[IO] = []
    seen: list[set[Path]] = []
    rows: list[dict] = []
    previous = {}

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    def save():
        record.parent.mkdir(parents=True, exist_ok=True)
        temporary = record.with_suffix(record.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
                    "runs": rows,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, record)

    try:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
        for entry in launches:
            if requested:
                break
            entry.log.parent.mkdir(parents=True, exist_ok=True)
            handle = entry.log.open("a")
            handles.append(handle)
            handle.write(
                f"\n[group allocation {os.environ.get('SLURM_JOB_ID', 'local')} "
                f"at {time.time_ns()}]\n"
            )
            handle.flush()
            seen.append(set((entry.root / "attempts").glob("*.json")))
            child = subprocess.Popen(
                entry.argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(entry.threads),
                    "MKL_NUM_THREADS": str(entry.threads),
                },
            )
            children.append(child)
            rows.append(
                {
                    "run_dir": str(entry.root),
                    "pid": child.pid,
                    "ready": False,
                    "signalled": False,
                    "returncode": None,
                    "argv": entry.argv,
                }
            )
        save()
        while any(child.poll() is None for child in children):
            changed = False
            for i, child in enumerate(children):
                code = child.poll()
                if code is not None and rows[i]["returncode"] is None:
                    rows[i]["returncode"] = code
                    changed = True
                    # Drain straggling workers immediately, while the process
                    # group still belongs to this child; never signal old PIDs later.
                    _kill_group(child, signal.SIGTERM)
                    if code != 0:
                        requested = True
                        failed |= code != 75
                if code is None:
                    if not rows[i]["ready"]:
                        fresh = set((launches[i].root / "attempts").glob("*.json")) - seen[i]
                        if fresh:
                            rows[i]["ready"] = True
                            changed = True
                    if requested and rows[i]["ready"] and not rows[i]["signalled"]:
                        with suppress(ProcessLookupError):
                            child.send_signal(signal.SIGUSR1)  # trainer PID only
                        rows[i]["signalled"] = True
                        changed = True
            if requested and stopped_at is None:
                stopped_at = time.monotonic()
            if changed:
                save()
            if (
                stopped_at is not None
                and time.monotonic() - stopped_at >= grace_seconds
                and any(child.poll() is None for child in children)
            ):
                failed = True
                break
            time.sleep(0.1)
        # The final child can exit between the loop condition and its poll body.
        codes = [child.poll() for child in children]
        for i, child in enumerate(children):
            if codes[i] is not None and rows[i]["returncode"] is None:
                _kill_group(child, signal.SIGTERM)
        failed |= any(code not in (0, 75) for code in codes)
        return 1 if failed else (75 if len(children) < len(launches) or 75 in codes else 0)
    finally:
        live = [child for child in children if child.poll() is None]
        for child in live:
            _kill_group(child, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while any(child.poll() is None for child in live) and time.monotonic() < deadline:
            time.sleep(0.05)
        for child in live:
            _kill_group(child, signal.SIGKILL)
            child.wait()
        for row, child in zip(rows, children, strict=True):
            row["returncode"] = child.returncode
        try:
            if rows:
                save()
        finally:
            for output in handles:
                output.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--grace-seconds", type=float, default=480)
    args = parser.parse_args()
    launches = parse_specs(json.loads(args.spec.read_text()))
    raise SystemExit(supervise(launches, args.record, args.grace_seconds))


if __name__ == "__main__":
    main()
