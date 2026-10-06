"""Optional atomic run records used by the CLI suite runner (no torch imports)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from egolearn import __version__
from egolearn.hub import sha256_file


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON file only after the complete payload has been serialized."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    def invalid_constant(value: str):
        raise ValueError(f"nonfinite JSON token {value}")

    return json.loads(path.read_text(), parse_constant=invalid_constant)


def source_checkout() -> Path | None:
    """Return the running package's editable checkout, if it has one."""
    root = Path(__file__).resolve().parents[2]
    if (root / "src/egolearn/run_record.py").resolve() == Path(__file__).resolve() and (
        root / ".git"
    ).exists():
        return root
    return None


def source_identity() -> dict[str, Any]:
    """Describe the running package, never an unrelated Git checkout in cwd."""
    identity: dict[str, Any] = {
        "commit": "unavailable",
        "dirty": "unavailable",
        "uv_lock_sha256": "unavailable",
        "image": os.environ.get("EGOLEARN_IMAGE", "unavailable"),
        "package_version": __version__,
    }
    # An editable src-layout checkout can be identified independently of cwd.
    if (root := source_checkout()) is not None:
        try:

            def git(*arguments: str) -> str:
                return subprocess.check_output(
                    ["git", "-C", str(root), *arguments],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()

            if Path(git("rev-parse", "--show-toplevel")).resolve() == root:
                identity["commit"] = git("rev-parse", "HEAD")
                identity["dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
                if (root / "uv.lock").is_file():
                    identity["uv_lock_sha256"] = sha256_file(root / "uv.lock")
                return identity
        except (OSError, subprocess.SubprocessError):
            pass
    # The immutable cluster archive has no .git. Its launcher records the commit
    # before building the wheel and passes the record's directory into the job.
    if os.environ.get("EGOLEARN_RUN_DIR"):
        try:
            run = Path(os.environ["EGOLEARN_RUN_DIR"])
            metadata = json.loads((run / "run.json").read_text())
            if metadata.get("commit") == os.environ.get("EGOLEARN_CODE_COMMIT"):
                identity["commit"] = metadata["commit"]
                identity["dirty"] = metadata.get("dirty", "unavailable")
                identity["uv_lock_sha256"] = sha256_file(run / "code/uv.lock")
        except (OSError, ValueError, KeyError):
            pass
    return identity


def finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    result = {}
    for key, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"metric {key!r} must be a finite number")
        result[key] = float(value)
    return result


class RunRecord:
    """Collect last-step training metrics without touching training state or RNGs."""

    mutates_sampling_state = False

    def __init__(self, entry: str, config: dict[str, Any]):
        self.value: dict[str, Any] = {
            "schema_version": 1,
            "entry": entry,
            "status": "running",
            "config": config,
            "seed": config.get("seed", 1337 if entry == "train" else 0),
            "metrics": {},
            "provenance": source_identity(),
        }

    def on_step(self, driver, logs: dict[str, float]) -> None:
        self.value["metrics"] = {f"train/last_step/{name}": value for name, value in logs.items()}
        self.value["metrics"]["train/last_step/step"] = driver.ctx.clock.step

    def on_admit(self, driver, n_events: int) -> None:
        pass

    def on_skip(self, driver, *, nonfinite: bool) -> None:
        self.value["metrics"]["train/steps"] = driver.ctx.clock.step
        self.value["metrics"]["train/skipped_steps"] = driver.skipped_steps
        self.value["nonfinite_steps"] = driver.nonfinite_steps
        if nonfinite:
            # A recorded run must not retry a diverged loss forever or report an
            # earlier finite minibatch as its final successful result.
            raise ValueError("training skipped nonfinite losses")

    def on_end(self, driver) -> None:
        self.value["metrics"]["train/steps"] = driver.ctx.clock.step
        self.value["metrics"]["train/skipped_steps"] = driver.skipped_steps
        self.value["nonfinite_steps"] = driver.nonfinite_steps
        for hook in driver.hooks:
            for name, metrics in getattr(hook, "last_results", {}).items():
                self.value["metrics"].update(
                    {f"probe/{name}/{key}": value for key, value in metrics.items()}
                )


@contextmanager
def record_run(config: dict[str, Any], entry: str) -> Iterator[RunRecord | None]:
    destination = config.get("record_json")
    if not destination:
        yield None
        return
    path = Path(destination)
    record = RunRecord(entry, config)
    started = time.monotonic()
    atomic_json(path, record.value)
    try:
        yield record
        if record.value.get("nonfinite_steps", 0):
            raise ValueError("training skipped nonfinite losses")
        record.value["metrics"] = finite_metrics(record.value["metrics"])
        record.value["status"] = "dry_run" if config.get("dry_run") else "completed"
    except BaseException as error:
        invalid = []
        finite = {}
        for name, value in record.value["metrics"].items():
            try:
                finite.update(finite_metrics({name: value}))
            except ValueError:
                invalid.append(name)
        record.value["status"] = (
            "nonfinite" if invalid or record.value.get("nonfinite_steps") else "failed"
        )
        record.value["error_type"] = type(error).__name__
        record.value["nonfinite_metrics"] = invalid
        record.value["metrics"] = finite
        raise
    finally:
        record.value["wall_seconds"] = time.monotonic() - started
        atomic_json(path, record.value)
