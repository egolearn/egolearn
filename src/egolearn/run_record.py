"""Atomic records shared by training, evaluation, and benchmark suites (no torch)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

from egolearn import __version__
from egolearn.hub import sha256_file
from egolearn.utils.input_provenance import input_identity, prepare_inputs

_ACTIVE: ContextVar[tuple[str, RunRecord | None] | None] = ContextVar("run_record", default=None)


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    """Replace a JSON file only after the complete payload has been serialized."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        if replace:
            os.replace(temporary, path)
        else:
            # Atomic exclusive creation: a repeated explicit path must not erase a run.
            os.link(temporary, path)
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
        self.path: Path | None = None
        self.value: dict[str, Any] = {
            "schema_version": 1,
            "entry": entry,
            "status": "running",
            "config": _plain(deepcopy(config)),
            "seed": config.get("seed", 1337 if entry == "train" else 0),
            "metrics": {},
            "provenance": source_identity(),
            "input_provenance": {},
        }

    @property
    def config(self) -> dict:
        return self.value["config"]

    def capture_output(self, name: str, path: str | Path) -> None:
        _, identity = input_identity(path, repo_type="model")
        self.value.setdefault("outputs", {})[name] = identity

    def training_provenance(self) -> dict:
        return deepcopy(
            {
                **self.value["provenance"],
                "seed": self.value["seed"],
                "input_provenance": self.value["input_provenance"],
            }
        )

    def evaluation(self, result) -> None:
        """Retain benchmark settings, versions and split metadata as well as scores."""
        evaluation = _plain(result.to_dict())
        json.dumps(evaluation, allow_nan=False)
        self.value["metrics"] = {
            f"{name}/{metric}": value
            for name, metrics in result.results.items()
            for metric, value in metrics.items()
        }
        self.value["evaluation"] = evaluation
        if result.checkpoint is not None:
            self.value["checkpoint"] = result.checkpoint

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
        self.value["model_config"] = deepcopy(
            getattr(getattr(driver, "model", None), "construction_config", None)
        )


@contextmanager
def record_run(
    config: dict[str, Any],
    entry: str,
    *,
    reuse: bool = False,
    inputs: dict[str, dict] | None = None,
) -> Iterator[RunRecord | None]:
    active = _ACTIVE.get()
    if reuse and active is not None and active[0] == entry:
        yield active[1]
        return
    destination = config.get("record_json", "auto")
    if not destination:
        token = _ACTIVE.set((entry, None))
        try:
            yield None
        finally:
            _ACTIVE.reset(token)
        return
    record = RunRecord(entry, config)
    if destination == "auto":
        root = Path(config.get("run_dir") or f"runs/{entry}")
        path = root / "records" / uuid.uuid4().hex / "record.json"
    else:
        path = Path(destination)
    record.path = path
    started = time.monotonic()
    atomic_json(path, record.value, replace=False)
    token = _ACTIVE.set((entry, record))
    try:
        record.value["input_provenance"] = (
            prepare_inputs(record.config) if inputs is None else deepcopy(inputs)
        )
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
        _ACTIVE.reset(token)
        record.value["wall_seconds"] = time.monotonic() - started
        atomic_json(path, record.value)


def recorded_fit(fit):
    """Give pipeline-built drivers the same record lifecycle as the training CLI."""

    @wraps(fit)
    def wrapped(driver, *args, **kwargs):
        config = getattr(driver, "run_config", None)
        if config is None:
            return fit(driver, *args, **kwargs)
        with record_run(
            config, "train", reuse=True, inputs=getattr(driver, "run_inputs", None)
        ) as record:
            if record is not None:
                record.value["seed"] = driver.ctx.seed
                record.value["start_step"] = driver.ctx.clock.step
            driver.run_provenance = record.training_provenance() if record is not None else None
            added = record is not None and record not in driver.hooks
            if added:
                driver.hooks.append(record)
            try:
                return fit(driver, *args, **kwargs)
            finally:
                if added:
                    driver.hooks.remove(record)

    return wrapped


def active_record(entry: str) -> RunRecord | None:
    active = _ACTIVE.get()
    return active[1] if active is not None and active[0] == entry else None
