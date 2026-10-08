"""Real subprocess checks for grouped pause and failure handling, without GPUs."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from egolearn.commands.train_group import BOOTSTRAP, Launch, parse_specs, supervise

FAKE = """
import json, os, signal, subprocess, sys, time
from pathlib import Path
root, mode, delay = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
root.mkdir(parents=True, exist_ok=True)
paused = False
def pause(*args):
    global paused
    paused = True
if mode == 'earlyfail':
    raise SystemExit(1)
time.sleep(delay)
signal.signal(signal.SIGUSR1, pause if mode != 'ignore' else signal.SIG_IGN)
(root/'attempts').mkdir(exist_ok=True)
(root/'attempts'/f'{time.time_ns()}.json').write_text('{}')
if mode == 'signal':
    os.kill(os.getppid(), signal.SIGUSR1)
    os.kill(os.getppid(), signal.SIGUSR1)
if mode == 'ignore':
    grandchild = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    (root/'grandchild.pid').write_text(str(grandchild.pid))
deadline=time.monotonic()+(.35 if mode in ('done','pause','fail','cap') else 20)
while not paused and time.monotonic()<deadline: time.sleep(.01)
code = 75 if paused or mode=='pause' else {'done':0,'fail':1,'cap':76}.get(mode,1)
(root/'result.json').write_text(json.dumps({'code':code,'omp':os.environ['OMP_NUM_THREADS']}))
raise SystemExit(code)
"""


@pytest.fixture
def launches(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(FAKE)

    def build(*modes):
        return [
            Launch(
                [sys.executable, str(script), str(tmp_path / str(i)), mode, str(delay)],
                tmp_path / str(i),
                tmp_path / f"{i}.log",
                2,
            )
            for i, (mode, delay) in enumerate(modes)
        ]

    return build


@pytest.mark.parametrize(
    "mode,code", [("done", 0), ("pause", 75), ("fail", 1), ("cap", 1), ("earlyfail", 1)]
)
def test_group_propagates_pause_and_failures(tmp_path, launches, mode, code):
    peers = [("done", 0)] if mode == "done" else [("wait", 0.6)]
    specs = launches((mode, 0), *peers)
    assert supervise(specs, tmp_path / "group.json", grace_seconds=5) == code
    record = json.loads((tmp_path / "group.json").read_text())
    assert all(row["returncode"] is not None for row in record["runs"])
    expected_peer = 0 if mode == "done" else 75
    peer = json.loads((specs[1].root / "result.json").read_text())
    assert peer == {"code": expected_peer, "omp": "2"}


def test_group_repeated_signal_waits_for_new_readiness(tmp_path, launches):
    specs = launches(("signal", 0.1), ("wait", 0.6))
    stale = specs[1].root / "attempts"
    stale.mkdir(parents=True)
    (stale / "old.json").write_text("{}")
    assert supervise(specs, tmp_path / "group.json", grace_seconds=5) == 75
    assert json.loads((specs[1].root / "result.json").read_text())["code"] == 75


def test_group_timeout_kills_process_group(tmp_path, launches):
    specs = launches(("pause", 0.4), ("ignore", 0))
    assert supervise(specs, tmp_path / "group.json", grace_seconds=0.4) == 1
    pid = int((specs[1].root / "grandchild.pid").read_text())
    deadline = time.monotonic() + 3
    while Path(f"/proc/{pid}/stat").exists() and time.monotonic() < deadline:
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            break  # dead, awaiting the operating system's reaper
        time.sleep(0.05)
    if Path(f"/proc/{pid}/stat").exists():
        assert Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"


def test_stop_during_launch_does_not_start_remaining_children(tmp_path, launches, monkeypatch):
    from egolearn.commands import train_group

    original = train_group.subprocess.Popen

    def stop_after_launch(*args, **kwargs):
        child = original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGUSR1)
        return child

    monkeypatch.setattr(train_group.subprocess, "Popen", stop_after_launch)
    specs = launches(("wait", 0.1), ("wait", 0))
    assert supervise(specs, tmp_path / "group.json", grace_seconds=5) == 75
    assert not specs[1].root.exists()


def test_bootstrap_ignores_late_usr1(tmp_path):
    marker = tmp_path / "returned"
    # The stand-in module restores the handler exactly as full_train.run does.
    code = (
        "import runpy, signal, time; from pathlib import Path; "
        "runpy.run_module=lambda *a,**k: signal.signal(signal.SIGUSR1, "
        "signal.signal(signal.SIGUSR1, lambda *a: None)); "
        + BOOTSTRAP
        + f"; Path({str(marker)!r}).touch(); time.sleep(.5)"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.01)
        assert marker.exists()
        child.send_signal(signal.SIGUSR1)
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize(
    "invalid", ["duplicate", "nested", "run_dir", "dry_run", "stop", "threads", "log"]
)
def test_specs_reject_ambiguous_or_partial_runs(tmp_path, invalid):
    spec = {
        "run_dir": str(tmp_path / "run"),
        "log": str(tmp_path / "out"),
        "threads": 4,
        "config_dir": "projects/vjepa2/configs",
        "config_name": "full_babyview_iid",
    }
    specs = [spec]
    if invalid in ("duplicate", "nested", "log"):
        second = {**spec, "log": str(tmp_path / "other.out")}
        if invalid == "nested":
            second["run_dir"] = str(tmp_path / "run" / "nested")
        elif invalid == "log":
            second.update(run_dir=str(tmp_path / "other"), log=spec["log"])
        specs.append(second)
    elif invalid == "threads":
        spec["threads"] = 0
    else:
        override = {
            "run_dir": "++run_dir=elsewhere",
            "dry_run": "dry_run=true",
            "stop": "run_control.stop_after_steps=1",
        }[invalid]
        spec["overrides"] = [override]
    with pytest.raises(ValueError):
        parse_specs(specs)
