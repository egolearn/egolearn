"""Exercise submission/preparation with fake Slurm and Apptainer, using clean Git repos."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(args, *, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)


def commit(checkout):
    for args in (["add", "."], ["commit", "-qm", "fixture"]):
        subprocess.run(["git", *args], cwd=checkout, check=True)


@pytest.fixture
def cluster(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "scripts/slurm", checkout / "scripts/slurm")
    shutil.copytree(ROOT / "containers", checkout / "containers")
    (checkout / "uv.lock").write_text("fixture lock")
    (checkout / "pyproject.toml").write_text("fixture project")
    (checkout / ".python-version").write_text("3.12")
    (checkout / ".gitignore").write_text("env.sh\n")
    (checkout / "program.py").write_text("print('submitted version')\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=checkout, check=True)
    commit(checkout)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['SBATCH_LOG'],'a') as f:\n"
        " f.write(json.dumps({'args':sys.argv[1:],'env':dict(os.environ)})+'\\n')\n"
        "print('12345')\n"
    )
    sbatch.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "SCRATCH": str(tmp_path / "scratch"),
    }
    # Use the actual example to exercise the required cache variables too.
    loaded = run(
        ["bash", "-c", "source scripts/slurm/env.example.sh; env -0"], cwd=checkout, env=env
    )
    env = dict(item.split("=", 1) for item in loaded.stdout.split("\0") if "=" in item)
    env.update(PATH=f"{bindir}:{os.environ['PATH']}", SBATCH_LOG=str(tmp_path / "sbatch.jsonl"))
    env["EGOLEARN_PREP_SBATCH_ARGS"] = "--partition=cpu"
    env["SBATCH_PARTITION"] = "gpu"
    return checkout, env


def submit(cluster, *args, ok=True):
    checkout, env = cluster
    log = Path(env["SBATCH_LOG"])
    log.unlink(missing_ok=True)
    result = run(["bash", "scripts/slurm/submit", *args], cwd=checkout, env=env)
    assert (result.returncode == 0) == ok, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def ready(call):
    env = call["env"]
    image = Path(env["SIF"])
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    venv = Path(env["EGOLEARN_VENV"])
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin/python").write_text("#!/bin/sh\nexit 0\n")
    (venv / "bin/python").chmod(0o755)
    (venv / ".egolearn-sync-stamp").write_text(env["EGOLEARN_EXPECTED_STAMP"])
    (venv / ".egolearn-sync.lock").touch()


def test_prep_chaining_cache_reuse_and_snapshot(cluster):
    checkout, _ = cluster
    _, calls = submit(cluster, "--dependency=afterok:9", "--", "python", "program.py")
    assert len(calls) == 2
    assert "--partition=cpu" in calls[0]["args"]
    assert "SBATCH_PARTITION" not in calls[0]["env"]
    assert "--dependency=afterok:12345,afterok:9" in calls[1]["args"]
    job = Path(calls[1]["env"]["EGOLEARN_RUN_DIR"])
    assert json.loads((job / "run.json").read_text())["dirty"] is False
    assert (job / "dirty.diff").read_text() == ""
    assert (job / "job-id").read_text().strip() == "12345"
    assert (job / "code/program.py").stat().st_mode & 0o222 == 0
    ready(calls[1])
    _, reused = submit(cluster, "--", "true")
    assert len(reused) == 1
    (checkout / "program.py").write_text("print('edited while queued')\n")
    assert (job / "work/program.py").read_text() == "print('submitted version')\n"


@pytest.mark.parametrize("change", ["tracked", "untracked", "staged"])
def test_dirty_checkout_refused(cluster, change):
    checkout, _ = cluster
    (checkout / ("new.py" if change == "untracked" else "program.py")).write_text("changed")
    if change == "staged":
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    result, calls = submit(cluster, "--", "true", ok=False)
    assert "dirty checkout" in result.stderr
    assert not calls


def test_ignored_credentials_not_snapshotted(cluster):
    checkout, _ = cluster
    (checkout / "env.sh").write_text("synthetic settings")
    _, calls = submit(cluster, "--", "true")
    assert not (Path(calls[-1]["env"]["EGOLEARN_RUN_DIR"]) / "code/env.sh").exists()


@pytest.mark.parametrize("change", ["lock", "extras", "image", "code"])
def test_environment_stamp_covers_dependencies_image_and_code(cluster, change):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    ready(calls[-1])
    old = calls[-1]["env"]["EGOLEARN_EXPECTED_STAMP"]
    if change == "lock":
        (checkout / "uv.lock").write_text("new lock")
    elif change == "extras":
        env["EGOLEARN_SYNC_ARGS"] = "--extra video"
    elif change == "image":
        (checkout / "containers/IMAGE").write_text("ghcr.io/lab/base@sha256:" + "a" * 64)
    else:
        (checkout / "program.py").write_text("new code")
    if change != "extras":
        commit(checkout)
    _, changed = submit(cluster, "--", "true")
    assert len(changed) == 2
    assert changed[-1]["env"]["EGOLEARN_EXPECTED_STAMP"] != old


def test_sweep_and_argument_preservation(cluster):
    _, calls = submit(
        cluster,
        "--sweep",
        "--max-parallel",
        "2",
        "--",
        "python",
        "program.py",
        "seed=1,2",
        "group@package=a,b",
        "layers=[1,2]",
        'label="a, b"',
        "literal=$(id)",
    )
    job = Path(calls[-1]["env"]["EGOLEARN_RUN_DIR"])
    rows = [shlex.split(row) for row in (job / "commands.sh").read_text().splitlines()]
    assert len(rows) == 4
    assert rows[0][-3:] == ["layers=[1,2]", 'label="a, b"', "literal=$(id)"]
    assert "--array=0-3%2" in calls[-1]["args"]
    assert rows[-1][2:4] == ["seed=2", "group@package=b"]


def test_dry_run_does_not_submit(cluster):
    result, calls = submit(cluster, "--dry-run", "--", "true")
    assert not calls
    assert "afterok:PREP_JOB_ID" in result.stdout


@pytest.mark.parametrize("name", ["HF_HOME", "WANDB_DIR", "EGOLEARN_JOB_ROOT", "UV_CACHE_DIR"])
def test_home_refused(cluster, name):
    _, env = cluster
    env[name] = env["HOME"] + "/bad"
    result, calls = submit(cluster, "--", "true", ok=False)
    assert "$HOME" in result.stderr
    assert not calls


def test_required_cache_cannot_fall_back_to_home(cluster):
    _, env = cluster
    del env["HF_HOME"]
    result, _ = submit(cluster, "--", "true", ok=False)
    assert "set HF_HOME" in result.stderr


@pytest.mark.parametrize("gpu", [False, True])
def test_run_uses_snapshot_and_readonly_bind(cluster, gpu):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    call = calls[-1]
    ready(call)
    fake = Path(env["SBATCH_LOG"]).parent / "bin/apptainer"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "print(json.dumps({'args':sys.argv[1:],'bind':os.environ['APPTAINER_BIND'],"
        "'cwd':os.getcwd(),'path':os.environ['APPTAINERENV_PREPEND_PATH']}))\n"
    )
    fake.chmod(0o755)
    child_env = {**call["env"], "SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "1" if gpu else "0"}
    child_env.pop("CUDA_VISIBLE_DEVICES", None)
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    result = run(
        ["bash", str(job / "code/scripts/slurm/run.sbatch"), "true"], cwd=checkout, env=child_env
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.splitlines()[-1])
    assert ("--nv" in data["args"]) is gpu
    assert f"{job}/code:{job}/code:ro" in data["bind"]
    assert data["cwd"] == str(job / "work")
    assert data["path"] == child_env["EGOLEARN_VENV"] + "/bin"


def test_prep_pulls_on_cpu_and_syncs_in_container(cluster):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    fake = Path(env["SBATCH_LOG"]).parent / "bin/apptainer"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "with open(os.environ['PREP_LOG'],'a') as f:\n"
        " f.write(json.dumps({'args':sys.argv[1:],'tmp':os.environ['APPTAINER_TMPDIR']})+'\\n')\n"
        "if sys.argv[1]=='pull': Path(sys.argv[2]).touch()\n"
        "if 'venv' in sys.argv:\n"
        " p=Path(os.environ['EGOLEARN_VENV'])/'bin/python'\n"
        " p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture'); p.chmod(0o755)\n"
    )
    fake.chmod(0o755)
    child_env = {
        **calls[0]["env"],
        "SLURM_JOB_ID": "12",
        "PREP_LOG": str(checkout.parent / "prep.log"),
    }
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    result = run(["bash", str(job / "code/scripts/slurm/prep.sbatch")], cwd=checkout, env=child_env)
    assert result.returncode == 0, result.stderr
    prep_calls = [json.loads(line) for line in Path(child_env["PREP_LOG"]).read_text().splitlines()]
    assert prep_calls[0]["args"][0] == "pull"
    assert all(call["tmp"].startswith("/tmp/egolearn-prep.") for call in prep_calls)
    sync = next(call["args"] for call in prep_calls if "sync" in call["args"])
    assert "--frozen" in sync and "--no-editable" in sync
    assert "--nv" not in sync
    assert (
        Path(child_env["EGOLEARN_VENV"]) / ".egolearn-sync-stamp"
    ).read_text().strip() == child_env["EGOLEARN_EXPECTED_STAMP"]


def test_array_workspaces_are_independent(cluster):
    _, calls = submit(cluster, "--sweep", "--", "python", "program.py", "seed=1,2")
    job = Path(calls[-1]["env"]["EGOLEARN_RUN_DIR"])
    (job / "tasks/0/work/results.json").write_text("first task")
    assert not (job / "tasks/1/work/results.json").exists()
    assert (job / "tasks/1/work/program.py").resolve() == job / "code/program.py"


def test_relative_output_resolves_before_chdir(cluster):
    checkout, _ = cluster
    _, calls = submit(cluster, "--output", "logs/job-%j.out", "--", "true")
    assert f"--output={checkout}/logs/job-%j.out" in calls[-1]["args"]


@pytest.mark.parametrize("option", ["--array=1-2", "--export=NONE", "--chdir=/tmp", "--wrap=true"])
def test_reserved_options_refused(cluster, option):
    result, calls = submit(cluster, option, "--", "true", ok=False)
    assert "managed by submit" in result.stderr
    assert not calls
