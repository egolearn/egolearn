"""Exercise submission/preparation with fake Slurm and Apptainer, using clean Git repos."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not shutil.which("flock") or not shutil.which("sha256sum"),
    reason="Slurm launcher tests require Linux flock and sha256sum",
)
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
    resource = Path("src/egolearn/commands/cluster_paths.txt")
    (checkout / resource).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / resource, checkout / resource)
    (checkout / "uv.lock").write_text("fixture lock")
    (checkout / "pyproject.toml").write_text("fixture project")
    (checkout / ".python-version").write_text("3.12")
    shutil.copyfile(ROOT / ".gitignore", checkout / ".gitignore")
    (checkout / "program.py").write_text("print('submitted version')\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=checkout, check=True)
    commit(checkout)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['TEST_SUBMIT_LOG'],'a') as f:\n"
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
    env.update(
        PATH=f"{bindir}:{os.environ['PATH']}", TEST_SUBMIT_LOG=str(tmp_path / "sbatch.jsonl")
    )
    env["EGOLEARN_PREP_SBATCH_ARGS"] = "--partition=cpu"
    env["SBATCH_PARTITION"] = "gpu"
    return checkout, env


def submit(cluster, *args, ok=True):
    checkout, env = cluster
    log = Path(env["TEST_SUBMIT_LOG"])
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
    (checkout / "env.sh").write_text("synthetic settings")
    _, calls = submit(cluster, "--dependency=afterok:9", "--", "python", "program.py")
    assert len(calls) == 2
    assert "--partition=cpu" in calls[0]["args"]
    assert "SBATCH_PARTITION" not in calls[0]["env"]
    assert "--dependency=afterok:12345,afterok:9" in calls[1]["args"]
    job = Path(calls[1]["env"]["EGOLEARN_RUN_DIR"])
    assert not (job / "code/env.sh").exists()
    assert json.loads((job / "run.json").read_text())["dirty"] is False
    assert (job / "dirty.diff").read_text() == ""
    assert (job / "job-id").read_text().strip() == "12345"
    assert (job / "code/program.py").stat().st_mode & 0o222 == 0
    ready(calls[1])
    _, reused = submit(cluster, "--", "true")
    assert len(reused) == 1
    (checkout / "program.py").write_text("print('edited while queued')\n")
    assert (job / "work/program.py").read_text() == "print('submitted version')\n"


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_dirty_checkout_refused(cluster, change):
    checkout, _ = cluster
    (checkout / ("new.py" if change == "untracked" else "program.py")).write_text("changed")
    result, calls = submit(cluster, "--", "true", ok=False)
    assert "dirty checkout" in result.stderr
    assert not calls


def test_bench_array_plan_matches_archived_wheel_identity(cluster, monkeypatch, tmp_path):
    from egolearn import run_record
    from egolearn.commands import bench

    checkout, env = cluster
    (checkout / "pyproject.toml").write_text('[project]\nname="egolearn"\n')
    commit(checkout)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("APPTAINER_BIND", raising=False)
    monkeypatch.chdir(checkout)
    # This fixture supplies the archived launcher tree; emulate its editable install.
    monkeypatch.setattr(bench, "source_checkout", checkout.resolve)
    spec = tmp_path / "suite.yaml"
    spec.write_text(
        "version: 1\nname: toy\nentry: train\nconfig: toy_streaming\nseeds: [1, 2]\n"
        "overrides: ['++driver.max_steps=1']\nmetrics: {train/steps: max}\n"
    )
    manifest = bench.plan_suite(spec, Path(env["EGOLEARN_ARTIFACTS"]) / "suite", [], submit=True)
    assert bench.run_suite(manifest, True, ["--time=00:01:00"]) == 0
    calls = [json.loads(line) for line in Path(env["TEST_SUBMIT_LOG"]).read_text().splitlines()]
    job = calls[-1]["env"]
    assert "--array=0-1" in calls[-1]["args"]
    assert "--no-requeue" in calls[-1]["args"]
    for name in ("EGOLEARN_RUN_DIR", "EGOLEARN_CODE_COMMIT", "EGOLEARN_IMAGE"):
        monkeypatch.setenv(name, job[name])
    monkeypatch.setattr(run_record, "__file__", str(tmp_path / "wheel/lib/egolearn/run_record.py"))
    assert run_record.source_identity() == json.loads(manifest.read_text())["provenance"]


@pytest.mark.parametrize("change", ["lock", "image", "code"])
def test_environment_stamp_covers_dependencies_image_and_code(cluster, change):
    checkout, _ = cluster
    _, calls = submit(cluster, "--", "true")
    ready(calls[-1])
    old = calls[-1]["env"]["EGOLEARN_EXPECTED_STAMP"]
    if change == "lock":
        (checkout / "uv.lock").write_text("new lock")
    elif change == "image":
        (checkout / "containers/IMAGE").write_text("ghcr.io/lab/base@sha256:" + "a" * 64)
    else:
        (checkout / "program.py").write_text("new code")
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


@pytest.mark.parametrize("cached,mode", [(False, "dry_run"), (True, "commands")])
def test_submission_needs_no_python_and_records_arguments(cluster, cached, mode):
    checkout, env = cluster
    if cached:
        _, calls = submit(cluster, "--", "true")
        ready(calls[-1])
    bindir = Path(env["TEST_SUBMIT_LOG"]).parent / "bin"
    for name in ("python", "python3"):
        executable = bindir / name
        executable.write_text("#!/bin/sh\necho 'Python ran on login node' >&2\nexit 99\n")
        executable.chmod(0o755)
    command = [
        "python",
        "program.py",
        "",
        'quotes"\\ café 🐣 $(id) `id`',
        "controls=" + "".join(chr(value) for value in range(1, 32)),
    ]
    if mode == "commands":
        commands = checkout.parent / "commands.txt"
        commands.write_text("python program.py 'quoted argument'\n")
        args = ["--commands", str(commands)]
    else:
        args = ["--dry-run", "--", *command]
    result, calls = submit(cluster, *args)
    assert "Python ran on login node" not in result.stderr
    job = Path(result.stdout.split("Run record: ", 1)[1].strip()).parent
    record = json.loads((job / "run.json").read_text())
    array = mode == "commands"
    assert record["command"] == ([] if array else command)
    assert record["commands_file"] == ("commands.sh" if array else None)
    assert record["dirty"] is False
    revision = run(["git", "rev-parse", "HEAD"], cwd=checkout, env=env).stdout.strip()
    assert record["commit"] == revision
    assert record["submitted"].endswith("Z")
    assert (job / "work/program.py").read_text() == "print('submitted version')\n"
    assert len(calls) == (0 if mode == "dry_run" else 1 if cached else 2)
    if mode == "dry_run":
        assert "afterok:PREP_JOB_ID" in result.stdout


def test_snapshot_preserves_unusual_names_and_rejects_links(cluster):
    checkout, env = cluster
    filename = "spaces and\nnewline.txt"
    (checkout / filename).write_text("fixture")
    commit(checkout)
    _, calls = submit(cluster, "--", "true")
    job = Path(calls[-1]["env"]["EGOLEARN_RUN_DIR"])
    assert (job / "work" / filename).read_text() == "fixture"
    (checkout / "external").symlink_to(checkout.parent)
    commit(checkout)
    result, calls = submit(cluster, "--", "true", ok=False)
    assert "unsupported archive entry" in result.stderr and not calls
    assert list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*")) == [job]


def test_failed_archive_removes_partial_snapshot(cluster):
    _, env = cluster
    fake = Path(env["TEST_SUBMIT_LOG"]).parent / "bin/tar"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    _, calls = submit(cluster, "--", "true", ok=False)
    assert not calls and not list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*"))


def test_non_utf8_arguments_fail_before_snapshot_creation(cluster):
    _, env = cluster
    result, calls = submit(cluster, "--", "true", os.fsdecode(b"invalid-\xff"), ok=False)
    assert "command arguments must be valid UTF-8" in result.stderr
    assert not calls and not list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*"))


def test_submodules_fail_before_snapshot_creation(cluster):
    checkout, env = cluster
    revision = run(["git", "rev-parse", "HEAD"], cwd=checkout, env=env).stdout.strip()
    (checkout / "vendor").mkdir()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"160000,{revision},vendor"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "add submodule"], cwd=checkout, check=True)
    result, calls = submit(cluster, "--", "true", ok=False)
    assert "submodules are not supported" in result.stderr
    assert not calls and not list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*"))


@pytest.mark.parametrize("name", ["HF_HOME", "EGOLEARN_JOB_ROOT"])
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


@pytest.mark.parametrize(
    ("gpu", "cpu_env", "threads"),
    [
        (False, {}, "1"),
        (
            True,
            {"OMP_NUM_THREADS": "2", "SLURM_CPUS_PER_TASK": "4", "SLURM_CPUS_ON_NODE": "8"},
            "2",
        ),
    ],
)
def test_run_uses_snapshot_and_readonly_bind(cluster, gpu, cpu_env, threads):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    call = calls[-1]
    ready(call)
    fake = Path(env["TEST_SUBMIT_LOG"]).parent / "bin/apptainer"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "print(json.dumps({'args':sys.argv[1:],'bind':os.environ['APPTAINER_BIND'],"
        "'cwd':os.getcwd(),'path':os.environ['APPTAINERENV_PREPEND_PATH'],"
        "'threads':os.environ['APPTAINERENV_OMP_NUM_THREADS']}))\n"
    )
    fake.chmod(0o755)
    child_env = {**call["env"], "SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "1" if gpu else "0"}
    child_env.update(cpu_env)
    child_env.pop("CUDA_VISIBLE_DEVICES", None)
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    result = run(
        ["bash", str(job / "code/scripts/slurm/run.sbatch"), "true"], cwd=checkout, env=child_env
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.splitlines()[-1])
    assert ("--nv" in data["args"]) is gpu
    assert f"{job}/code:{job}/code:ro" in data["bind"]
    destinations = [
        entry.split(":")[1] if ":" in entry else entry for entry in data["bind"].split(",")
    ]
    assert len(destinations) == len(set(destinations))
    assert f"{child_env['EGOLEARN_VENV']}:{child_env['EGOLEARN_VENV']}:ro" in data["bind"]
    assert data["cwd"] == str(job / "work")
    assert data["path"] == child_env["EGOLEARN_VENV"] + "/bin"
    assert data["threads"] == threads


def test_prep_pulls_on_cpu_and_syncs_in_container(cluster):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    fake = Path(env["TEST_SUBMIT_LOG"]).parent / "bin/apptainer"
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


@pytest.mark.parametrize("option", ["--export=NONE", "--wrap=true"])
def test_reserved_options_refused(cluster, option):
    result, calls = submit(cluster, option, "--", "true", ok=False)
    assert "managed by submit" in result.stderr
    assert not calls


def test_redundant_prep_does_not_wait_for_running_jobs(cluster):
    import fcntl

    checkout, _ = cluster
    _, calls = submit(cluster, "--", "true")
    ready(calls[-1])
    child_env = {**calls[-1]["env"], "SLURM_JOB_ID": "12"}
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    lock = Path(child_env["EGOLEARN_VENV"]) / ".egolearn-sync.lock"
    with lock.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        result = subprocess.run(
            ["bash", str(job / "code/scripts/slurm/prep.sbatch")],
            cwd=checkout,
            env=child_env,
            text=True,
            capture_output=True,
            timeout=10,
        )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("script", "config"),
    [("train", "projects/toy.yaml"), ("bench", "config.yaml")],
)
def test_compatibility_wrappers_are_executable_and_request_gpus(cluster, script, config):
    checkout, env = cluster
    wrapper = checkout / f"scripts/slurm/{script}.sbatch"
    assert os.access(wrapper, os.X_OK)
    args = [config, "seed=19"] if script == "train" else [config]
    result = run([str(wrapper), *args], cwd=checkout, env=env)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["TEST_SUBMIT_LOG"]).read_text().splitlines()]
    assert "--gres=gpu:1" in calls[-1]["args"]
    assert "--cpus-per-task=8" in calls[-1]["args"]
    assert any(arg.startswith("--time=") for arg in calls[-1]["args"])
    assert "uv" not in calls[-1]["args"]
    if script == "train":
        job = Path(calls[-1]["env"]["EGOLEARN_RUN_DIR"])
        assert json.loads((job / "run.json").read_text())["command"] == [
            "egl",
            "train",
            config,
            "seed=19",
        ]


@pytest.mark.parametrize("style", ["-ejoined", "env"])
def test_relative_stderr_path_is_resolved_before_chdir(cluster, style):
    checkout, env = cluster
    if style == "env":
        env["SBATCH_ERROR"] = "logs/%j.err"
        args = []
    else:
        args = ["-elogs/%j.err"]
    _, calls = submit(cluster, *args, "--", "true")
    assert f"--error={checkout}/logs/%j.err" in calls[-1]["args"]
    assert (checkout / "logs").is_dir()


def test_prep_drops_inherited_slurm_settings_and_exports_required_environment(cluster):
    _, env = cluster
    for key in (
        "GPUS_PER_TASK",
        "CPUS_PER_GPU",
        "MEM_PER_GPU",
        "TRES_PER_TASK",
        "QOS",
        "CONSTRAINT",
        "RESERVATION",
        "EXPORT",
    ):
        env["SBATCH_" + key] = "fixture"
    env["SBATCH_ACCOUNT"] = "lab"
    _, calls = submit(cluster, "--", "true")
    assert {key for key in calls[0]["env"] if key.startswith("SBATCH_")} == {"SBATCH_ACCOUNT"}
    assert "--export=ALL" in calls[-1]["args"]


def test_failed_validation_does_not_leave_snapshot(cluster):
    _, env = cluster
    _, calls = submit(cluster, "--commands", "/missing/commands", ok=False)
    assert not calls and not list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*"))


def test_uncertain_submission_preserves_code_and_prep(cluster):
    _, env = cluster
    bindir = Path(env["TEST_SUBMIT_LOG"]).parent / "bin"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        sbatch.read_text().replace(
            "print('12345')",
            "if any(a.endswith('/run.sbatch') for a in sys.argv): sys.exit(2)\nprint('12345')",
        )
    )
    cancel = bindir / "scancel"
    cancel.write_text('#!/bin/sh\nprintf \'%s\' "$1" > "$CANCEL_LOG"\n')
    cancel.chmod(0o755)
    env["CANCEL_LOG"] = str(bindir.parent / "cancelled")
    _, calls = submit(cluster, "--", "true", ok=False)
    assert len(calls) == 2
    assert not Path(env["CANCEL_LOG"]).exists()
    assert list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*/code/program.py"))


def test_unignored_logs_cannot_dirty_the_next_submission(cluster):
    result, calls = submit(cluster, "-o", "unignored/job-%j.log", "--", "true", ok=False)
    assert "must be git-ignored" in result.stderr
    assert not calls


@pytest.mark.parametrize("script,per_gpu", [("train", False), ("bench", True)])
def test_wrapper_resource_environment_overrides_defaults(cluster, script, per_gpu):
    checkout, env = cluster
    env.update(SBATCH_TIMELIMIT="01:00:00", SBATCH_GRES="gpu:typed:2", SBATCH_MEM_PER_GPU="32G")
    env["SBATCH_CPUS_PER_TASK"] = "16"
    if per_gpu:
        env["SBATCH_CPUS_PER_GPU"] = "8"
    args = ["toy_streaming"] if script == "train" else ["config.yaml"]
    result = run([str(checkout / f"scripts/slurm/{script}.sbatch"), *args], cwd=checkout, env=env)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["TEST_SUBMIT_LOG"]).read_text().splitlines()]
    assert not any(arg.startswith(("--time=", "--gres=", "--mem=")) for arg in calls[-1]["args"])
    assert calls[-1]["env"]["SBATCH_GRES"] == "gpu:typed:2"
    assert ("--cpus-per-task=16" in calls[-1]["args"]) is not per_gpu
    assert "--cpus-per-task=8" not in calls[-1]["args"]


@pytest.mark.parametrize(
    ("script", "resource_env", "cpu_default", "gpu_default"),
    [
        ("train", {"SBATCH_TRES_PER_TASK": "cpu=16"}, False, True),
        ("bench", {"SBATCH_TRES_PER_TASK": "gres/gpu=1"}, True, False),
        ("train", {"SBATCH_TRES_PER_TASK": "license/ansys=1,gres/gpu:a100=2,cpu=16"}, False, False),
        ("bench", {"SBATCH_TRES_PER_TASK": "license/ansys=1"}, True, True),
    ],
)
def test_wrapper_tres_overrides_only_matching_resources(
    cluster, script, resource_env, cpu_default, gpu_default
):
    checkout, env = cluster
    env.update(resource_env)
    args = ["toy_streaming"] if script == "train" else ["config.yaml"]
    result = run([str(checkout / f"scripts/slurm/{script}.sbatch"), *args], cwd=checkout, env=env)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["TEST_SUBMIT_LOG"]).read_text().splitlines()]
    assert ("--cpus-per-task=8" in calls[-1]["args"]) is cpu_default
    assert ("--gres=gpu:1" in calls[-1]["args"]) is gpu_default


@pytest.mark.parametrize("source", ["/dev/null", "/data/task-%a/input.txt"])
def test_slurm_input_devices_and_patterns_pass_through(cluster, source):
    _, calls = submit(cluster, "--input", source, "--", "true")
    assert f"--input={source}" in calls[-1]["args"]


def test_prep_fails_promptly_on_unsupported_file_locking(cluster):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    child_env = {**calls[0]["env"], "SLURM_JOB_ID": "12"}
    image = Path(child_env["SIF"])
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    fake = Path(env["TEST_SUBMIT_LOG"]).parent / "bin/flock"
    fake.write_text("#!/bin/sh\nexit 71\n")
    fake.chmod(0o755)
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    result = subprocess.run(
        ["bash", str(job / "code/scripts/slurm/prep.sbatch")],
        cwd=checkout,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "cannot lock" in result.stderr


def test_accepted_job_survives_failure_to_write_job_id(cluster):
    _, env = cluster
    sbatch = Path(env["TEST_SUBMIT_LOG"]).parent / "bin/sbatch"
    sbatch.write_text(
        sbatch.read_text().replace(
            "print('12345')",
            "if any(a.endswith('/run.sbatch') for a in sys.argv):\n"
            " os.mkdir(os.path.join(os.environ['EGOLEARN_RUN_DIR'],'job-id'))\nprint('12345')",
        )
    )
    result, calls = submit(cluster, "--", "true")
    assert "was accepted; could not write" in result.stderr
    job = Path(calls[-1]["env"]["EGOLEARN_RUN_DIR"])
    assert (job / "code/program.py").is_file()


def test_uncertain_prep_submission_preserves_its_code(cluster):
    _, env = cluster
    sbatch = Path(env["TEST_SUBMIT_LOG"]).parent / "bin/sbatch"
    sbatch.write_text(sbatch.read_text().replace("print('12345')", "sys.exit(1)"))
    result, calls = submit(cluster, "--", "true", ok=False)
    assert len(calls) == 1
    assert "Submission outcome uncertain" in result.stderr
    assert list(Path(env["EGOLEARN_JOB_ROOT"]).glob("run-*/code/scripts/slurm/prep.sbatch"))


@pytest.mark.parametrize(
    ("status", "opt_in", "requeued"),
    [
        (75, True, True),
        (75, False, False),
        (1, True, False),
        (0, True, False),
    ],
)
def test_run_requeues_only_explicit_checkpointed_pause(cluster, status, opt_in, requeued):
    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    call = calls[-1]
    ready(call)
    bindir = Path(env["TEST_SUBMIT_LOG"]).parent / "bin"
    (bindir / "apptainer").write_text(f"#!/bin/bash\nexit {status}\n")
    (bindir / "apptainer").chmod(0o755)
    log = checkout.parent / "requeues.txt"
    (bindir / "scontrol").write_text(
        f"#!/bin/bash\nprintf '%s\\n' \"$*\" > {shlex.quote(str(log))}\n"
    )
    (bindir / "scontrol").chmod(0o755)
    child_env = {
        **call["env"],
        "SLURM_JOB_ID": "912",
        "EGOLEARN_REQUEUE_ON_PAUSE": str(int(opt_in)),
    }
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    result = run(
        ["bash", str(job / "code/scripts/slurm/run.sbatch"), "true"], cwd=checkout, env=child_env
    )
    assert result.returncode == (0 if requeued else status), result.stderr
    assert log.exists() == requeued
    if requeued:
        assert log.read_text().strip() == "requeue 912"


@pytest.mark.parametrize("late_signal", [False, True])
def test_run_forwards_pause_signal_and_preserves_collected_exit_status(cluster, late_signal):
    import signal
    import time

    checkout, env = cluster
    _, calls = submit(cluster, "--", "true")
    call = calls[-1]
    ready(call)
    bindir = Path(env["TEST_SUBMIT_LOG"]).parent / "bin"
    marker, requeues = checkout.parent / "ready", checkout.parent / "requeues"
    (bindir / "apptainer").write_text(
        f"#!{sys.executable}\nimport os,signal,time\nfrom pathlib import Path\n"
        "def pause(*args):\n"
        " signal.signal(signal.SIGUSR1,signal.SIG_IGN)\n"
        + (" os.kill(os.getppid(),signal.SIGUSR1)\n" if late_signal else "")
        + " raise SystemExit(75)\n"
        "signal.signal(signal.SIGUSR1,pause)\n"
        f"Path({str(marker)!r}).touch()\n"
        "while True: time.sleep(.01)\n"
    )
    (bindir / "apptainer").chmod(0o755)
    (bindir / "scontrol").write_text(
        f"#!/bin/bash\nprintf '%s\\n' \"$*\" > {shlex.quote(str(requeues))}\n"
    )
    (bindir / "scontrol").chmod(0o755)
    child_env = {**call["env"], "SLURM_JOB_ID": "912", "EGOLEARN_REQUEUE_ON_PAUSE": "1"}
    job = Path(child_env["EGOLEARN_RUN_DIR"])
    process = subprocess.Popen(
        ["bash", str(job / "code/scripts/slurm/run.sbatch"), "true"],
        cwd=checkout,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        process.send_signal(signal.SIGUSR1)
        _, errors = process.communicate(timeout=15)
        assert process.returncode == 0, errors
        assert requeues.read_text().strip() == "requeue 912"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
