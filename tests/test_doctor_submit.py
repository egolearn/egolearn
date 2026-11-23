"""Diagnostics are read-only; submission preserves the launcher's argument contract."""

import os
import subprocess

import pytest

from egolearn.commands import doctor, submit


def test_doctor_does_not_create_paths_or_print_environment_values(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EGOLEARN_DATA", str(tmp_path / "missing data"))
    monkeypatch.setenv("TEST_TOKEN", "synthetic-value-that-must-not-appear")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert doctor.main(["--skip-cuda", "--require-env", "TEST_TOKEN"]) == 0
    output = capsys.readouterr().out
    assert "TEST_TOKEN: set" in output
    assert "synthetic-value" not in output
    assert "warn: data-root" in output
    assert not list(tmp_path.iterdir())
    monkeypatch.delenv("TEST_TOKEN")
    assert doctor.main(["--skip-cuda", "--require-env", "TEST_TOKEN"]) == 1


def test_cluster_paths_resolve_symlinks_and_nonexistent_suffixes(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    assert doctor.storage_check("cache", str(alias / "new/cache"), home).status == "fail"
    assert doctor.storage_check("cache", "relative", home).status == "fail"
    assert doctor.storage_check("cache", str(tmp_path / "scratch/new"), home).status == "ok"


def test_doctor_reports_missing_safetensors(monkeypatch):
    def version(package):
        if package == "safetensors":
            raise doctor.importlib.metadata.PackageNotFoundError(package)
        return "1.0"

    monkeypatch.setattr(doctor.importlib.metadata, "version", version)
    checks = doctor.inspect_environment(extras=("train",), skip_cuda=True)
    train = next(check for check in checks if check.name == "extra:train")
    assert train.status == "fail"
    assert train.detail == "missing safetensors"


def test_cuda_query_failures_are_bounded_and_reported(monkeypatch):
    def timed_out(*args, **kwargs):
        assert kwargs["timeout"] == 15
        raise subprocess.TimeoutExpired(args[0], 15)

    monkeypatch.setattr(subprocess, "run", timed_out)
    assert doctor.check_cuda(False).status == "warn"
    assert doctor.check_cuda(True).status == "fail"


def test_cluster_checks_report_all_missing_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for names in doctor.cluster_variables().values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    checks = doctor.inspect_environment(cluster=True, skip_cuda=True)
    missing = {c.name for c in checks if c.status == "fail"}
    assert set(doctor.cluster_variables()["required"]) <= missing


@pytest.mark.parametrize("location", ["login", "compute", "container"])
def test_cluster_tools_match_the_execution_location(tmp_path, monkeypatch, location):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for names in doctor.cluster_variables().values():
        for name in names:
            monkeypatch.setenv(name, str(tmp_path / "scratch" / name))
    for name in ("SLURM_JOB_ID", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        monkeypatch.delenv(name, raising=False)
    if location != "login":
        monkeypatch.setenv("SLURM_JOB_ID", "123")
    if location == "container":
        monkeypatch.setenv("APPTAINER_CONTAINER", "/image.sif")
        monkeypatch.setenv("HOME", os.environ["APPTAINER_HOME"])
        monkeypatch.setenv("TMPDIR", os.environ["APPTAINER_HOME"] + "/tmp")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    failures = {
        check.name
        for check in doctor.inspect_environment(cluster=True, skip_cuda=True)
        if check.status == "fail"
    }
    assert (
        failures
        == {
            "login": {"bash", "git", "sbatch"},
            "compute": {"apptainer"},
            "container": set(),
        }[location]
    )


@pytest.mark.parametrize(
    "in_allocation,missing",
    [
        (False, None),
        (True, "uv"),
    ],
)
def test_submission_doctor_checks_tools_on_the_planning_host(
    tmp_path, monkeypatch, capsys, in_allocation, missing
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for names in doctor.cluster_variables().values():
        for name in names:
            monkeypatch.setenv(name, str(tmp_path / "scratch" / name))
    for name in ("SLURM_JOB_ID", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        monkeypatch.delenv(name, raising=False)
    if in_allocation:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
    binaries = {"bash", "git", "sbatch", "iconv", "tar", "uv"}
    monkeypatch.setattr(
        doctor.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in binaries and name != missing else None,
    )

    assert doctor.main(["--submit", "--skip-cuda"]) == int(missing is not None)
    output = capsys.readouterr().out
    for binary in binaries:
        expected = f"fail: {binary}: not on PATH" if binary == missing else f"ok: {binary}: on PATH"
        assert expected in output
    assert "apptainer:" not in output
    # --submit also checks cluster storage, even outside a Slurm allocation.
    assert "ok: SIF_DIR:" in output
    assert str(tmp_path) not in output
    assert not list(tmp_path.iterdir())


def test_submit_from_installed_package_preserves_argv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="egolearn"\n')
    script = tmp_path / "scripts/slurm/submit"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n")
    calls = []
    monkeypatch.setattr(os, "execvp", lambda *args: calls.append(args))
    arguments = ["--job-name=a b", "--", "python", "file with spaces.py", "a=$(literal)"]
    assert submit.main(arguments) == 0
    assert calls == [("bash", ["bash", str(script), *arguments])]
