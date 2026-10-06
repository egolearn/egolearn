"""Diagnostics are read-only; submission preserves the launcher's argument contract."""

import os
import subprocess
import sys

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


def test_required_extras_missing_is_failure(monkeypatch):
    def absent(package):
        raise doctor.importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(doctor.importlib.metadata, "version", absent)
    checks = doctor.inspect_environment(extras=("train",), skip_cuda=True)
    assert next(c for c in checks if c.name == "extra:train").status == "fail"
    assert next(c for c in checks if c.name == "extra:video").status == "warn"


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


def test_submit_help_needs_no_checkout_but_execution_does(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert submit.main(["--help"]) == 0
    assert submit.main(["--", "true"]) == 1
    assert "checkout root" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    [
        ["data", "list", "--json"],
        ["doctor", "--skip-cuda"],
        ["submit", "--help"],
    ],
)
def test_operational_commands_do_not_import_torch(tmp_path, command):
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'hydra', 'numpy'}:
            raise AssertionError(f'operational command imported {fullname}')
sys.meta_path.insert(0, Block())
from egolearn.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, *command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_verify_and_table_do_not_import_training_dependencies(tmp_path):
    from egolearn.hub import write_checksums

    data = tmp_path / "data"
    data.mkdir()
    (data / "payload").write_text("synthetic")
    write_checksums(data)
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'hydra', 'numpy'}:
            raise AssertionError(f'command imported {fullname}')
sys.meta_path.insert(0, Block())
from egolearn.cli import main
assert main(['data', 'verify', sys.argv[1]]) == 0
try:
    main(['bench', 'table', sys.argv[2]])
except SystemExit as error:
    assert error.code == 1
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(data), str(tmp_path / "missing.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
