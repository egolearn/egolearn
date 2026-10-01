"""The installed CLI preserves Hydra composition and invocation semantics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egolearn import __version__


def run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).parent / "egolearn"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
        timeout=90,
    )


def test_help_version_and_invalid_command(tmp_path):
    for args in [(), ("--help",)]:
        result = run_cli(tmp_path, *args)
        assert result.returncode == 0, result.stderr
        assert "train" in result.stdout and "eval" in result.stdout
    result = run_cli(tmp_path, "--version")
    assert result.returncode == 0
    assert result.stdout.strip() == f"egolearn {__version__}"
    assert run_cli(tmp_path, "unknown").returncode == 2


def test_top_level_help_does_not_import_training_or_hydra(tmp_path):
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockOptionalImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'hydra', 'numpy'}:
            raise AssertionError(f'CLI help imported {fullname}')

sys.meta_path.insert(0, BlockOptionalImports())
from egolearn.cli import main
main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "train" in result.stdout


@pytest.mark.parametrize("command,config", [("train", "toy_streaming"), ("eval", "evaluate")])
def test_packaged_config_matches_existing_module(tmp_path, command, config):
    result = run_cli(tmp_path, command, config, "--cfg", "job", "seed=19")
    module = "egolearn.train" if command == "train" else "egolearn.evaluate"
    legacy = subprocess.run(
        [sys.executable, "-m", module, "--config-name", config, "--cfg", "job", "seed=19"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == legacy.stdout
    assert OmegaConf.create(result.stdout).seed == 19


@pytest.mark.parametrize("filename", ["example.yaml", "toy_streaming.yaml"])
def test_file_config_preserves_defaults_overrides_and_caller_directory(tmp_path, filename):
    project = tmp_path / "project with spaces"
    project.mkdir()
    (project / "shared.yaml").write_text("seed: 7\nmessage: initial\nrun_dir: runs/example\n")
    (project / filename).write_text("defaults: [shared, _self_]\n")
    result = run_cli(
        tmp_path,
        "train",
        f"project with spaces/{filename}",
        "--cfg",
        "job",
        "message='hello world'",
        "+extra=[1,2]",
        "~seed",
    )
    assert result.returncode == 0, result.stderr
    config = OmegaConf.create(result.stdout)
    assert config.message == "hello world"
    assert config.extra == [1, 2]
    assert "seed" not in config
    assert config.run_dir == "runs/example"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["project with spaces"]


def test_native_hydra_flags_and_eval_default(tmp_path):
    result = run_cli(tmp_path, "train", "--config-name=toy_streaming", "--cfg", "job")
    assert result.returncode == 0, result.stderr
    assert OmegaConf.create(result.stdout).seed == 7
    result = run_cli(tmp_path, "eval", "--cfg", "job", "checkpoint=model.pth")
    assert result.returncode == 0, result.stderr
    assert OmegaConf.create(result.stdout).checkpoint == "model.pth"
    result = run_cli(tmp_path, "train", "toy_streaming", "--help")
    assert result.returncode == 0, result.stderr
    assert "seed: 7" in result.stdout


@pytest.mark.parametrize("flag", ["--config-name=other", "-cn=other", "--config-dir=."])
def test_conflicting_config_selectors_are_rejected(tmp_path, flag):
    result = run_cli(tmp_path, "train", "toy_streaming", flag)
    assert result.returncode == 2
    assert "not both" in result.stderr


def test_missing_file_and_hydra_errors_fail(tmp_path):
    missing = run_cli(tmp_path, "train", "missing.yaml")
    assert missing.returncode == 2 and "config file not found" in missing.stderr
    invalid = run_cli(tmp_path, "train", "toy_streaming", "unknown_setting=1", "--cfg", "job")
    assert invalid.returncode != 0
    assert "unknown_setting" in invalid.stderr


def test_multirun_creates_jobs_in_callers_directory(tmp_path):
    result = run_cli(
        tmp_path,
        "train",
        "toy_streaming",
        "--multirun",
        "seed=1,2",
        "dry_run=true",
        "hydra.sweep.dir=sweep",
        "hydra.job.chdir=true",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for number, seed in enumerate((1, 2)):
        config = OmegaConf.load(tmp_path / "sweep" / str(number) / ".hydra/config.yaml")
        assert config.seed == seed
    assert result.stdout.count("[dry-run] pipeline valid.") == 2
