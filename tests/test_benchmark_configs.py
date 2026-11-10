"""Benchmark entry points compose inherited project configs before building."""

import os
import runpy
import sys
from pathlib import Path

import pytest

import egolearn


@pytest.mark.parametrize(
    ("script", "config_name", "relative"),
    [
        ("bench_full_pipeline.py", "saycam_default", False),
        ("bench_dataloading.py", "saycam_default", True),
        ("bench_full_pipeline.py", "saycam_iid", True),
    ],
)
def test_benchmarks_compose_babycl_defaults(tmp_path, monkeypatch, script, config_name, relative):
    root = Path(__file__).resolve().parents[1]
    config = root / "projects/babycl/configs" / f"{config_name}.yaml"
    monkeypatch.chdir(tmp_path)
    config_arg = os.path.relpath(config, tmp_path) if relative else str(config)
    monkeypatch.setattr(sys, "argv", [script, "--config", config_arg])

    class ConfigCaptured(Exception):
        pass

    def capture(cfg):
        if config_name == "saycam_iid":
            assert cfg["driver"]["mode"] == "offline"
            assert (
                cfg["driver"]["_target_"] == "egolearn.methods.babycl.offline.BabyCLOfflineDriver"
            )
            assert cfg["driver"]["epochs"] == 1
        else:
            assert cfg["driver"]["advance"]["schema"] == "VisualSegment"
        assert cfg["model"]["encoders"]
        assert [loss["weight"] for loss in cfg["losses"]] == [0.5, 0.5, 0.5]
        assert cfg["solver"]["groups"][0]["lr"] > 0
        assert cfg["stream"]["_target_"] == "stream:saycam"
        assert "defaults" not in cfg
        if script == "bench_full_pipeline.py":
            assert cfg["driver"]["max_steps"] == 200
        else:
            assert cfg["hooks"] == {} and "probes" not in cfg
        raise ConfigCaptured

    monkeypatch.setattr(egolearn, "build", capture)
    with pytest.raises(ConfigCaptured):
        runpy.run_path(str(root / "scripts/bench" / script), run_name="__main__")
