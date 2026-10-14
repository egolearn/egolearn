"""Run records preserve failures and cannot present skipped nonfinite updates as success."""

import json
from types import SimpleNamespace

import pytest

from egolearn.run_record import RunRecord, atomic_json, record_run


def test_atomic_record_preserves_previous_json_on_serialization_failure(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"valid": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_metrics_are_identified_in_failure_record(tmp_path, value):
    path = tmp_path / "record.json"
    with pytest.raises(ValueError), record_run({"record_json": str(path)}, "eval") as record:
        assert record is not None
        record.value["metrics"] = {"good": 1.0, "bad": value}
    result = json.loads(path.read_text())
    assert result["status"] == "nonfinite"
    assert result["nonfinite_metrics"] == ["bad"]
    assert result["metrics"] == {"good": 1.0}
    assert result["wall_seconds"] >= 0


def test_collector_distinguishes_last_step_and_final_probes():
    record = RunRecord("train", {})
    assert record.value["seed"] == 1337
    driver = SimpleNamespace(
        ctx=SimpleNamespace(clock=SimpleNamespace(step=8)),
        skipped_steps=2,
        nonfinite_steps=0,
        hooks=[SimpleNamespace(last_results={"objects": {"top1": 0.8}})],
    )
    record.on_step(driver, {"loss": 3.0})
    driver.ctx.clock.step = 9
    record.on_end(driver)
    assert record.value["metrics"] == {
        "train/last_step/loss": 3.0,
        "train/last_step/step": 8,
        "train/steps": 9,
        "train/skipped_steps": 2,
        "probe/objects/top1": 0.8,
    }


@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_driver_skipped_nonfinite_updates_make_run_unpublishable(
    tmp_path, toy_config, monkeypatch, mode
):
    import torch

    import egolearn

    cfg = toy_config(tmp_path)
    cfg["stream"].update(n_sessions=1, blocks_per_session=1, block_len=8)
    cfg["driver"]["warmup"] = {}
    cfg["record_json"] = str(tmp_path / "record.json")
    if mode == "offline":
        for key in ("admission", "memories", "recipe", "routes", "label_stores"):
            cfg.pop(key, None)
        cfg["driver"] = {
            "mode": "offline",
            "epochs": 1,
            "batch": {"slot": "curr", "n": 4, "views": ["v1", "v2"]},
        }
        cfg["losses"] = cfg["losses"][:1]
    pipe = egolearn.build(cfg)
    for loss in pipe.losses:
        monkeypatch.setattr(loss, "compute", lambda *args: {"loss": torch.tensor(float("nan"))})
    with pytest.raises(ValueError, match="nonfinite"), record_run(cfg, "train") as record:
        pipe.driver.hooks.append(record)
        pipe.driver.fit()
    result = json.loads((tmp_path / "record.json").read_text())
    assert result["status"] == "nonfinite"
    assert result["nonfinite_steps"] > 0
    assert result["metrics"]["train/steps"] == 0
