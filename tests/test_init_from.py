"""`init_from`: start a run from another run's weights, by name, without resuming it."""

from __future__ import annotations

import copy
import subprocess
import sys
from pickle import UnpicklingError

import pytest
import torch
from omegaconf import OmegaConf

import egolearn
from egolearn.utils.snapshot import load_snapshot, save_snapshot


def test_train_rejects_initialization_and_resume_before_building(tmp_path):
    # Deliberately omit stream/model config and use a nonexistent checkpoint:
    # the incompatible modes must fail before any pipeline or checkpoint I/O.
    OmegaConf.save(
        {
            "init_from": {"checkpoint": "missing.pth"},
            "resume": "auto",
            "dry_run": True,
        },
        tmp_path / "conflict.yaml",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "egolearn.train",
            "--config-path",
            str(tmp_path),
            "--config-name",
            "conflict",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "init_from and resume are mutually exclusive" in result.stderr
    assert "init_from=null" in result.stderr


def _sub(state: dict, prefix: str) -> dict:
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture
def trained(toy_config, tmp_path):
    """A short run whose weights differ from the seed's initialization, saved as
    both a plain checkpoint and a full snapshot."""
    pipe = egolearn.build(toy_config(tmp_path / "src"))
    pipe.driver.fit()
    state = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    ckpt = tmp_path / "final.pth"
    torch.save(
        {
            "model": state,
            "step": pipe.ctx.clock.step,
            "export_contract": pipe.model.export_contract(),
        },
        ckpt,
    )
    snap = tmp_path / "snapshot.pth"
    save_snapshot(str(snap), pipe.driver)
    return state, ckpt, snap


def test_include_copies_only_the_named_encoder(toy_config, tmp_path, trained):
    state, ckpt, _ = trained
    cfg = toy_config(tmp_path / "dst")
    cfg["init_from"] = {"checkpoint": str(ckpt), "include": ["encoders.vision."]}
    pipe = egolearn.build(cfg)
    got = pipe.model.state_dict()
    assert _equal(_sub(got, "encoders.vision."), _sub(state, "encoders.vision."))
    # heads keep their fresh (seed-derived) initialization: the source run
    # trained them, so they must differ from the source's final weights
    assert not _equal(_sub(got, "heads."), _sub(state, "heads."))
    assert pipe.init_from["tensors"] == len(_sub(state, "encoders.vision."))
    assert pipe.init_from["step"] == 2
    assert "init_from" in pipe.summary()
    # provenance travels with the model so later checkpoints record their lineage
    assert pipe.model.construction_config["init_from"]["sha256"] == pipe.init_from["sha256"]


def test_string_prefixes_select_whole_components(toy_config, tmp_path, trained):
    state, checkpoint, _ = trained
    cfg = toy_config(tmp_path)
    cfg["init_from"] = {
        "checkpoint": str(checkpoint),
        "include": "encoders.vision.",
        "exclude": "encoders.vision.net.0.bias",
    }
    fresh = egolearn.build(toy_config(tmp_path)).model.state_dict()
    pipe = egolearn.build(cfg)
    for name, tensor in pipe.model.state_dict().items():
        selected = name.startswith("encoders.vision.") and name != "encoders.vision.net.0.bias"
        assert torch.equal(tensor, state[name] if selected else fresh[name]), name


@pytest.mark.parametrize(
    "payload,error", [(b"", EOFError), (b"invalid", UnpicklingError), (b"PK\x03\x04", RuntimeError)]
)
def test_corrupt_checkpoint_has_pipeline_context(toy_config, tmp_path, payload, error):
    checkpoint = tmp_path / "corrupt.pth"
    checkpoint.write_bytes(payload)
    cfg = toy_config(tmp_path)
    cfg["init_from"] = {"checkpoint": str(checkpoint)}
    with pytest.raises(egolearn.PipelineError, match="init_from:") as caught:
        egolearn.build(cfg)
    assert isinstance(caught.value.__cause__, error)


@pytest.mark.parametrize("initialized", [False, True])
def test_resumed_checkpoints_keep_live_config_and_lineage(
    toy_config, tmp_path, trained, initialized
):
    _, checkpoint, _ = trained
    cfg = toy_config(tmp_path / "initialized")
    if initialized:
        cfg["init_from"] = {"checkpoint": str(checkpoint)}
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    snapshot = str(tmp_path / "initialized.pth")
    save_snapshot(snapshot, pipe.driver)
    cfg = toy_config(tmp_path / "resumed")
    cfg["init_from"] = None
    cfg["driver"]["max_steps"] = 3
    cfg["hooks"]["checkpoint_every"] = 1
    # Explicit defaults keep the weights compatible while changing the live
    # construction metadata, including the imports needed to rebuild the model.
    cfg["model"]["heads"]["proj"]["field"] = "pooled"
    cfg["_imports_"] = ["egolearn.model.heads"]
    resumed = egolearn.build(cfg)
    expected = copy.deepcopy(resumed.model.construction_config)
    if initialized:
        expected["init_from"] = pipe.init_from
    load_snapshot(snapshot, resumed.driver)
    assert resumed.model.construction_config == expected
    resumed.driver.fit()
    final = torch.load(tmp_path / "resumed/run/final.pth", weights_only=True)
    assert final["step"] == 3
    assert final["model_config"] == expected
    resumed_snapshot = str(tmp_path / "resumed.pth")
    save_snapshot(resumed_snapshot, resumed.driver)
    assert torch.load(resumed_snapshot, weights_only=False)["model_config"] == expected


@pytest.mark.parametrize("missing_config", [False, True])
def test_resume_without_saved_model_config_keeps_live_config(
    toy_config, tmp_path, trained, missing_config
):
    _, _, snapshot = trained
    state = torch.load(snapshot, weights_only=False)
    if missing_config:
        del state["model_config"]
    else:
        state["model_config"] = None
    torch.save(state, snapshot)
    resumed = egolearn.build(toy_config(tmp_path / "resumed"))
    expected = copy.deepcopy(resumed.model.construction_config)
    load_snapshot(str(snapshot), resumed.driver)
    assert resumed.model.construction_config == expected


def test_default_copies_everything_and_snapshots_are_accepted(toy_config, tmp_path, trained):
    state, _, snap = trained
    cfg = toy_config(tmp_path / "dst")
    cfg["init_from"] = {"checkpoint": str(snap)}
    pipe = egolearn.build(cfg)
    assert _equal(pipe.model.state_dict(), state)
    assert pipe.init_from["step"] == 2  # read from the snapshot's clock
    # a fresh run: nothing of the source's optimizer/clock state came along
    assert pipe.ctx.clock.step == 0
    assert not pipe.driver.optimizer.state


def test_initialized_run_is_deterministic(toy_config, tmp_path, trained):
    _, ckpt, _ = trained
    cfg = toy_config(tmp_path / "dst")
    cfg["init_from"] = {"checkpoint": str(ckpt), "include": ["encoders.vision."]}
    a = egolearn.build(copy.deepcopy(cfg))
    a.driver.fit()
    b = egolearn.build(copy.deepcopy(cfg))
    b.driver.fit()
    assert _equal(a.model.state_dict(), b.model.state_dict())


@pytest.mark.parametrize(
    "init_from,match",
    [
        ({"checkpoint": "MISSING", "include": ["encoders.vision."]}, "not found"),
        ({"checkpoint": None, "include": ["encoders.nope."]}, "matches nothing"),
        ({"checkpoint": None, "include": []}, "empty"),
    ],
)
def test_strict_mode_rejects_stale_or_partial_selections(
    toy_config, tmp_path, trained, init_from, match
):
    _, ckpt, _ = trained
    cfg = toy_config(tmp_path / "dst")
    init_from = dict(init_from)
    if init_from["checkpoint"] is None:
        init_from["checkpoint"] = str(ckpt)
    else:
        init_from["checkpoint"] = str(tmp_path / init_from["checkpoint"])
    cfg["init_from"] = init_from
    with pytest.raises(egolearn.PipelineError, match=match):
        egolearn.build(cfg)


def test_shape_mismatch_fails_strict_and_is_skipped_otherwise(toy_config, tmp_path, trained):
    _, ckpt, _ = trained
    cfg = toy_config(tmp_path / "dst", dim=32)  # a different architecture
    cfg["init_from"] = {"checkpoint": str(ckpt), "include": ["encoders.vision."]}
    with pytest.raises(egolearn.PipelineError, match="shape mismatch"):
        egolearn.build(cfg)
    # non-strict: the first conv (3->16) still matches; the rest is skipped
    cfg["init_from"]["strict"] = False
    pipe = egolearn.build(cfg)
    assert pipe.init_from["tensors"] >= 1
    assert pipe.init_from["skipped"]


def test_strict_mode_rejects_a_checkpoint_missing_part_of_the_encoder(
    toy_config, tmp_path, trained
):
    state, _, _ = trained
    partial = {k: v for k, v in state.items() if k != "encoders.vision.net.0.bias"}
    ckpt = tmp_path / "partial.pth"
    torch.save({"model": partial, "step": 2}, ckpt)
    cfg = toy_config(tmp_path / "dst")
    cfg["init_from"] = {"checkpoint": str(ckpt), "include": ["encoders.vision."]}
    with pytest.raises(egolearn.PipelineError, match="absent from"):
        egolearn.build(cfg)
    # an explicit exclude of the same tensor is a deliberate choice, not an error
    cfg["init_from"]["exclude"] = ["encoders.vision.net.0.bias"]
    pipe = egolearn.build(cfg)
    assert pipe.init_from["tensors"] == len(partial) - len(_sub(partial, "heads."))
