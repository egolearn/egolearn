"""`init_from`: start a run from another run's weights, by name, without resuming it."""

from __future__ import annotations

import copy

import pytest
import torch

import egolearn
from egolearn.utils.snapshot import save_snapshot


def _cfg(tmp_path, dim: int = 64):
    return {
        "run_dir": str(tmp_path / "run"),
        "seed": 7,
        "device": "cpu",
        "stream": {
            "_target_": "stream:toy",
            "n_sessions": 1,
            "blocks_per_session": 4,
            "block_len": 40,
            "img_size": 64,
        },
        "views": {
            "v1": {"name": "simclr_strong", "size": 64, "view_idx": 0},
            "v2": {"name": "simclr_strong", "size": 64, "view_idx": 1},
        },
        "label_stores": ["fixed"],
        "admission": [
            {"_target_": "admission:frame_emitter", "subsample": 1},
            {
                "_target_": "admission:fixed_window_labeler",
                "class_length": 40,
                "label_store": "fixed",
            },
        ],
        "memories": {
            "frames": {
                "tiers": {
                    "short": {"capacity": 30, "update": "fifo"},
                    "long": {"capacity": 60, "update": "reservoir"},
                },
                "admit_to": ["short", "long"],
                "overflow": {},
            }
        },
        "routes": {"RawFrame": "frames"},
        "recipe": {
            "slots": [
                {
                    "name": "curr",
                    "source": {"kind": "tier", "memory": "frames", "tier": "short"},
                    "n": 4,
                    "unit": "frame",
                    "views": ["v1", "v2"],
                    "labels_from": "fixed",
                    "strict": True,
                },
            ]
        },
        "model": {
            "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": dim}},
            "heads": {
                "proj": {"encoder": "vision", "_target_": "head:mlp", "sizes": [dim, dim, 32]},
                "cls": {"encoder": "vision", "_target_": "head:mlp", "sizes": [dim, dim, 8]},
            },
        },
        "losses": [
            {
                "_target_": "loss:ntxent",
                "name": "inst",
                "slots": ["curr"],
                "head": "proj",
                "temperature": 0.1,
                "weight": 1.0,
            },
            {
                "_target_": "loss:tcl",
                "name": "temp",
                "slots": ["curr"],
                "head": "cls",
                "num_classes": 8,
                "weight": 1.0,
            },
        ],
        "solver": {
            "opt": "sgd",
            "groups": [{"match": ".*", "lr": 0.05, "weight_decay": 0.0}],
            "schedule": {"constant": True},
        },
        "driver": {
            "advance": {"kind": "observations", "n": 8},
            "rate": {"steps_per_admit": 1, "resample": True},
            "warmup": {"admitted": 20},
            "max_steps": 2,
        },
        "hooks": {"log_every": 1},
    }


def _sub(state: dict, prefix: str) -> dict:
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture
def trained(tmp_path):
    """A short run whose weights differ from the seed's initialization, saved as
    both a plain checkpoint and a full snapshot."""
    pipe = egolearn.build(_cfg(tmp_path / "src"))
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


def test_include_copies_only_the_named_encoder(tmp_path, trained):
    state, ckpt, _ = trained
    cfg = _cfg(tmp_path / "dst")
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


def test_default_copies_everything_and_snapshots_are_accepted(tmp_path, trained):
    state, _, snap = trained
    cfg = _cfg(tmp_path / "dst")
    cfg["init_from"] = {"checkpoint": str(snap)}
    pipe = egolearn.build(cfg)
    assert _equal(pipe.model.state_dict(), state)
    assert pipe.init_from["step"] == 2  # read from the snapshot's clock
    # a fresh run: nothing of the source's optimizer/clock state came along
    assert pipe.ctx.clock.step == 0
    assert not pipe.driver.optimizer.state


def test_initialized_run_is_deterministic(tmp_path, trained):
    _, ckpt, _ = trained
    cfg = _cfg(tmp_path / "dst")
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
def test_strict_mode_rejects_stale_or_partial_selections(tmp_path, trained, init_from, match):
    _, ckpt, _ = trained
    cfg = _cfg(tmp_path / "dst")
    init_from = dict(init_from)
    if init_from["checkpoint"] is None:
        init_from["checkpoint"] = str(ckpt)
    else:
        init_from["checkpoint"] = str(tmp_path / init_from["checkpoint"])
    cfg["init_from"] = init_from
    with pytest.raises(egolearn.PipelineError, match=match):
        egolearn.build(cfg)


def test_shape_mismatch_fails_strict_and_is_skipped_otherwise(tmp_path, trained):
    _, ckpt, _ = trained
    cfg = _cfg(tmp_path / "dst", dim=32)  # a different architecture
    cfg["init_from"] = {"checkpoint": str(ckpt), "include": ["encoders.vision."]}
    with pytest.raises(egolearn.PipelineError, match="shape mismatch"):
        egolearn.build(cfg)
    # non-strict: the first conv (3->16) still matches; the rest is skipped
    cfg["init_from"]["strict"] = False
    pipe = egolearn.build(cfg)
    assert pipe.init_from["tensors"] >= 1
    assert pipe.init_from["skipped"]


def test_strict_mode_rejects_a_checkpoint_missing_part_of_the_encoder(tmp_path, trained):
    state, _, _ = trained
    partial = {k: v for k, v in state.items() if k != "encoders.vision.net.0.bias"}
    ckpt = tmp_path / "partial.pth"
    torch.save({"model": partial, "step": 2}, ckpt)
    cfg = _cfg(tmp_path / "dst")
    cfg["init_from"] = {"checkpoint": str(ckpt), "include": ["encoders.vision."]}
    with pytest.raises(egolearn.PipelineError, match="absent from"):
        egolearn.build(cfg)
    # an explicit exclude of the same tensor is a deliberate choice, not an error
    cfg["init_from"]["exclude"] = ["encoders.vision.net.0.bias"]
    pipe = egolearn.build(cfg)
    assert pipe.init_from["tensors"] == len(partial) - len(_sub(partial, "heads."))
