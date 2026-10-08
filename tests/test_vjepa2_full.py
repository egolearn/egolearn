"""Causal clip admission, matched budgets and exact full-run continuation."""

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.index import index_stream
from egolearn.data.sources.base import DenseFrameChannel, Session, Stream
from egolearn.evaluation import load_model
from egolearn.methods.vjepa2.admission import DenseClipEmitter
from egolearn.methods.vjepa2.full_train import (
    PREFLIGHT_STOP,
    UpdateSchedule,
    preflight_batch,
    prepare,
    run,
)

CONFIGS = Path(__file__).resolve().parents[1] / "projects/vjepa2/configs"


@pytest.fixture
def frames(tmp_path):
    root = tmp_path / "frames"
    for sid, n in (("child/session_1", 19), ("child/session_2", 22)):
        directory = root / sid
        directory.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (68, 64), (i * 11 % 256, i * 7 % 256, 170)).save(
                directory / f"frame_{i + 1:08d}.jpg"
            )
    return root


def config(root, frames, mode):
    overrides = [
        f"stream.root_dir={frames}",
        "device=cpu",
        "loader.workers=0",
        "driver.prefetch=false",
        "wandb.mode=disabled",
        "run_control.batch_size=4",
        "run_control.advance_clips=2",
        "driver.batch.clip_len=4",
        "views.clip.size=64",
        "model.encoders.video.dim=32",
        "model.encoders.video.depth=2",
        "model.encoders.video.heads=2",
        "model.predictors.pred.dim=32",
        "model.predictors.pred.depth=2",
        "model.predictors.pred.heads=2",
        "solver.schedule.warmup_min_steps=2",
        "hooks.snapshot_every=2",
        "hooks.log_every=1",
        "preflight=true",
        f"run_dir={root}",
    ]
    if mode == "streaming":
        overrides.append("driver.warmup.events_emitted=2")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = OmegaConf.to_container(
            compose(config_name=f"full_babyview_{mode}", overrides=overrides), resolve=True
        )
    assert isinstance(cfg, dict)
    cfg["losses"][0]["precision"] = "float32"
    return cfg


@pytest.mark.parametrize("frame_stride", [1, 2])
def test_emitter_matches_index_and_never_emits_future_frames(frame_stride):
    class Source(Stream):
        def sessions(self):
            # Includes a forward gap, duplicate timestamp and discarded tails.
            for sid, ticks in (
                ("a", [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 13, 14, 15, 16]),
                ("b", list(range(13))),
            ):
                channel = DenseFrameChannel(
                    "video", sid, [t / 5 for t in ticks], [(sid, i) for i in range(len(ticks))]
                )
                yield Session(sid, {"video": channel})

    source = Source()
    emitter = DenseClipEmitter(clip_len=4, clip_stride=frame_stride)
    actual = []
    for session in source.sessions():
        emitter.begin_session(session)
        for seen, obs in enumerate(session.observations(), start=1):
            emitter.ingest(obs)
            for event in emitter.poll():
                ref = event[K.VIDEO_REF]
                assert ref.locator[2] == seen
                assert ref.t1 == obs.t
                actual.append(ref)
        emitter.end_session()
        assert not emitter.poll()
    expected, _ = index_stream(source, clip_len=4, clip_stride=frame_stride, max_frame_gap=0.3)
    assert actual == expected
    assert all(r.session == r.locator[0] for r in actual)


def test_schedule_endpoints_and_raw_progress_independence():
    schedule = UpdateSchedule(200, 0.05, 2, 0)
    a, b = Context(device="cpu"), Context(device="cpu")
    a.clock.total_observations, b.clock.total_observations = 200, 12345
    for step in (0, 9, 10, 199):
        a.clock.step = b.clock.step = step
        a.clock.admitted, b.clock.admitted = step, step * 23 + 7
        assert schedule.multiplier(a) == schedule.multiplier(b)
    a.clock.step = 0
    assert schedule.multiplier(a) == 0.1
    a.clock.step = 9
    assert schedule.multiplier(a) == 1
    a.clock.step = 10
    assert schedule.multiplier(a) == 1
    a.clock.step = 199
    assert 0 < schedule.multiplier(a) < 0.001
    with pytest.raises(ValueError, match="schedule differs"):
        schedule.load_state_dict(UpdateSchedule(201, 0.05, 2, 0).state_dict())


@pytest.mark.parametrize(
    ("mode", "workers"), [("iid", 0), ("streaming", 0), ("iid", 2), ("streaming", 2)]
)
def test_full_run_exact_resume_baseline_export_and_metrics(tmp_path, frames, mode, workers):
    reference = config(tmp_path / f"reference_{mode}", frames, mode)
    interrupted = config(tmp_path / f"split_{mode}", frames, mode)
    for cfg in (reference, interrupted):
        cfg["loader"]["workers"] = workers
        cfg["driver"]["prefetch"] = workers > 0 and mode == "iid"
    assert run(reference) == 0
    interrupted["run_control"]["stop_after_steps"] = 1 if workers else 2
    assert run(interrupted) == PREFLIGHT_STOP
    split_root = Path(interrupted["run_dir"])
    assert not (split_root / "final.pth").exists()
    initial = (split_root / "random.pth").read_bytes()
    partial = torch.load(split_root / "snapshot_latest.pth", weights_only=False)
    assert partial["schedule"]["updates"] == 4
    interrupted["run_control"]["stop_after_steps"] = 0
    assert run(interrupted) == 0
    assert (split_root / "random.pth").read_bytes() == initial
    final = torch.load(split_root / "final.pth", weights_only=True)
    expected = torch.load(Path(reference["run_dir"]) / "final.pth", weights_only=True)
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(final["model"][name], tensor, rtol=0, atol=0)
    for key, state in expected["optimizer"]["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(
                final["optimizer"]["state"][key][name], tensor, rtol=0, atol=0
            )
    load_model(split_root / "final.pth")  # restricted load also works for streaming exports
    status = json.loads((split_root / "status.json").read_text())
    assert status["state"] == "complete" and status["step"] == 4
    if mode == "streaming":
        assert status["admitted"] == 41 and status["events_emitted"] == 9
    rows = [json.loads(s) for s in (split_root / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert all(r["frames_per_s"] == 4 * r["clips_per_s"] > 0 for r in rows)
    assert all(math.isfinite(v) for row in rows for v in row.values())
    assert [r["lr"] for r in rows] == [
        json.loads(s)["lr"]
        for s in (Path(reference["run_dir"]) / "metrics.jsonl").read_text().splitlines()
    ]


def test_modes_have_identical_initial_models_and_budgets(tmp_path, frames):
    states, budgets = [], []
    for mode in ("iid", "streaming"):
        pipe, _, budget, _ = prepare(config(tmp_path / mode, frames, mode))
        pipe.losses[0].initialize_target(pipe.model)
        states.append(deepcopy(pipe.model.state_dict()))
        budgets.append(budget)
        pipe.loader.close()
    assert budgets[0] == budgets[1]
    for name, tensor in states[0].items():
        torch.testing.assert_close(states[1][name], tensor, rtol=0, atol=0)


def test_streaming_preflight_is_actual_batch_and_tail_does_not_repeat(tmp_path, frames):
    cfg = config(tmp_path / "run", frames, "streaming")
    pipe, _, budget, refs = prepare(cfg)
    expected = preflight_batch(pipe, refs)
    assert pipe.ctx.clock.admitted == pipe.ctx.clock.compose_seq == 0
    pipe.driver._run_warmup()
    assert pipe.ctx.clock.events_emitted == 2
    assert pipe.driver._advance_once()
    actual = pipe.composer.compose(pipe.recipe, pipe.memories, {}, pipe.ctx, seq=0)
    torch.testing.assert_close(actual["clips"]["views"], expected["clips"]["views"], rtol=0, atol=0)
    assert [e[K.VIDEO_REF] for e in actual["clips"]["meta"]] == refs[:4]
    while pipe.driver._advance_once():
        pass
    assert pipe.ctx.clock.events_emitted == budget["clips"]
    assert pipe.ctx.clock.admitted == budget["frames"]
    assert not pipe.driver._advance_once()
    pipe.loader.close()


@pytest.mark.parametrize("change", ["frames", "clips", "batch", "emitter", "warmup", "transcript"])
def test_mismatched_recipes_and_mounts_fail(tmp_path, frames, change):
    cfg = config(tmp_path / change, frames, "streaming")
    if change in ("frames", "clips"):
        cfg["run_control"][f"expected_{change}"] = 1000
    elif change == "batch":
        cfg["recipe"]["slots"][0]["n"] = 3
    elif change == "emitter":
        cfg["admission"] = [{"_target_": "admission:frame_emitter"}]
    elif change == "warmup":
        cfg["driver"]["warmup"] = {"admitted": 2}
    else:
        cfg["stream"]["transcript_json"] = "not-read.json"
    with pytest.raises(ValueError):
        prepare(cfg)


def test_manifest_mismatch_and_skipped_update_fail(tmp_path, frames, monkeypatch):
    cfg = config(tmp_path / "run", frames, "streaming")
    cfg["run_control"]["stop_after_steps"] = 1
    assert run(cfg) == PREFLIGHT_STOP
    changed = deepcopy(cfg)
    changed["solver"]["groups"][0]["lr"] *= 2
    with pytest.raises(ValueError, match="manifest differs"):
        run(changed)
    cfg["record_json"] = None
    cfg["preflight"] = False
    cfg["run_control"]["stop_after_steps"] = 0
    from egolearn.methods.vjepa2.loss import VJEPA2

    monkeypatch.setattr(VJEPA2, "compute", lambda *args: {"l1": torch.tensor(float("nan"))})
    with pytest.raises(RuntimeError, match="skipped"):
        run(cfg)
    assert not (Path(cfg["run_dir"]) / "final.pth").exists()


def test_raw_frame_pseudo_clips_are_rejected(tmp_path, frames):
    cfg = config(tmp_path / "bad", frames, "streaming")
    cfg["driver"]["max_steps"] = 1
    cfg["admission"] = [{"_target_": "admission:frame_emitter"}]
    cfg["routes"] = {"RawFrame": "clips"}
    cfg["driver"]["advance"]["schema"] = "RawFrame"
    with pytest.raises(egolearn.pipeline.PipelineError, match="dense_clip"):
        egolearn.build(cfg)


def test_final_tail_drain_and_completed_run_are_idempotent(tmp_path, frames):
    # Eight complete clips, followed by two frames: no stale fourth update.
    for index in range(19, 23):
        (frames / "child/session_2" / f"frame_{index:08d}.jpg").unlink()
    cfg = config(tmp_path / "complete", frames, "streaming")
    assert run(cfg) == 0
    root = Path(cfg["run_dir"])
    status = json.loads((root / "status.json").read_text())
    assert (status["step"], status["admitted"], status["events_emitted"]) == (3, 37, 8)
    final = (root / "final.pth").read_bytes()
    assert run(cfg) == 0
    assert (root / "final.pth").read_bytes() == final
    assert json.loads((root / "status.json").read_text()) == status


def test_wandb_failure_after_fit_is_not_reported_as_success(tmp_path, frames, monkeypatch):
    import sys
    from types import SimpleNamespace

    from egolearn.methods.vjepa2.full_train import RunControl

    finished = []
    fake = SimpleNamespace(
        summary={},
        log=lambda *a, **kw: None,
        finish=lambda **kw: finished.append(kw.get("exit_code", 0)),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: fake))
    original = RunControl.on_step

    def invalidate_final_count(self, driver, logs):
        original(self, driver, logs)
        if driver.ctx.clock.step == self.budget["updates"]:
            driver.ctx.clock.admitted -= 1

    monkeypatch.setattr(RunControl, "on_step", invalidate_final_count)
    cfg = config(tmp_path / "failed", frames, "streaming")
    cfg["wandb"]["mode"] = "online"
    with pytest.raises(RuntimeError, match="counts differ"):
        run(cfg)
    assert finished == [1]
    assert fake.summary["training_state"] == "failed"
    assert not (Path(cfg["run_dir"]) / "final.pth").exists()


@pytest.mark.parametrize("trigger", ["time", "signal"])
def test_exit75_pause_resumes_exactly(tmp_path, frames, monkeypatch, trigger):
    import os
    import signal

    from egolearn.methods.vjepa2.full_train import PAUSED, RunControl

    cfg = config(tmp_path / trigger, frames, "streaming")
    reference = config(tmp_path / "reference", frames, "streaming")
    assert run(reference) == 0
    original = RunControl.on_step
    if trigger == "time":
        cfg["run_control"]["max_seconds"] = 1e-9
    else:

        def request_at_first_step(self, driver, logs):
            if driver.ctx.clock.step == 1:
                os.kill(os.getpid(), signal.SIGUSR1)
            original(self, driver, logs)

        monkeypatch.setattr(RunControl, "on_step", request_at_first_step)
    assert run(cfg) == PAUSED
    root = Path(cfg["run_dir"])
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "paused" and status["step"] == 1
    assert not (root / "final.pth").exists()
    monkeypatch.setattr(RunControl, "on_step", original)
    cfg["run_control"]["max_seconds"] = 0
    assert run(cfg) == 0
    final = torch.load(root / "final.pth", weights_only=True)["model"]
    expected = torch.load(Path(reference["run_dir"]) / "final.pth", weights_only=True)["model"]
    for name in final:
        torch.testing.assert_close(final[name], expected[name], rtol=0, atol=0)


def test_step_zero_snapshot_failure_recovers_verified_baseline(tmp_path, frames, monkeypatch):
    from egolearn.methods.vjepa2 import full_train

    cfg = config(tmp_path / "recovery", frames, "iid")
    original = full_train.save_snapshot

    def fail_initial_save(*args):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(full_train, "save_snapshot", fail_initial_save)
    with pytest.raises(OSError, match="storage"):
        run(cfg)
    root = Path(cfg["run_dir"])
    baseline = (root / "random.pth").read_bytes()
    assert json.loads((root / "status.json").read_text())["step"] == 0
    monkeypatch.setattr(full_train, "save_snapshot", original)
    assert run(cfg) == 0
    assert (root / "random.pth").read_bytes() == baseline
    (root / "final.pth").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="export verification"):
        run(cfg)


def test_cluster_paths_schedule_and_missing_batches_fail_early(tmp_path, frames, monkeypatch):
    cfg = config(tmp_path / "run", frames, "streaming")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    relative = {**cfg, "run_dir": "relative"}
    with pytest.raises(ValueError, match="absolute run_dir"):
        run(relative)
    cfg["solver"]["schedule"]["constant"] = True
    with pytest.raises(ValueError, match="schedule supports"):
        prepare(cfg)
    del cfg["solver"]["schedule"]["constant"]
    pipe, _, _, _ = prepare(cfg)
    with pytest.raises(RuntimeError, match="batch is missing"):
        pipe.driver._train_steps()
    pipe.loader.close()
    # Pilot runner rejects streaming before writing the random baseline.
    from egolearn.methods.vjepa2.train import run as pilot_run

    cfg["driver"]["max_steps"] = 1
    with pytest.raises(ValueError, match="full_train"):
        pilot_run(cfg)
    assert not (Path(cfg["run_dir"]) / "random.pth").exists()
