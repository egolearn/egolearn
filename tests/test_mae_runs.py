"""Full-pass budgets, interruption recovery, and nested BabyView sources."""

import json
import pickle
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.sources.saycam import BabyViewStream
from egolearn.methods.mae import MAE, MAEDecoder
from egolearn.methods.mae.train import PAUSED, PREFLIGHT_STOP, prepare, run
from egolearn.model.build import build_model


@pytest.fixture
def config(tmp_path):
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        composed = compose(config_name="toy_mae")
    cfg = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(cfg, dict)
    root = tmp_path / "frames" / "S_20200101"
    root.mkdir(parents=True)
    for i in range(9):
        Image.new("RGB", (32, 40), (i * 20, 80, 120)).save(root / f"S_{i * 0.2:.2f}.jpg")
    cfg["stream"] = {"_target_": "stream:saycam", "root_dir": str(root.parent)}
    cfg["driver"].pop("max_steps")
    cfg["run_control"] = {"batch_size": 4, "stop_after_steps": 0, "max_seconds": 0}
    cfg["run_dir"] = str(tmp_path / "run")
    cfg["hooks"] = {"log_every": 1}
    return cfg


def offline(cfg):
    for key in ("memories", "admission", "routes", "recipe"):
        cfg.pop(key)
    cfg["driver"] = {
        "mode": "offline",
        "batch": {"slot": "frames", "n": 4, "views": ["reconstruction"]},
    }


@pytest.mark.parametrize("iid", [False, True])
def test_full_pass_pause_resume_matches_uninterrupted(config, tmp_path, iid):
    if iid:
        offline(config)
    assert prepare(config)[2] == {"frames": 9, "updates": 6, "batch_size": 4}
    assert run(config) == 0
    reference = torch.load(tmp_path / "run" / "final.pth", weights_only=False)
    config["run_dir"] = str(tmp_path / "resumed")
    config["run_control"]["stop_after_steps"] = 2
    assert run(config) == PREFLIGHT_STOP
    assert not (tmp_path / "resumed" / "final.pth").exists()
    assert json.loads((tmp_path / "resumed" / "status.json").read_text())["state"] == "paused"
    config["run_control"]["stop_after_steps"] = 0
    assert run(config) == 0
    result = torch.load(tmp_path / "resumed" / "final.pth", weights_only=False)
    assert result["context"]["clock"] == reference["context"]["clock"]
    for name, tensor in reference["model"].items():
        torch.testing.assert_close(result["model"][name], tensor, rtol=0, atol=0)
    status = json.loads((tmp_path / "resumed" / "status.json").read_text())
    assert status["state"] == "complete" and status["step"] == 6
    if not iid:
        assert status["admitted"] == 9


def test_resume_refuses_changed_recipe(config):
    config["run_control"]["stop_after_steps"] = 1
    assert run(config) == PREFLIGHT_STOP
    config["seed"] += 1
    with pytest.raises(ValueError, match="manifest differs"):
        run(config)


def test_runner_rejects_incompatible_budgets(config):
    config["driver"]["advance"]["n"] = 2
    with pytest.raises(ValueError, match="one frame"):
        prepare(config)
    config["driver"]["advance"]["n"] = 1
    config["stream"]["transcript_json"] = "unused.json"
    with pytest.raises(ValueError, match="frames only"):
        prepare(config)


def test_bfloat16_forward_keeps_loss_and_gradients_float32(config):
    model = build_model(config["model"], imports=config["_imports_"])
    loss = MAE("frames", "vision", "reconstruction", precision="bfloat16")
    batch = {"frames": {K.SLOT_VIEWS: torch.randn(2, 1, 3, 32, 32)}}
    observed = []
    decoder = model.predictors["reconstruction"]
    assert isinstance(decoder, MAEDecoder)
    handle = decoder.pixels.register_forward_hook(
        lambda module, args, output: observed.append(output.dtype)
    )
    value = loss.compute(batch, model, Context(device="cpu"))["mse"]
    handle.remove()
    assert observed == [torch.bfloat16]
    assert value.dtype == torch.float32 and torch.isfinite(value)
    value.backward()
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with pytest.raises(ValueError, match="precision"):
        MAE("frames", "vision", "reconstruction", precision="float16")


def test_nested_babyview_orders_children_dates_parts_and_frames(tmp_path):
    ids = ["S2/S2_2024-02-01_10_rec", "S2/S2_2024-02-01_2_rec", "S1/S1_2024-03-01_1_rec"]
    for sid in ids:
        p = tmp_path / sid
        p.mkdir(parents=True)
        for i in (10, 2, 1):
            Image.new("RGB", (8, 16), (i, i, i)).save(p / f"frame_{i:08d}.png")
    stream = BabyViewStream(str(tmp_path), session_glob="*/*", natural_order=True)
    assert stream.session_ids() == [ids[2], ids[1], ids[0]]
    assert stream.total_observations() == 9
    sessions = list(stream.sessions())
    frames = list(sessions[0].observations())
    assert [f.t for f in frames] == [0, 0.2, 1.8]
    assert stream.open_frame(frames[1].ref, 0).getpixel((0, 0)) == (2, 2, 2)
    worker_copy = pickle.loads(pickle.dumps(stream))
    assert not worker_copy._frames_cache
    assert worker_copy.open_frame(frames[2].ref, 0).getpixel((0, 0)) == (10, 10, 10)
    # Flat roots and explicit ordering remain supported.
    assert BabyViewStream(str(tmp_path / "S2")).session_ids() == [
        Path(ids[0]).name,
        Path(ids[1]).name,
    ]
    assert BabyViewStream(str(tmp_path), sessions=ids).session_ids() == ids


@pytest.mark.parametrize("dataset", ["saycam", "babyview"])
@pytest.mark.parametrize("mode", ["streaming", "iid"])
def test_full_project_configs_compose_and_resolve_budget(config, tmp_path, dataset, mode):
    directory = Path(__file__).resolve().parents[1] / "projects" / "streammae" / "configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        composed = compose(
            config_name=f"full_{dataset}_{mode}", overrides=[f"stream.root_dir={tmp_path}"]
        )
    plain = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(plain, dict)
    assert plain["losses"][0]["precision"] == "bfloat16"
    assert plain["hooks"]["checkpoint_every"] == 0
    assert plain["hooks"]["snapshot_every"] > 0
    assert plain["solver"]["schedule"]["floor"] == 1
    # Exercise pipeline construction with tiny geometry and a real frame directory.
    for key in ("device", "stream", "model", "views", "loader"):
        plain[key] = deepcopy(config[key])
    plain["run_control"]["batch_size"] = 4
    if mode == "iid":
        plain["driver"]["batch"]["n"] = 4
    else:
        for key in ("driver", "admission", "memories", "recipe", "routes"):
            plain[key] = deepcopy(config[key])
    assert prepare(plain)[2]["updates"] == 6


@pytest.mark.parametrize(
    "child_exit,requeue,expected",
    [(75, "1", 0), (75, "0", 75), (76, "1", 76), (1, "1", 1), (0, "1", 0)],
)
def test_slurm_launcher_requeues_only_requested_pauses(tmp_path, child_exit, requeue, expected):
    import subprocess

    binary = tmp_path / "bin"
    binary.mkdir()
    apptainer = binary / "apptainer"
    apptainer.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$MAE_RUN_ROOT/args.txt"\n' + f"exit {child_exit}\n"
    )
    apptainer.chmod(0o755)
    scheduler = binary / "scontrol"
    scheduler.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > "$MAE_RUN_ROOT/requeue.txt"\n')
    scheduler.chmod(0o755)
    overlays = tmp_path / "overlays.txt"
    overlays.write_text("/first.sqf\n/last.sqf")  # deliberately no final newline
    env = {
        "PATH": f"{binary}:/usr/bin:/bin",
        "MAE_SOURCE": str(tmp_path),
        "MAE_PYTHON": "/unused/python",
        "MAE_IMAGE": "/unused/image",
        "MAE_BINDPATH": str(tmp_path),
        "MAE_OVERLAY_LIST": str(overlays),
        "MAE_RUN_ROOT": str(tmp_path / "run"),
        "MAE_REQUEUE": requeue,
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    script = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "mae.sbatch"
    result = subprocess.run(["bash", str(script), "full_saycam_streaming"], env=env, timeout=10)
    assert result.returncode == expected
    arguments = (tmp_path / "run" / "args.txt").read_text().splitlines()
    assert "/first.sqf:ro" in arguments and "/last.sqf:ro" in arguments
    assert any("SLURM_JOB_ID=123" in arg for arg in arguments)
    assert any("CUDA_VISIBLE_DEVICES=0" in arg for arg in arguments)
    marker = tmp_path / "run" / "requeue.txt"
    assert marker.exists() == (child_exit == 75 and requeue == "1")
    if marker.exists():
        assert marker.read_text() == "requeue 123\n"


def test_sources_reject_missing_and_non_directory_roots(tmp_path):
    from egolearn.data.sources.saycam import SaycamStream

    for source in (SaycamStream, BabyViewStream):
        with pytest.raises(FileNotFoundError):
            source(str(tmp_path / "missing")).session_ids()
        file = tmp_path / "file"
        file.touch()
        with pytest.raises(NotADirectoryError):
            source(str(file)).session_ids()


def test_expected_inventory_count_is_enforced(config):
    config["run_control"]["expected_frames"] = 10
    with pytest.raises(ValueError, match="expected 10 frames, found 9"):
        prepare(config)


@pytest.mark.parametrize("iid", [False, True])
def test_milestone_matches_update_budget_and_precedes_pause(config, tmp_path, iid):
    if iid:
        offline(config)
    config["hooks"]["snapshot_milestones"] = [0.5]
    config["run_control"]["stop_after_steps"] = 3
    assert run(config) == PREFLIGHT_STOP
    state = torch.load(tmp_path / "run" / "snapshot_050pct.pth", weights_only=False)
    assert state["context"]["clock"]["step"] == 3  # ceil(.5 * 6), in both modes


def test_timed_pause_uses_requeue_exit_code(config, tmp_path):
    config["run_control"]["max_seconds"] = 1e-9
    assert run(config) == PAUSED
    assert json.loads((tmp_path / "run" / "status.json").read_text())["step"] == 1


def test_invalid_update_cannot_overwrite_a_valid_snapshot(tmp_path):
    from types import SimpleNamespace

    from egolearn.methods.mae.train import UpdateSnapshots

    path = tmp_path / "snapshot_latest.pth"
    path.write_bytes(b"previous good snapshot")
    hook = UpdateSnapshots(tmp_path, 2, {"snapshot_every": 1, "snapshot_milestones": [0.5]})
    driver = SimpleNamespace(ctx=SimpleNamespace(clock=SimpleNamespace(step=1, compose_seq=2)))
    with pytest.raises(RuntimeError, match="preserving"):
        hook.on_step(driver, {})
    assert path.read_bytes() == b"previous good snapshot"
    assert not (tmp_path / "snapshot_050pct.pth").exists()


def test_resume_refuses_changed_runtime(config, monkeypatch):
    import egolearn.methods.mae.train as train_module

    monkeypatch.setattr(train_module, "runtime_identity", lambda: {"source_sha256": "candidate1"})
    config["run_control"]["stop_after_steps"] = 1
    assert run(config) == PREFLIGHT_STOP
    monkeypatch.setattr(train_module, "runtime_identity", lambda: {"source_sha256": "candidate2"})
    with pytest.raises(ValueError, match="manifest differs"):
        run(config)
