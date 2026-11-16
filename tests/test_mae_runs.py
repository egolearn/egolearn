"""Full-pass budgets, interruption recovery, and nested BabyView sources."""

import json
import pickle
import signal
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

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
from egolearn.training.hooks import WandbHook


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
    assert prepare(config)[2] == {
        "frames": 9,
        "updates": 6,
        "batch_size": 4,
        "stride": 1,
        "updates_per_epoch": 2,
        "window_size": 0 if iid else 4,
        "reference_window_size": 4,
        "drop_fraction": 0,
        "examples": 24,
    }
    assert run(config) == 0
    reference = torch.load(tmp_path / "run" / "final.pth", weights_only=False)
    config["run_dir"] = str(tmp_path / "resumed")
    config["run_control"]["stop_after_steps"] = 2
    assert run(config) == PREFLIGHT_STOP
    assert not (tmp_path / "resumed" / "final.pth").exists()
    assert json.loads((tmp_path / "resumed" / "status.json").read_text())["state"] == "paused"
    [record_path] = (tmp_path / "resumed" / "records").glob("*/record.json")
    record = json.loads(record_path.read_text())
    assert record["status"] == "paused" and record["pause_exit_code"] == PREFLIGHT_STOP
    assert "error_type" not in record
    assert record["metrics"]["train/steps"] == record["metrics"]["train/last_step/step"] == 2
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
    records = [json.loads(p.read_text()) for p in record_path.parent.parent.glob("*/record.json")]
    assert sorted(r["status"] for r in records) == ["completed", "paused"]


def test_resume_refuses_changed_recipe(config):
    config["run_control"]["stop_after_steps"] = 1
    assert run(config) == PREFLIGHT_STOP
    config["seed"] += 1
    with pytest.raises(ValueError, match="manifest differs"):
        run(config)


def test_full_pass_rejects_explicit_resume(config, tmp_path, monkeypatch):
    import egolearn

    build = Mock(side_effect=AssertionError("must reject resume before building"))
    monkeypatch.setattr(egolearn, "build", build)
    config["resume"] = str(tmp_path / "previous" / "snapshot_050pct.pth")
    with pytest.raises(ValueError, match=r"resume.*snapshot_latest\.pth"):
        run(config)
    build.assert_not_called()
    assert not (Path(config["run_dir"]) / "manifest.json").exists()


@pytest.mark.parametrize("filename", ["snapshot_latest.pth", "snapshot_050pct.pth"])
def test_resume_refuses_checkpoints_without_manifest(config, filename, monkeypatch):
    import egolearn.methods.mae.train as train_module

    root = Path(config["run_dir"])
    root.mkdir()
    checkpoint = root / filename
    checkpoint.write_bytes(b"foreign checkpoint")
    load = Mock(side_effect=AssertionError("must not load an untracked checkpoint"))
    monkeypatch.setattr(train_module, "load_snapshot", load)
    with pytest.raises(ValueError, match="checkpoint directory has no scientific manifest"):
        run(config)
    load.assert_not_called()
    assert checkpoint.read_bytes() == b"foreign checkpoint"
    assert sorted(p.name for p in root.glob("*.pth")) == [filename]
    assert not (root / "manifest.json").exists()
    assert not (root / "status.json").exists()


def test_resume_allows_relocation_and_logging_changes(config, tmp_path, monkeypatch):
    config["run_control"]["stop_after_steps"] = 2
    assert run(config) == PREFLIGHT_STOP
    original = Path(config["run_dir"])
    manifest = (original / "manifest.json").read_bytes()
    relocated = tmp_path / "relocated"
    original.rename(relocated)
    config["run_dir"] = str(relocated)
    config["run_control"]["stop_after_steps"] = 0
    config["wandb"] = {"mode": "online", "project": "synthetic"}
    config["hooks"] = {"log_every": 2, "snapshot_every": 1, "snapshot_milestones": [1.0]}
    config["record_json"] = str(tmp_path / "resumed_record.json")
    tracker = Mock()
    monkeypatch.setattr(WandbHook, "_ensure_run", lambda self: tracker)
    assert run(config) == 0
    record = json.loads(Path(config["record_json"]).read_text())
    assert record["start_step"] == 2 and record["status"] == "completed"
    assert (relocated / "manifest.json").read_bytes() == manifest
    assert (relocated / "final.pth").exists()
    assert (relocated / "snapshot_100pct.pth").exists()
    assert [call.kwargs["step"] for call in tracker.log.call_args_list] == [4, 6]


def test_runner_rejects_incompatible_budgets(config):
    config["driver"]["advance"]["n"] = 2
    with pytest.raises(ValueError, match="advance must equal stride"):
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
    assert plain["run_control"]["stride"] == 32
    assert plain["driver"]["advance"]["n"] == 32
    assert plain["driver"]["warmup"]["admitted"] == 480
    assert plain["hooks"]["snapshot_each_epoch"] is True
    assert plain["views"]["reconstruction"]["uint8"] is True
    assert plain["loader"]["packed"] is True
    assert plain["loader"]["pin_memory"] is True
    if mode == "iid":
        assert plain["driver"]["prefetch"] is True
        assert plain["loader"]["cache_frames"] == 0
    else:
        assert plain["loader"]["cache_frames"] == 1024
    # Exercise pipeline construction with tiny geometry and a real frame directory.
    for key in ("device", "stream", "model", "views", "loader"):
        plain[key] = deepcopy(config[key])
    plain["run_control"]["batch_size"] = 4
    plain["run_control"]["stride"] = 1
    plain["driver"]["prefetch"] = False
    if mode == "iid":
        plain["driver"]["batch"]["n"] = 4
    else:
        for key in ("driver", "admission", "memories", "recipe", "routes"):
            plain[key] = deepcopy(config[key])
    plain["wandb"] = {"mode": "offline", "project": "synthetic"}
    pipe, resolved, budget = prepare(plain)
    try:
        assert budget["updates"] == 6
        assert resolved["solver"]["schedule"]["total_steps"] == 6
        assert resolved["driver"]["max_steps"] == 6
        assert pipe.driver.run_config == resolved
        hook = next(h for h in pipe.driver.hooks if isinstance(h, WandbHook))
        assert hook._init_kw["config"] == resolved
    finally:
        pipe.loader.close()


@pytest.fixture
def mae_launcher(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    apptainer = binary / "apptainer"
    apptainer.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$MAE_RUN_ROOT/args.txt"\n'
        'exit "${TEST_CHILD_EXIT:-0}"\n'
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
        "SLURM_JOB_ID": "123",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    script = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "mae.sbatch"
    return script, env


@pytest.mark.parametrize("recipe", ["full_saycam_streaming", "matched_saycam", "matched_babyview"])
@pytest.mark.parametrize(
    "child_exit,requeue,expected",
    [(75, "1", 0), (75, "0", 75), (76, "1", 76), (1, "1", 1), (0, "1", 0)],
)
def test_slurm_launcher_requeues_only_requested_pauses(
    tmp_path, mae_launcher, recipe, child_exit, requeue, expected
):
    import subprocess

    script, env = mae_launcher
    env.update(TEST_CHILD_EXIT=str(child_exit), MAE_REQUEUE=requeue)
    result = subprocess.run(["bash", str(script), recipe], env=env, timeout=10)
    assert result.returncode == expected
    arguments = (tmp_path / "run" / "args.txt").read_text().splitlines()
    assert "/first.sqf:ro" in arguments and "/last.sqf:ro" in arguments
    assert any("SLURM_JOB_ID=123" in arg for arg in arguments)
    assert any("CUDA_VISIBLE_DEVICES=0" in arg for arg in arguments)
    marker = tmp_path / "run" / "requeue.txt"
    assert marker.exists() == (child_exit == 75 and requeue == "1")
    if marker.exists():
        assert marker.read_text() == "requeue 123\n"


@pytest.mark.parametrize("recipe", ["matched_saycam", "matched_babyview"])
def test_slurm_matched_recipes_separate_ablations_and_seeds(tmp_path, mae_launcher, recipe):
    import subprocess

    script, env = mae_launcher
    directory = script.parents[2] / "projects" / "streammae" / "configs"
    runs = set()
    for overrides, ablation, seed in [
        ([], "motion", 1337),
        (["ablation=vanilla"], "vanilla", 1337),
        (["ablation=vanilla", "seed=42"], "vanilla", 42),
    ]:
        result = subprocess.run(["bash", str(script), recipe, *overrides], env=env, timeout=10)
        assert result.returncode == 0
        arguments = (tmp_path / "run" / "args.txt").read_text().splitlines()
        position = arguments.index("--config-name")
        assert arguments[position + 1] == recipe
        with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
            cfg = compose(
                config_name=recipe,
                overrides=arguments[position + 2 :],
                return_hydra_config=True,
            )
        assert cfg.run_dir == str(tmp_path / "run" / recipe / ablation / f"seed_{seed}")
        assert cfg.hydra.run.dir == f"{cfg.run_dir}/hydra/123_0"
        runs.add(cfg.run_dir)
    assert len(runs) == 3


@pytest.mark.parametrize("recipe", ["full_saycam_streaming", "matched_saycam", "matched_babyview"])
def test_slurm_launcher_respects_output_override(tmp_path, mae_launcher, recipe):
    import subprocess

    script, env = mae_launcher
    env["SLURM_RESTART_COUNT"] = "2"
    output = tmp_path / "custom output"
    result = subprocess.run(["bash", str(script), recipe, f"run_dir={output}"], env=env, timeout=10)
    assert result.returncode == 0
    arguments = (tmp_path / "run" / "args.txt").read_text().splitlines()
    position = arguments.index("--config-name")
    directory = script.parents[2] / "projects" / "streammae" / "configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        cfg = compose(
            config_name=recipe,
            overrides=arguments[position + 2 :],
            return_hydra_config=True,
        )
    assert cfg.run_dir == str(output)
    assert cfg.hydra.run.dir == f"{output}/hydra/123_2"


def test_slurm_launcher_rejects_unknown_recipe(tmp_path, mae_launcher):
    import subprocess

    script, env = mae_launcher
    result = subprocess.run(
        ["bash", str(script), "matched_unknown"],
        env=env,
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unsupported MAE config: matched_unknown" in result.stderr
    assert not (tmp_path / "run" / "args.txt").exists()


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


@pytest.mark.parametrize("phase", ["prepare", "load_snapshot"])
@pytest.mark.parametrize("trigger", ["time", "signal"])
def test_startup_and_replay_honor_pause_before_training(
    config, tmp_path, monkeypatch, phase, trigger
):
    import egolearn.methods.mae.train as train_module

    start_step = 0
    if phase == "load_snapshot":
        config["run_control"]["stop_after_steps"] = 2
        assert run(config) == PREFLIGHT_STOP
        config["run_control"]["stop_after_steps"] = 0
        start_step = 2
    original = getattr(train_module, phase)
    old_handler = signal.getsignal(signal.SIGUSR1)
    now = 0.0
    monkeypatch.setattr(train_module.time, "monotonic", lambda: now)
    config["run_control"]["max_seconds"] = 30

    def interrupted(*args):
        nonlocal now
        if trigger == "signal":
            # Check first so a broken runner cannot terminate the pytest process.
            assert signal.getsignal(signal.SIGUSR1) != old_handler
            signal.raise_signal(signal.SIGUSR1)
        else:
            now += 31
        return original(*args)

    monkeypatch.setattr(train_module, phase, interrupted)
    assert run(config) == PAUSED
    assert signal.getsignal(signal.SIGUSR1) == old_handler
    root = tmp_path / "run"
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "paused" and status["step"] == start_step
    assert status["seconds_this_attempt"] == (31 if trigger == "time" else 0)
    state = torch.load(root / "snapshot_latest.pth", weights_only=False)
    assert state["context"]["clock"]["step"] == start_step
    assert not (root / "final.pth").exists()
    records = [json.loads(p.read_text()) for p in root.glob("records/*/record.json")]
    [record] = [r for r in records if r.get("pause_exit_code") == PAUSED]
    assert record["status"] == "paused" and "error_type" not in record
    assert record["start_step"] == record["metrics"]["train/steps"] == start_step


def test_startup_failure_restores_signal_handler_and_records_failure(config, tmp_path, monkeypatch):
    import egolearn.methods.mae.train as train_module

    old_handler = signal.getsignal(signal.SIGUSR1)

    def fail(_):
        assert signal.getsignal(signal.SIGUSR1) != old_handler
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(train_module, "prepare", fail)
    with pytest.raises(RuntimeError, match="synthetic setup failure"):
        run(config)
    assert signal.getsignal(signal.SIGUSR1) == old_handler
    [record_path] = (tmp_path / "run").glob("records/*/record.json")
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed" and record["error_type"] == "RuntimeError"


def test_resume_replaces_only_milestones_ahead_of_rolling_snapshot(config, tmp_path, monkeypatch):
    from egolearn.methods.mae.train import RunControl

    config["hooks"].update(snapshot_every=3, snapshot_each_epoch=True, snapshot_milestones=[2 / 3])
    on_step = RunControl.on_step

    def crash(self, driver, logs):
        if driver.ctx.clock.step == 4:
            raise RuntimeError("allocation lost after milestone")
        on_step(self, driver, logs)

    monkeypatch.setattr(RunControl, "on_step", crash)
    with pytest.raises(RuntimeError, match="allocation lost"):
        run(config)
    root = tmp_path / "run"
    latest = torch.load(root / "snapshot_latest.pth", weights_only=False)
    assert latest["context"]["clock"]["step"] == 3
    kept = root / "snapshot_epoch_001.pth"
    kept_bytes = kept.read_bytes()
    abandoned = {
        name: (root / name).read_bytes()
        for name in ("snapshot_epoch_002.pth", "snapshot_066pct.pth")
    }
    # Pause at the restored step to inspect retirement before any new update.
    monkeypatch.setattr(RunControl, "on_step", on_step)
    config["run_control"]["stop_after_steps"] = 3
    assert run(config) == PREFLIGHT_STOP
    for name, contents in abandoned.items():
        assert not (root / name).exists()
        [archived] = (root / "abandoned_snapshots").glob(f"*/{name}")
        assert archived.read_bytes() == contents
    assert kept.read_bytes() == kept_bytes
    # Missing milestones behind the restored step cannot be reconstructed later.
    kept.unlink()
    config["run_control"]["stop_after_steps"] = 0
    assert run(config) == 0
    assert not kept.exists()
    for name, contents in abandoned.items():
        assert (root / name).read_bytes() != contents
        state = torch.load(root / name, weights_only=False)
        assert state["context"]["clock"]["step"] == 4


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


@pytest.mark.parametrize("iid", [False, True])
def test_stride_tail_epoch_checkpoints_and_resume(config, tmp_path, iid):
    config["run_control"]["stride"] = 2
    config["driver"]["advance"]["n"] = 2
    config["driver"]["warmup"]["admitted"] = 2
    config["hooks"]["snapshot_each_epoch"] = True
    if iid:
        offline(config)
    # N=9, B=4: windows end at 4,6,8,9. No tail frame is discarded.
    assert prepare(config)[2]["updates"] == 4
    assert run(config) == 0
    reference = torch.load(tmp_path / "run" / "final.pth", weights_only=False)
    for epoch in (1, 2):
        state = torch.load(tmp_path / "run" / f"snapshot_epoch_{epoch:03d}.pth", weights_only=False)
        assert state["context"]["clock"]["step"] == epoch * 2
        if not iid:
            assert state["context"]["clock"]["admitted"] == (6 if epoch == 1 else 9)
    config["run_dir"] = str(tmp_path / "resumed_stride")
    config["run_control"]["stop_after_steps"] = 2
    assert run(config) == PREFLIGHT_STOP
    config["run_control"]["stop_after_steps"] = 0
    assert run(config) == 0
    result = torch.load(tmp_path / "resumed_stride" / "final.pth", weights_only=False)
    for name, tensor in reference["model"].items():
        torch.testing.assert_close(result["model"][name], tensor, rtol=0, atol=0)


def test_packed_prefetch_pause_resume_matches_serial(config, tmp_path):
    offline(config)
    config["views"]["reconstruction"]["uint8"] = True
    assert run(config) == 0
    reference = torch.load(tmp_path / "run" / "final.pth", weights_only=False)
    config["run_dir"] = str(tmp_path / "prefetched")
    config["loader"] = {"workers": 2, "packed": True}
    config["driver"]["prefetch"] = True
    config["run_control"]["stop_after_steps"] = 1
    assert run(config) == PREFLIGHT_STOP
    config["loader"] = {"workers": 0, "packed": False}
    config["driver"]["prefetch"] = False
    config["run_control"]["stop_after_steps"] = 0
    assert run(config) == 0
    result = torch.load(tmp_path / "prefetched" / "final.pth", weights_only=False)
    assert len(list((tmp_path / "prefetched" / "attempts").glob("*.json"))) == 2
    assert result["context"]["clock"] == reference["context"]["clock"]
    for name, tensor in reference["model"].items():
        torch.testing.assert_close(result["model"][name], tensor, rtol=0, atol=0)


def test_epoch_checkpoint_includes_short_final_epoch(config, tmp_path):
    config["run_control"]["stride"] = 3
    config["driver"]["advance"]["n"] = 3
    config["driver"]["warmup"]["admitted"] = 1
    config["hooks"]["snapshot_each_epoch"] = True
    assert run(config) == 0  # 3 updates, 2 updates per IID epoch
    state = torch.load(tmp_path / "run" / "snapshot_epoch_002.pth", weights_only=False)
    assert state["context"]["clock"]["step"] == 3


@pytest.mark.parametrize(
    "ablation,dataset,window,drop_path,jitter,candidates",
    [
        ("vanilla", "saycam", 4, 0, 0, 0),
        ("iid", "saycam", 0, 0, 0, 0),
        ("datadrop", "saycam", 8, 0, 0, 0),
        ("drop_path", "saycam", 8, 0.25, 0, 0),
        ("color_jitter", "saycam", 8, 0.25, 0.5, 0),
        ("random_crops", "saycam", 8, 0.25, 0.5, 1),
        ("motion", "babyview", 8, 0.25, 0.5, 4),
    ],
)
def test_matched_ablations_train_with_same_budget_and_lr(
    config, tmp_path, ablation, dataset, window, drop_path, jitter, candidates
):
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parents[1] / "projects/streammae/configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        composed = compose(
            config_name=f"matched_{dataset}",
            overrides=[
                f"ablation={ablation}",
                f"stream.root_dir={tmp_path / 'frames'}",
                "run_control.batch_size=4",
                "run_control.reference_window_size=8",
                "run_control.stride=3",
                "driver.warmup.admitted=5",
                "device=cpu",
            ],
        )
    cfg = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(cfg, dict)
    assert cfg["model"]["encoders"]["vision"].get("drop_path_rate", 0) == drop_path
    assert cfg["views"]["reconstruction"].get("jitter_probability", 0) == jitter
    assert cfg["views"]["reconstruction"].get("motion_candidates", 0) == candidates
    assert cfg["run_dir"].endswith(f"/{ablation}/seed_1337")
    cfg["model"] = deepcopy(config["model"])
    cfg["model"]["encoders"]["vision"]["drop_path_rate"] = drop_path
    cfg["views"]["reconstruction"]["size"] = 32
    if candidates:
        cfg["views"]["reconstruction"]["patch_size"] = 8
        cfg["admission"][0]["patch_size"] = 8
    cfg["stream"]["session_glob"] = "*"
    cfg["losses"] = config["losses"]
    cfg["solver"] = config["solver"]
    cfg["run_dir"] = str(tmp_path / ablation)
    cfg["loader"] = {"workers": 0}
    cfg["driver"]["prefetch"] = False
    cfg["hooks"] = {"log_every": 1}
    assert run(cfg) == 0
    root = Path(cfg["run_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    budget = manifest["budget"]
    assert budget["frames"] == 9
    assert budget["updates"] == 2  # first update at 8, then consume the short tail
    assert budget["examples"] == 8
    assert budget["window_size"] == window
    assert budget["reference_window_size"] == 8
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [r["lr"] for r in rows] == pytest.approx([0.0005, 0.001])
    assert [r["train/consumed_frames"] for r in rows] == ([4, 8] if ablation == "iid" else [8, 9])
    assert [r["train/optimizer_updates"] for r in rows] == [1, 2]
    assert [r["train/examples_seen"] for r in rows] == [4, 8]
    assert all(r["train/window_size"] == window for r in rows)
    assert all(r["train/retained_examples"] == 4 for r in rows)
    assert all(r["throughput/examples_per_second"] > 0 for r in rows)
    record_path = next((root / "records").glob("*/record.json"))
    record = json.loads(record_path.read_text())
    assert record["config"]["driver"]["max_steps"] == 2
    assert record["config"]["solver"]["schedule"]["total_steps"] == 2
    assert record["metrics"]["train/last_step/train/optimizer_updates"] == 2


def test_reference_window_validation(config):
    config["run_control"]["reference_window_size"] = 3
    with pytest.raises(ValueError, match="at least window_size"):
        prepare(config)
    config["run_control"]["reference_window_size"] = 10
    config["driver"]["warmup"]["admitted"] = 9
    with pytest.raises(ValueError, match="at least one reference window"):
        prepare(config)
    config["run_control"]["reference_window_size"] = 8
    with pytest.raises(ValueError, match="warmup must equal reference window"):
        prepare(config)


def test_budget_throughput_counts_only_current_attempt(config, monkeypatch):
    from egolearn.methods.mae.train import BudgetMetrics

    pipe, _, budget = prepare(config)
    pipe.ctx.clock.step = 2
    pipe.ctx.clock.admitted = 5
    times = iter([10.0, 12.0])
    monkeypatch.setattr("egolearn.methods.mae.train.time.monotonic", lambda: next(times))
    hook = BudgetMetrics(budget, pipe.driver)
    pipe.ctx.clock.step = 3
    pipe.ctx.clock.admitted = 6
    logs: dict[str, float] = {}
    hook.on_step(pipe.driver, logs)
    assert logs["train/optimizer_updates"] == 3
    assert logs["train/consumed_frames"] == 6
    assert logs["throughput/updates_per_second"] == 0.5
    assert logs["throughput/frames_per_second"] == 0.5
    assert logs["throughput/examples_per_second"] == 2
    pipe.loader.close()
