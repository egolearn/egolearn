"""Synthetic StreamMAE tests: residual gating, causal crops, DataDrop and resume."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.core import keys as K
from egolearn.data.loading import Loader
from egolearn.data.views import ViewBank
from egolearn.methods.mae.model import DropPath, MAEEncoder
from egolearn.methods.mae.train import PREFLIGHT_STOP, prepare, run
from egolearn.methods.streammae.motion import MotionFrames, patch_motion
from egolearn.methods.streammae.views import StreamMAEView, select_crop


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(1)
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        config = OmegaConf.to_container(compose(config_name="toy_mae"), resolve=True)
    assert isinstance(config, dict)
    root = tmp_path / "frames"
    for session in ("S_a", "S_b"):
        directory = root / session
        directory.mkdir(parents=True)
        for index in range(8):
            pixels = np.zeros((40, 48, 3), dtype=np.uint8)
            pixels[8:24, index * 4 : index * 4 + 12] = 220
            Image.fromarray(pixels).save(directory / f"S_{index * 0.2:.2f}.png")
    config.update(
        _imports_=["egolearn.methods.streammae"],
        stream={"_target_": "stream:saycam", "root_dir": str(root)},
        run_dir=str(tmp_path / "run"),
        run_control={"batch_size": 2, "window_size": 8, "stride": 2},
        admission=[{"_target_": "admission:streammae_frames", "patch_size": 8}],
        views={"reconstruction": {"name": "streammae", "size": 32, "patch_size": 8}},
    )
    config["memories"]["frames"]["tiers"]["window"]["capacity"] = 8
    config["recipe"]["slots"][0]["n"] = 2
    config["driver"].pop("max_steps")
    config["driver"]["advance"]["n"] = 2
    config["driver"]["warmup"]["admitted"] = 6
    config["model"]["encoders"]["vision"]["drop_path_rate"] = 0.25
    config["hooks"] = {"log_every": 1}
    return config


def test_drop_path_gates_examples_and_preserves_expectation():
    torch.manual_seed(3)
    drop = DropPath(0.5)
    x = torch.ones(4096, 3, 4)
    result = drop(x)
    assert set(result.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(result, result[:, :1, :1].expand_as(result))
    assert abs(result.mean().item() - 1) < 0.05
    drop.eval()
    assert torch.equal(drop(x), x)


def test_drop_path_preserves_checkpoint_keys_and_zero_behavior():
    baseline = MAEEncoder(image_size=32, patch_size=8, dim=32, depth=3, heads=4)
    regularized = MAEEncoder(
        image_size=32, patch_size=8, dim=32, depth=3, heads=4, drop_path_rate=0.25
    )
    regularized.load_state_dict(baseline.state_dict(), strict=True)
    rates = []
    for block in regularized.blocks:
        assert isinstance(block.dropout1, DropPath)
        rates.append(block.dropout1.probability)
    assert rates == [0, 0.125, 0.25]
    images = torch.rand(4, 3, 32, 32)
    baseline.eval()
    regularized.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            baseline(images).tokens, regularized(images).tokens, atol=0, rtol=0
        )
    regularized.train()
    torch.manual_seed(0)
    first = regularized(images).tokens
    torch.manual_seed(1)
    second = regularized(images).tokens
    assert not torch.equal(first, second)
    for block in regularized.blocks:
        assert isinstance(block.dropout1, DropPath) and isinstance(block.dropout2, DropPath)
        block.dropout1.probability = block.dropout2.probability = 0
    baseline.train()
    torch.testing.assert_close(baseline(images).tokens, regularized(images).tokens, atol=0, rtol=0)


def test_motion_l1_signed_arithmetic_patch_average_and_crop_selection():
    current = np.zeros((32, 48, 3), dtype=np.uint8)
    previous = np.full_like(current, 255)
    motion = patch_motion(current, previous, 16)
    np.testing.assert_array_equal(motion, np.full((2, 3), 765, dtype=np.float32))
    previous[:] = 0
    current[16:, 32:] = 100
    motion = patch_motion(current, previous, 16)
    assert motion.sum() == 300
    assert select_crop(motion, (1, 1), torch.tensor([[0, 0], [1, 2], [1, 0]])) == (1, 2)
    # No motion still chooses a uniformly drawn candidate, rather than the origin.
    assert select_crop(np.zeros_like(motion), (1, 1), torch.tensor([[1, 1], [0, 0]])) == (1, 1)
    assert not patch_motion(current, None, 16).any()
    assert not patch_motion(current, np.zeros((16, 16, 3), dtype=np.uint8), 16).any()


def test_admission_scores_every_frame_before_drop_and_resets_sessions(cfg):
    pipe, _, budget = prepare(cfg)
    assert budget["updates"] == 5 and budget["window_size"] == 8
    pipe.driver._run_warmup()
    assert pipe.driver._advance_once()
    entries = pipe.memories["frames"].tiers["window"].entries
    assert len(entries) == 8
    assert not entries[0][K.VIEW_CONTEXT]["motion"].any()
    assert all(entry[K.VIEW_CONTEXT]["motion"].any() for entry in entries[1:])
    assert pipe.composer is not None and pipe.recipe is not None
    for _ in range(3):
        batch = pipe.composer.compose(pipe.recipe, pipe.memories, pipe.driver.stream_head, pipe.ctx)
        assert batch["frames"][K.SLOT_VIEWS].shape == (2, 1, 3, 32, 32)
        chosen = batch["frames"][K.SLOT_META]
        assert len({entry[K.VIDEO_REF].key for entry in chosen}) == 2
    # Sampling never changes the complete window or stream position.
    assert len(pipe.memories["frames"].tiers["window"]) == 8
    assert pipe.ctx.clock.admitted == 8
    assert pipe.driver._advance_once()
    entries = pipe.memories["frames"].tiers["window"].entries
    assert not entries[-2][K.VIEW_CONTEXT]["motion"].any()
    assert entries[-1][K.VIEW_CONTEXT]["motion"].any()
    pipe.loader.close()


def test_contextual_loading_matches_packed_workers(cfg):
    pipe, _, _ = prepare(cfg)
    pipe.driver._run_warmup()
    pipe.driver._advance_once()
    entries = pipe.memories["frames"].tiers["window"].entries
    work = [
        ("frame", entry[K.VIDEO_REF], 0, ["reconstruction"], i + 7, entry[K.VIEW_CONTEXT])
        for i, entry in enumerate(entries)
    ]
    expected = pipe.loader.tensor_batch(work)
    loader = Loader(
        pipe.stream,
        pipe.views,
        workers=2,
        packed=True,
        cache_frames=8,
        imports=["egolearn.methods.streammae"],
    )
    try:
        torch.testing.assert_close(loader.tensor_batch(work), expected, atol=0, rtol=0)
        torch.testing.assert_close(loader.tensor_batch(work), expected, atol=0, rtol=0)
    finally:
        loader.close()
        pipe.loader.close()


def test_color_jitter_and_context_validation():
    image = Image.new("RGB", (64, 48), (80, 110, 140))
    context = {
        "motion": np.zeros((3, 4), dtype=np.float32),
        "patch_size": 16,
        "frame_size": (48, 64),
    }
    plain = StreamMAEView(size=32, crop_probability=0, jitter_probability=0)
    jitter = StreamMAEView(size=32, crop_probability=0, jitter_probability=1)
    torch.manual_seed(0)
    expected = plain(image, context)
    torch.manual_seed(0)
    changed = jitter(image, context)
    assert not torch.equal(expected, changed)
    with pytest.raises(ValueError, match="geometry disagree"):
        plain(image, {**context, "patch_size": 8})
    bank = ViewBank({"reconstruction": {"name": "streammae"}})
    with pytest.raises(ValueError, match="requires frame context"):
        bank.apply(image, ["reconstruction"])


def test_two_stage_crop_geometry_and_uint8_equivalence():
    image = Image.fromarray(np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3))
    context = {
        "motion": np.zeros((3, 4), dtype=np.float32),
        "patch_size": 16,
        "frame_size": (48, 64),
    }
    options: dict[str, Any] = dict(
        size=32, crop_size=(32, 32), crop_probability=1, jitter_probability=0
    )
    floats, uint8 = StreamMAEView(**options), StreamMAEView(**options, uint8=True)
    torch.manual_seed(8)
    expected = floats(image, context)
    torch.manual_seed(8)
    raw = uint8(image, context)
    assert raw.dtype == torch.uint8
    normalized = (
        raw.float() / 255 - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    torch.testing.assert_close(expected, normalized, atol=0, rtol=0)
    with pytest.raises(ValueError, match="does not fit"):
        StreamMAEView(crop_size=(64, 64), crop_probability=1)(image, context)


@pytest.mark.parametrize("pause_step,read_count", [(1, 8), (2, 7)])
def test_streammae_pause_resume_matches_and_does_not_redecode_history(
    cfg, tmp_path, monkeypatch, pause_step, read_count
):
    assert run(cfg) == 0
    reference = torch.load(tmp_path / "run" / "final.pth", weights_only=False)
    cfg["run_dir"] = str(tmp_path / "resumed")
    cfg["run_control"]["stop_after_steps"] = pause_step
    assert run(cfg) == PREFLIGHT_STOP
    cfg["run_control"]["stop_after_steps"] = 0
    reads = []
    original = MotionFrames._read

    def counted(self, ref):
        reads.append(ref)
        return original(self, ref)

    monkeypatch.setattr(MotionFrames, "_read", counted)
    assert run(cfg) == 0
    # At the session boundary no predecessor should be decoded.
    assert len(reads) == read_count
    restored = torch.load(tmp_path / "resumed" / "final.pth", weights_only=False)
    assert restored["context"]["clock"] == reference["context"]["clock"]
    for name in reference["model"]:
        torch.testing.assert_close(
            restored["model"][name], reference["model"][name], atol=0, rtol=0
        )


@pytest.mark.parametrize("dataset", ["saycam", "babyview", "walkingtours", "wtpp_london"])
def test_full_streammae_recipes(dataset):
    directory = Path(__file__).resolve().parents[1] / "projects/streammae/configs"
    overrides = ["stream.root_dir=/unused"]
    if dataset == "walkingtours":
        overrides.append("stream.sessions=[zurich,london]")
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        composed = compose(config_name=f"full_{dataset}_streammae", overrides=overrides)
    cfg = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(cfg, dict)
    assert cfg["recipe"]["slots"][0]["n"] == 512
    assert cfg["memories"]["frames"]["tiers"]["window"]["capacity"] == 2048
    stride = 1 if dataset in ("walkingtours", "wtpp_london") else 32
    assert cfg["driver"]["warmup"]["admitted"] == 2048 - stride
    assert cfg["driver"]["advance"]["n"] == stride
    assert cfg["model"]["encoders"]["vision"]["drop_path_rate"] == 0.25
    assert cfg["views"]["reconstruction"]["uint8"] is True
    assert cfg["solver"]["groups"][0]["lr"] == 1.6e-4
    if dataset in ("walkingtours", "wtpp_london"):
        assert cfg["stream"]["_target_"] == "stream:walkingtours"
        assert cfg["stream"]["sessions"] == (
            ["london"] if dataset == "wtpp_london" else ["zurich", "london"]
        )
        assert cfg["stream"]["fps"] == 3.75
        assert cfg["views"]["reconstruction"]["frame_size"] == [720, 1280]
        assert cfg["admission"][0]["frame_size"] == [720, 1280]
        assert cfg["views"]["reconstruction"]["crop_size"] == [336, 592]
        assert cfg["hooks"]["snapshot_each_epoch"] is False


def test_prepare_rejects_inconsistent_window_and_geometry(cfg):
    broken = deepcopy(cfg)
    broken["run_control"]["window_size"] = 1
    with pytest.raises(ValueError, match="window_size"):
        prepare(broken)
    cfg["views"]["reconstruction"]["patch_size"] = 16
    with pytest.raises(egolearn.pipeline.PipelineError, match="geometry must agree"):
        prepare(cfg)


def test_stream_head_rejected_when_metadata_is_restored_from_memory(cfg):
    cfg["recipe"]["slots"][0]["source"] = {"kind": "stream_head", "schema": "RawFrame"}
    with pytest.raises(egolearn.pipeline.PipelineError, match="snapshot-restored memory"):
        egolearn.build(cfg)


def test_serial_augmentation_does_not_reseed_cuda(cfg, monkeypatch):
    pipe, _, _ = prepare(cfg)
    pipe.driver._run_warmup()
    entry = pipe.memories["frames"].tiers["window"].entries[-1]
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    pipe.loader.frame_views(
        entry[K.VIDEO_REF], 0, ["reconstruction"], seed=11, context=entry[K.VIEW_CONTEXT]
    )
    assert calls == []
    pipe.loader.close()


@pytest.mark.parametrize("offline", [False, True])
def test_context_view_without_producer_fails_before_manifest(cfg, tmp_path, offline):
    if offline:
        for key in ("admission", "memories", "routes", "recipe"):
            cfg.pop(key)
        cfg["run_control"]["window_size"] = 2
        cfg["driver"] = {
            "mode": "offline",
            "batch": {"slot": "frames", "n": 2, "views": ["reconstruction"]},
        }
    else:
        cfg["admission"] = [{"_target_": "admission:frame_emitter", "subsample": 1}]
    with pytest.raises(egolearn.pipeline.PipelineError, match=r"requires.*frame context"):
        run(cfg)
    assert not (tmp_path / "run" / "manifest.json").exists()
    assert not (tmp_path / "run" / "snapshot_latest.pth").exists()


def test_explicit_crop_rejected_on_admission_even_when_branch_disabled(cfg):
    cfg["views"]["reconstruction"].update(crop_size=[64, 64], crop_probability=0)
    pipe, _, _ = prepare(cfg)
    with pytest.raises(ValueError, match="does not fit"):
        pipe.driver._run_warmup()
    pipe.loader.close()


def test_context_view_without_kind_rejected_before_manifest(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(StreamMAEView, "context_kind", None)
    cfg["admission"] = [{"_target_": "admission:frame_emitter", "subsample": 1}]
    with pytest.raises(egolearn.pipeline.PipelineError, match=r"requires.*frame context"):
        run(cfg)
    assert not (tmp_path / "run" / "manifest.json").exists()


@pytest.mark.parametrize("subsample,count", [(2, 1), (1, 2)])
def test_full_pass_rejects_subsampling_and_multiple_emitters(cfg, subsample, count):
    cfg["views"]["reconstruction"] = {"name": "mae", "size": 32}
    cfg["admission"] = [
        {"_target_": "admission:frame_emitter", "subsample": subsample} for _ in range(count)
    ]
    with pytest.raises(ValueError, match="without subsampling"):
        prepare(cfg)


@pytest.mark.parametrize("dataset", ["saycam", "babyview", "wtpp_london"])
def test_full_merged_recipe_prepares_on_synthetic_data(cfg, dataset, tmp_path):
    root = tmp_path / dataset
    folder = (
        root
        / {"saycam": "S_20200101", "babyview": "child/recording", "wtpp_london": "london"}[dataset]
    )
    folder.mkdir(parents=True)
    for index in range(16):
        Image.new("RGB", (32, 48), (index * 10, 20, 30)).save(folder / f"frame_{index + 1:08d}.png")
    directory = Path(__file__).resolve().parents[1] / "projects/streammae/configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        composed = compose(
            config_name=f"full_{dataset}_streammae",
            overrides=[
                f"stream.root_dir={root}",
                "device=cpu",
                "run_control.batch_size=2",
                "run_control.window_size=8",
                "run_control.stride=2",
                "driver.warmup.admitted=6",
            ],
        )
    actual = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(actual, dict)
    actual["model"] = cfg["model"]
    actual["admission"][0]["patch_size"] = 8
    actual["views"]["reconstruction"].update(size=32, patch_size=8)
    actual["loader"] = {"workers": 0}
    pipe, _, budget = prepare(actual)
    assert budget["frames"] == 16 and budget["updates"] == 5
    pipe.driver._run_warmup()
    assert pipe.driver._advance_once()
    assert pipe.composer is not None and pipe.recipe is not None
    batch = pipe.composer.compose(pipe.recipe, pipe.memories, pipe.driver.stream_head, pipe.ctx)
    assert batch["frames"][K.SLOT_VIEWS].shape == (2, 1, 3, 32, 32)
    pipe.loader.close()


def test_walkingtours_training_consumes_tail_and_resumes_exactly(cfg, tmp_path):
    root = tmp_path / "walkingtours"
    for session, count in (("zurich", 9), ("london", 8)):
        folder = root / session
        folder.mkdir(parents=True)
        for index in range(1, count + 1):
            Image.new("RGB", (48, 40), (index * 20, 50, 90)).save(folder / f"frame_{index:08d}.png")
    cfg["stream"] = {
        "_target_": "stream:walkingtours",
        "root_dir": str(root),
        "sessions": ["zurich", "london"],
        "fps": 3.75,
    }
    assert run(cfg) == 0
    reference = torch.load(tmp_path / "run" / "final.pth", weights_only=False)
    assert reference["context"]["clock"]["admitted"] == 17
    assert reference["context"]["clock"]["step"] == 6
    cfg["run_dir"] = str(tmp_path / "resumed")
    cfg["run_control"]["stop_after_steps"] = 2
    assert run(cfg) == PREFLIGHT_STOP
    cfg["run_control"]["stop_after_steps"] = 0
    assert run(cfg) == 0
    restored = torch.load(tmp_path / "resumed" / "final.pth", weights_only=False)
    assert restored["context"]["clock"] == reference["context"]["clock"]
    for name in reference["model"]:
        torch.testing.assert_close(
            restored["model"][name], reference["model"][name], atol=0, rtol=0
        )
