"""V-JEPA 2 reference math, actual updates, exact resume, and real clip indexing."""

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.data.sources.saycam import BabyViewStream, SaycamStream
from egolearn.evaluation import load_model
from egolearn.methods.vjepa2 import VJEPA2Encoder, VJEPA2Predictor
from egolearn.methods.vjepa2.masking import multiblock
from egolearn.methods.vjepa2.train import run
from egolearn.model.transformer import Attention, Block, _rotate, rope_3d
from egolearn.training.hooks import Hook
from egolearn.utils.snapshot import load_snapshot, save_snapshot

CONFIGS = Path(__file__).resolve().parents[1] / "projects/vjepa2/configs"


def config(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = OmegaConf.to_container(compose(config_name="smoke"), resolve=True)
    assert isinstance(cfg, dict)
    cfg["run_dir"] = str(tmp_path)
    cfg["preflight"] = False
    cfg["record_json"] = None
    cfg["hooks"]["snapshot_every"] = 0
    return cfg


def test_rope_matches_reference_frequency_repeat_and_preserves_dtype():
    pos = torch.tensor([[0, 2, 7, 11], [1, 3, 6, 9]])
    x = torch.randn(2, 3, 4, 32)
    grid = (2, 2, 3)
    coordinates = [pos // 6, pos % 6 // 3, pos % 3]
    expected = []
    # Independently express the published 10-channel axis transform.
    for i, coordinates_i in enumerate(coordinates):
        omega = 10000 ** (-torch.arange(5).float() / 5)
        angles = coordinates_i[:, None, :, None] * omega
        c, s = angles.cos().repeat(1, 1, 1, 2), angles.sin().repeat(1, 1, 1, 2)
        part = x[..., i * 10 : (i + 1) * 10]
        turned = torch.empty_like(part)
        turned[..., ::2], turned[..., 1::2] = -part[..., 1::2], part[..., ::2]
        expected.append(part * c + turned * s)
    expected.append(x[..., 30:])
    cos, sin = rope_3d(pos, grid, 32, convention="vjepa2")
    torch.testing.assert_close(_rotate(x, cos, sin), torch.cat(expected, -1))
    assert _rotate(x.bfloat16(), cos, sin).dtype == torch.bfloat16
    paired = rope_3d(pos, grid, 32)
    assert not torch.allclose(_rotate(x, *paired), _rotate(x, cos, sin))
    attention = Attention(96, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert attention(torch.randn(2, 4, 96), cos, sin).dtype == torch.bfloat16


def test_reference_initialization_mask_selection_and_visible_positions():
    torch.manual_seed(11)
    encoder = VJEPA2Encoder(dim=32, depth=2, heads=2, patch=16, tubelet=2)
    predictor = VJEPA2Predictor(32, dim=32, depth=2, heads=2)
    assert encoder.norm.eps == 1e-6
    assert encoder.embed.bias is not None
    assert torch.count_nonzero(encoder.embed.bias) == 0
    assert torch.count_nonzero(predictor.mask_token) == 0
    assert isinstance(encoder.blocks[0], Block) and isinstance(encoder.blocks[1], Block)
    assert encoder.blocks[1].attn.proj.weight.std() < encoder.blocks[0].attn.proj.weight.std()
    clips = torch.randn(2, 4, 3, 64, 64)
    out = encoder(clips)
    ids = torch.arange(32).expand(2, -1)
    torch.testing.assert_close(out.tokens, encoder(clips, visible=ids).tokens)
    ctx_pos, tgt_pos = ids[:, :8], ids[:, 8:]
    tokens = encoder(clips, visible=ctx_pos).tokens
    pred = predictor(tokens, ctx_pos=ctx_pos, tgt_pos=tgt_pos, grid=(2, 4, 4))
    pred.square().mean().backward()
    assert predictor.mask_token.grad is not None
    assert encoder.embed.weight.grad is not None
    assert predictor.mask_token.grad[0].abs().sum() > 0
    assert torch.count_nonzero(predictor.mask_token.grad[1]) == 0
    assert encoder.embed.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="not trained"):
        predictor(tokens, ctx_pos=ctx_pos, tgt_pos=tgt_pos, grid=(2, 4, 4), mask_index=1)
    # Noncausal attention is invariant to the order of context tokens.
    torch.testing.assert_close(
        pred,
        predictor(tokens.flip(1), ctx_pos=ctx_pos.flip(1), tgt_pos=tgt_pos, grid=(2, 4, 4)),
        atol=1e-6,
        rtol=1e-5,
    )


def test_masks_share_block_size_round_area_and_are_deterministic():
    args: dict[str, Any] = dict(
        grid=(2, 4, 4), n=5, num_blocks=1, scale=(0.15, 0.15), aspect=(1, 1)
    )
    context, target = multiblock(**args, rng=random.Random(9))
    # int(16*.15)=2 -> round(sqrt(2))=1, not round(sqrt(2.4))=2.
    assert target.shape == (5, 2)
    assert context.shape == (5, 30)
    torch.testing.assert_close(target, multiblock(**args, rng=random.Random(9))[1])
    for c, t in zip(context, target, strict=True):
        assert not set(c.tolist()) & set(t.tolist())
        assert sorted(c.tolist() + t.tolist()) == list(range(32))
        assert t[1] - t[0] == 16
    with pytest.raises(ValueError, match="spatial tokens"):
        multiblock((2, 1, 1), 2, 1, (0.1, 0.2), random.Random(0))
    with pytest.raises(ValueError, match="empty or covers"):
        multiblock((2, 2, 2), 2, 1, (0.1, 0.1), random.Random(0))


def test_training_baseline_diagnostics_and_checkpoint_reload(tmp_path):
    cfg = config(tmp_path)
    cfg["record_json"] = "auto"
    cfg["preflight"] = True
    # Context creation seeds model construction identically; no input decoding consumes it.
    initial = egolearn.build(cfg)
    initial.losses[0].initialize_target(initial.model)
    expected = {k: v.clone() for k, v in initial.model.state_dict().items()}
    pipe = run(cfg)
    baseline = torch.load(tmp_path / "random.pth", weights_only=True)
    for name, value in baseline["model"].items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert pipe.ctx.clock.step == 4
    assert not torch.equal(
        pipe.model.encoders["video"].embed.weight, expected["encoders.video.embed.weight"]
    )
    assert all(p.grad is None for p in pipe.model.encoders["video_target"].parameters())
    restored = load_model(tmp_path / "final.pth")
    for name, tensor in pipe.model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 4
    assert all(math.isfinite(value) for line in lines for value in line.values())
    assert lines[-1]["frames_per_s"] == 4 * lines[-1]["clips_per_s"]
    assert lines[-1]["jepa/view0/target_std"] > 0
    record = json.loads(next(tmp_path.glob("records/*/record.json")).read_text())
    assert record["status"] == "completed"
    assert {"random.pth", "final.pth"} <= record["outputs"].keys()
    with pytest.raises(FileExistsError, match="already has checkpoints"):
        run(cfg)


def test_resume_keeps_exact_model_optimizer_and_ema_with_same_schedule(tmp_path):
    cfg = config(tmp_path / "reference")
    reference = egolearn.build(cfg)
    reference.driver.fit()

    class Pause(Hook):
        def on_step(self, driver, logs):
            if driver.ctx.clock.step == 2:
                save_snapshot(str(tmp_path / "resume.pth"), driver)
                raise InterruptedError

    split = egolearn.build({**cfg, "run_dir": str(tmp_path / "partial")})
    split.driver.hooks.append(Pause())
    with pytest.raises(InterruptedError):
        split.driver.fit()
    resumed = egolearn.build({**cfg, "run_dir": str(tmp_path / "resumed")})
    load_snapshot(str(tmp_path / "resume.pth"), resumed.driver)
    resumed.driver.fit()
    for name, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert resumed.ctx.clock.compose_seq == reference.ctx.clock.compose_seq
    for state_id, state in reference.driver.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value,
                resumed.driver.optimizer.state_dict()["state"][state_id][name],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "change, message",
    [
        ("accumulation", "grad_accumulation"),
        ("target", "specs must match"),
    ],
)
def test_invalid_training_contracts_fail_at_build(tmp_path, change, message):
    cfg = config(tmp_path)
    if change == "accumulation":
        cfg["driver"]["grad_accumulation"] = 2
    else:
        cfg["model"]["encoders"]["video_target"]["tubelet"] = 1
    with pytest.raises(egolearn.PipelineError, match=message):
        egolearn.build(cfg)


def test_bfloat16_forward_backward(tmp_path):
    cfg = config(tmp_path)
    cfg["losses"][0]["precision"] = "bfloat16"
    pipe = egolearn.build(cfg)
    refs = pipe.driver._build_index()
    batch = pipe.driver._compose(refs[:2])
    loss = pipe.losses[0].compute(batch, pipe.model, pipe.ctx)["l1"]
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert pipe.model.encoders["video"].embed.weight.grad.abs().sum() > 0


def test_nested_babyview_gap_index_and_strict_filenames(tmp_path):
    root = tmp_path / "data"
    for session in ("child/session2", "child/session10"):
        path = root / session
        path.mkdir(parents=True)
        for idx in [1, 2, 3, 4, 8, 9, 10, 11, 12]:
            Image.new("RGB", (64, 64)).save(path / f"frame_{idx:08d}.jpg")
    cfg = config(tmp_path / "run")
    cfg["stream"] = dict(
        _target_="stream:babyview",
        root_dir=str(root),
        fps=5,
        session_glob="*/*",
        natural_order=True,
        strict_filenames=True,
    )
    pipe = egolearn.build(cfg)
    assert pipe.stream.session_ids() == ["child/session2", "child/session10"]
    refs = pipe.driver._build_index()
    assert [r.locator[1:] for r in refs] == [(0, 4), (4, 8), (0, 4), (4, 8)]
    assert pipe.driver.index_stats["gaps"] == 2
    assert pipe.driver.index_stats["discarded_frames"] == 2
    assert not pipe.stream.__getstate__()["_frames_cache"]
    malformed = root / "child/session2/not_a_frame.jpg"
    Image.new("RGB", (64, 64)).save(malformed)
    with pytest.raises(ValueError, match="filename"):
        BabyViewStream(str(root), session_glob="*/*", strict_filenames=True).total_observations()
    with pytest.raises(ValueError, match="filename"):
        pipe.stream.frame_times([malformed.name])


def test_wrong_fps_and_zero_windows_fail(tmp_path):
    cfg = config(tmp_path)
    cfg["stream"]["fps"] = 10
    pipe = egolearn.build(cfg)
    with pytest.raises(ValueError, match="expected_fps"):
        pipe.driver._build_index()
    cfg["stream"]["fps"] = 5
    cfg["stream"]["block_len"] = 1
    cfg["stream"]["blocks_per_session"] = 1
    pipe = egolearn.build(cfg)
    with pytest.raises(RuntimeError, match="dataset index"):
        pipe.driver.fit()
    image_dir = tmp_path / "images/session"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_dir / "invalid.jpg")
    with pytest.raises(ValueError, match="filename"):
        SaycamStream(str(image_dir.parent), strict_filenames=True).total_observations()


def test_intphys_default_uses_the_trained_token_and_layer_norm_targets(tmp_path):
    from egolearn.evaluation.voe import VoEProbe

    pipe = egolearn.build(config(tmp_path))
    pipe.losses[0].initialize_target(pipe.model)
    probe = VoEProbe("test", str(tmp_path), num_frames=4, context_frames=[2], size=64)
    clips = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        assert torch.isfinite(probe.surprise(pipe.model, clips)[2]).all()
    # No mask index is forwarded by the probe; VJEPA2Predictor defaults to trained token 0.


def test_runner_resume_preserves_baseline_ema_and_canonical_metrics(tmp_path, monkeypatch):
    reference = run(config(tmp_path / "reference"))
    cfg = config(tmp_path / "partial")
    cfg["hooks"]["snapshot_every"] = 2
    original_build = egolearn.build

    class Interrupt(Hook):
        def on_step(self, driver, logs):
            if driver.ctx.clock.step == 3:
                raise InterruptedError("synthetic interruption after latest snapshot")

    def interrupted_build(cfg):
        pipe = original_build(cfg)
        pipe.driver.hooks.append(Interrupt())
        return pipe

    with monkeypatch.context() as patch:
        patch.setattr(egolearn, "build", interrupted_build)
        with pytest.raises(InterruptedError):
            run(cfg)
    baseline = (tmp_path / "partial/random.pth").read_bytes()
    metrics = tmp_path / "partial/metrics.jsonl"
    # Interrupted JSONL appends must also be recoverable.
    with metrics.open("a") as file:
        file.write('{"step":')
    resumed = run({**cfg, "resume": "auto"})
    assert (tmp_path / "partial/random.pth").read_bytes() == baseline
    for name, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert [row["elapsed_s"] for row in rows] == sorted(row["elapsed_s"] for row in rows)
    assert len(list(metrics.parent.glob("metrics-abandoned-*.jsonl"))) == 1
    assert rows[0]["lr"] > 0


def test_runner_marks_wandb_failed_on_training_exception(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from egolearn.training.hooks import WandbHook

    cfg = config(tmp_path)
    pipe = egolearn.build(cfg)
    codes = []
    wandb = WandbHook(every=999)
    wandb._run = SimpleNamespace(finish=lambda exit_code: codes.append(exit_code))

    class Failure(Hook):
        def on_step(self, driver, logs):
            raise RuntimeError("synthetic training failure")

    pipe.driver.hooks.extend([wandb, Failure()])
    monkeypatch.setattr(egolearn, "build", lambda cfg: pipe)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run(cfg)
    assert codes == [1]


def test_preflight_uses_next_shuffled_batch_without_consuming_clock(tmp_path, monkeypatch):
    from egolearn.methods.vjepa2.train import _preflight_inputs

    pipe = egolearn.build(config(tmp_path))
    refs = pipe.driver._build_index()
    order = list(range(len(refs)))
    pipe.ctx.fresh_rng("offline.shuffle.epoch0").shuffle(order)
    expected = pipe.driver._pick(refs, order, 0)
    original_compose = pipe.driver._compose
    seen = []

    def compose(refs, seq=None):
        seen.extend(refs)
        batch = original_compose(refs, seq=seq)
        # Partial duplicates are measured; they do not make the whole clip static.
        batch["clips"]["views"][:, :, 1] = batch["clips"]["views"][:, :, 0]
        return batch

    monkeypatch.setattr(pipe.driver, "_compose", compose)
    result = _preflight_inputs(pipe)
    assert seen == expected
    assert result["jepa/identical_adjacent_fraction"] >= 1 / 3
    assert pipe.ctx.clock.step == pipe.ctx.clock.compose_seq == 0


@pytest.mark.parametrize("workers", [0, 2])
def test_babyview_training_with_strict_adapter_and_workers(tmp_path, workers):
    import numpy as np

    root = tmp_path / "data/child/session"
    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(16):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            root / f"frame_{i + 1:08d}.jpg"
        )
    cfg = config(tmp_path / "run")
    cfg["stream"] = dict(
        _target_="stream:babyview",
        root_dir=str(root.parents[1]),
        session_glob="*/*",
        strict_filenames=True,
        fps=5,
    )
    cfg["loader"]["workers"] = workers
    cfg["driver"]["prefetch"] = workers > 0
    cfg["driver"]["max_steps"] = 2
    pipe = run(cfg)
    assert pipe.ctx.clock.step == 2
    assert pipe.driver.index_stats["fps_source"] == "declared_frame_index_fps"


def test_metrics_resume_repairs_complete_row_without_final_newline(tmp_path):
    from egolearn.training.hooks import MetricsHook

    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 2, "elapsed_s": 3.0}))
    hook = MetricsHook(str(tmp_path))
    hook._prepare_history(next_step=3)
    assert path.read_text().endswith("\n")
    assert hook.elapsed_offset == 3.0
    assert len(list(tmp_path.glob("metrics-abandoned-*.jsonl"))) == 1
