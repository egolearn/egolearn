"""Video pretraining: the V-JEPA path (multiblock masks, video ViT with
rotary positions, predictor, EMA-target loss) and offline clip batches."""

import random

import pytest
import torch

import egolearn


# ---------------------------------------------------------------------------
# components in isolation
# ---------------------------------------------------------------------------
def test_multiblock_masks_split_the_grid():
    from egolearn.model.masking import multiblock

    grid = (2, 4, 4)
    ctx, tgt = multiblock(grid, 3, num_blocks=2, scale=(0.2, 0.2), rng=random.Random(5))
    ctx2, tgt2 = multiblock(grid, 3, num_blocks=2, scale=(0.2, 0.2), rng=random.Random(5))
    # same rng, same masks
    assert torch.equal(ctx, ctx2) and torch.equal(tgt, tgt2)
    n_tokens = 2 * 4 * 4
    for b in range(3):
        c, t = set(ctx[b].tolist()), set(tgt[b].tolist())
        assert c and t and not (c & t)  # both non-empty, disjoint
        assert all(0 <= i < n_tokens for i in c | t)

    # a single block: every sample masks the same COUNT, so nothing is
    # trimmed, and the mask covers the full clip duration (both timesteps
    # hold the same spatial positions)
    _, tgt1 = multiblock(
        grid, 2, num_blocks=1, scale=(0.25, 0.25), aspect=(1.0, 1.0), rng=random.Random(0)
    )
    for b in range(2):
        spatial = {i % 16 for i in tgt1[b].tolist()}
        assert tgt1.size(1) == 2 * len(spatial)


def test_video_vit_tokens_positions_and_visible_forward():
    from egolearn.model.encoders.video import VideoViTEncoder

    enc = VideoViTEncoder(dim=32, depth=2, heads=2, patch=16, tubelet=2)
    enc.eval()
    x = torch.rand(2, 4, 3, 32, 32)  # -> token grid (2, 2, 2), N = 8

    full = enc(x)
    assert full.tokens is not None and full.tokens.shape == (2, 8, 32)
    assert full.pooled.shape == (2, 32)
    assert full.aux["grid"].tolist() == [2, 2, 2]

    # visible = every token is the same computation as no mask at all
    all_idx = torch.arange(8).expand(2, -1)
    again = enc(x, visible=all_idx)
    assert torch.allclose(full.tokens, again.tokens, atol=1e-6)

    # a real mask: only the kept tokens go through attention
    kept = torch.tensor([[0, 3, 5], [1, 2, 7]])
    part = enc(x, visible=kept)
    assert part.tokens is not None and part.tokens.shape == (2, 3, 32)

    with pytest.raises(ValueError, match="divisible"):
        enc(torch.rand(2, 5, 3, 32, 32))  # T=5 does not divide tubelet 2


def test_vjepa_predictor_fills_target_positions():
    from egolearn.model.predictors import VJEPAPredictor

    pred = VJEPAPredictor(encoder_dim=32, dim=16, depth=1, heads=2)
    tokens = torch.rand(2, 5, 32)
    ctx_pos = torch.arange(5).expand(2, -1)
    tgt_pos = torch.tensor([[5, 6, 7], [5, 6, 7]])
    out = pred(tokens, ctx_pos=ctx_pos, tgt_pos=tgt_pos, grid=(2, 2, 2))
    assert out.shape == (2, 3, 32)  # predictions back at encoder width


def test_own_forward_slots_spares_shared_consumers():
    from egolearn.model.losses.base import Loss, own_forward_slots

    class Own(Loss):
        own_forward = True

    class Shared(Loss):
        pass

    a, b = Own("a"), Shared("b")
    a.requires_slots, b.requires_slots = ["clips"], ["frames"]
    assert own_forward_slots([a, b]) == frozenset({"clips"})
    # a slot ALSO read by a standard loss keeps its precomputed feats
    b.requires_slots = ["clips", "frames"]
    assert own_forward_slots([a, b]) == frozenset()


# ---------------------------------------------------------------------------
# the full V-JEPA composition, streaming and offline
# ---------------------------------------------------------------------------
def _vjepa_model():
    return {
        "encoders": {
            "video": {
                "_target_": "encoder:video_vit",
                "dim": 32,
                "depth": 2,
                "heads": 2,
                "patch": 16,
                "tubelet": 2,
            },
            "video_target": {
                "_target_": "encoder:video_vit",
                "dim": 32,
                "depth": 2,
                "heads": 2,
                "patch": 16,
                "tubelet": 2,
                "frozen": True,
            },
        },
        "predictors": {
            "pred": {
                "inputs": ["video"],
                "_target_": "predictor:vjepa",
                "encoder_dim": 32,
                "dim": 16,
                "depth": 1,
                "heads": 2,
            }
        },
    }


def _vjepa_losses():
    return [
        {
            "_target_": "loss:vjepa",
            "name": "jepa",
            "slot": "clips",
            "context": "video",
            "target": "video_target",
            "predictor": "pred",
            "masks": {
                "short": {"num_blocks": 2, "scale": [0.3, 0.3]},
                "long": {"num_blocks": 1, "scale": [0.6, 0.6]},
            },
        }
    ]


def _vjepa_cfg(tmp_path):
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
        "views": {"cv": {"name": "clip_rrc", "size": 48, "scale_min": 0.5}},
        "admission": [{"_target_": "admission:frame_emitter", "subsample": 1}],
        "memories": {
            "frames": {
                "tiers": {"short": {"capacity": 30, "update": "fifo"}},
                "admit_to": ["short"],
                "overflow": {},
            }
        },
        "routes": {"RawFrame": "frames"},
        "recipe": {
            "slots": [
                {
                    "name": "clips",
                    "source": {"kind": "tier", "memory": "frames", "tier": "short"},
                    "n": 2,
                    "unit": "clip",
                    "clip_len": 4,
                    "clip_stride": 1,
                    "views": ["cv"],
                    "strict": True,
                }
            ]
        },
        "model": _vjepa_model(),
        "losses": _vjepa_losses(),
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


def test_vjepa_trains_and_ema_target_tracks(tmp_path):
    pipe = egolearn.build(_vjepa_cfg(tmp_path))
    assert {"video_representation", "prediction"} <= pipe.model.capabilities()
    # the jepa slot is skipped by the shared pooled precompute
    assert pipe.driver._skip_feats == frozenset({"clips"})

    tgt0 = {k: v.clone() for k, v in pipe.model.encoders["video_target"].state_dict().items()}
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2  # every step produced a finite loss

    tgt = pipe.model.encoders["video_target"].state_dict()
    src = pipe.model.encoders["video"].state_dict()
    # step 0 replaced the target's random init with a copy of the context...
    assert any(not torch.equal(tgt[k], tgt0[k]) for k in tgt)
    # ...and the EMA keeps it near-but-not-equal to the trained context
    assert any(not torch.equal(tgt[k], src[k]) for k in tgt)


def test_vjepa_same_seed_twice_gives_identical_weights(tmp_path):
    states = []
    for _ in range(2):
        pipe = egolearn.build(_vjepa_cfg(tmp_path))
        pipe.driver.fit()
        states.append({k: v.clone() for k, v in pipe.model.state_dict().items()})
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), f"weights differ at {k}"


def test_vjepa_resume_matches_uninterrupted(tmp_path):
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    cfg = _vjepa_cfg(tmp_path)
    cfg["driver"]["max_steps"] = 4
    ref = egolearn.build(cfg)
    ref.driver.fit()

    p1 = egolearn.build(_vjepa_cfg(tmp_path))
    p1.driver.fit()  # max_steps: 2
    snap = str(tmp_path / "vjepa_snap.pth")
    save_snapshot(snap, p1.driver)

    cfg2 = _vjepa_cfg(tmp_path)
    cfg2["driver"]["max_steps"] = 4
    p2 = egolearn.build(cfg2)
    load_snapshot(snap, p2.driver)
    p2.driver.fit()

    resumed = p2.model.state_dict()
    for k, v in ref.model.state_dict().items():
        assert torch.equal(v, resumed[k]), f"resume diverged at {k} (EMA target included)"


def test_vjepa_target_must_be_frozen(tmp_path):
    cfg = _vjepa_cfg(tmp_path)
    # an existing but trainable encoder cannot be the EMA target: the loss
    # would update it outside the optimizer while the optimizer trains it
    cfg["model"]["encoders"]["bad_target"] = {"_target_": "encoder:toy_cnn", "dim": 32}
    cfg["losses"][0]["target"] = "bad_target"
    with pytest.raises(egolearn.PipelineError, match="frozen"):
        egolearn.build(cfg)


def _offline_vjepa_cfg(tmp_path):
    cfg = _vjepa_cfg(tmp_path)
    for key in ("admission", "memories", "routes", "recipe"):
        del cfg[key]
    cfg["driver"] = {
        "mode": "offline",
        "batch": {
            "slot": "clips",
            "n": 2,
            "views": ["cv"],
            "unit": "clip",
            "clip_len": 4,
            "clip_stride": 1,
        },
        "epochs": 1,
        "max_steps": 2,
    }
    return cfg


def test_offline_clip_unit_indexes_windows_and_trains(tmp_path):
    pipe = egolearn.build(_offline_vjepa_cfg(tmp_path))
    refs = pipe.driver._build_index()
    # 160 toy frames -> 40 non-overlapping 4-frame windows
    assert len(refs) == 40
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2
    rep = pipe.driver.timer.report()
    assert "batch.compose" in rep and rep["model.forward"]["count"] == 2


def test_clip_worker_pool_matches_serial(tmp_path):
    # clips travel through the worker pool like frames; the worker count
    # stays a pure throughput knob for them too
    def first_batch(workers):
        cfg = _vjepa_cfg(tmp_path / f"w{workers}")
        cfg["loader"] = {"workers": workers}
        pipe = egolearn.build(cfg)
        pipe.driver._run_warmup()
        batch = pipe.composer.compose(pipe.recipe, pipe.memories, {}, pipe.ctx)
        pipe.loader.close()
        assert batch is not None
        return batch

    serial, pooled = first_batch(0), first_batch(2)
    assert torch.equal(serial["clips"]["views"], pooled["clips"]["views"])


def test_offline_clip_unit_needs_a_video_encoder(tmp_path):
    cfg = _offline_vjepa_cfg(tmp_path)
    cfg["model"] = {"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 32}}}
    cfg["losses"] = []
    with pytest.raises(egolearn.PipelineError, match="video"):
        egolearn.build(cfg)
