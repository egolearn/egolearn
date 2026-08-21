"""Core contract tests (no real data, no network, no GPU).

Run: pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import egolearn
from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.memory.labels import LabelStore
from egolearn.memory.table import TieredMemory


# ---------------------------------------------------------------------------
def test_key_registry_rejects_unknown_key():
    with pytest.raises(K.KeyFlowError):
        K.require_keys({"time.t0": 0.0}, [K.VIDEO_REF], "test-component")


def test_label_store_revision_and_merge():
    store = LabelStore("t")
    store.assign(["a", "b", "c"], [0, 1, 2])
    assert store.resolve("b") == 1
    store.assign(["b"], [5])  # retroactive relabel (MS recluster)
    assert store.resolve("b") == 5
    store.remap(5, 0)  # class merge (MS prototypes)
    assert store.resolve("b") == 0
    store.remap(2, 5)  # chained redirect resolves fully
    assert store.resolve("c") == 0


def test_two_tier_spill_vs_parallel_admission():
    ctx = Context(seed=0, device="cpu")
    spill = TieredMemory(
        "s",
        tiers={
            "short": {"capacity": 2, "update": "fifo"},
            "long": {"capacity": 8, "update": "reservoir"},
        },
        admit_to=["short"],
        overflow={"short": "long"},
    )
    for i in range(5):
        spill.admit({"i": i}, ctx)
    assert len(spill.tiers["short"]) == 2  # newest two
    assert len(spill.tiers["long"]) == 3  # spilled oldest three

    par = TieredMemory(
        "p",
        tiers={
            "short": {"capacity": 2, "update": "fifo"},
            "long": {"capacity": 8, "update": "reservoir_weighted"},
        },
        admit_to=["short", "long"],
        overflow={},
    )
    for i in range(5):
        par.admit({"i": i}, ctx)
    assert len(par.tiers["short"]) == 2
    assert len(par.tiers["long"]) == 5  # long saw every item (MS)


def test_mix_sampling_backfills_deficit():
    ctx = Context(seed=0, device="cpu")
    mem = TieredMemory(
        "m",
        tiers={
            "short": {"capacity": 10, "update": "fifo"},
            "long": {"capacity": 10, "update": "fifo"},
        },
        admit_to=["short"],
        overflow={"short": "long"},
    )
    for i in range(6):  # long stays empty
        mem.admit({"i": i}, ctx)
    got = mem.sample_mix(4, {"short": 0.25, "long": 0.75}, ctx)
    assert len(got) == 4  # deficit backfilled from short


def test_ms_greedy_clustering_finds_block_boundary():
    # verbatim MS port (egolearn.methods.storyboard): two coherent blocks
    # of 10 -> the single forced cut must land at 10
    from egolearn.methods.storyboard import greedy_hierarchical_clustering

    a = torch.ones(10, 10)
    z = torch.zeros(10, 10)
    sim = torch.cat([torch.cat([a, z], 1), torch.cat([z, a], 1)], 0)
    cps = greedy_hierarchical_clustering(sim, num_clusters=2)
    assert cps == [10]


def test_project_imports_register_components():
    # the config `_imports_` mechanism: importing a project module makes its
    # method-novel components addressable through the shared registries
    import importlib

    importlib.import_module("egolearn.methods.babycl")
    from egolearn.config.registry import ADMISSION

    assert ADMISSION.get("babycl_online_segmenter") is not None
    assert ADMISSION.get("an_chunk_emitter") is not None


def test_poodle_stub_raises_helpfully():
    from egolearn.config.registry import DECODERS

    with pytest.raises(NotImplementedError, match="Poodle"):
        DECODERS.get("poodle")()


def test_clip_unit_loads_contiguous_consistent_clips():
    # clip data unit: [B, V, T, C, H, W], one aug-parameter draw per clip
    import random

    from egolearn.core.refs import ItemRef
    from egolearn.data.loading import Loader
    from egolearn.data.sources.toy import ToyStream
    from egolearn.data.views import ViewBank

    stream = ToyStream(n_sessions=1, blocks_per_session=2, block_len=20, img_size=64)
    views = ViewBank(
        {
            "c1": {"name": "clip_simclr", "size": 48, "view_idx": 0},
            "c2": {"name": "clip_simclr", "size": 48, "view_idx": 1},
            "ce": {"name": "clip_center_crop", "size": 48},
        }
    )
    loader = Loader(stream, views)
    ref = ItemRef("toy_00", "video", 0.0, 7.8, ("toy_00", 0, 40))  # 40-frame span

    out = loader.clip_views(ref, ["c1", "c2"], clip_len=8, stride=2, rng=random.Random(0))
    assert out.shape == (2, 8, 3, 48, 48)  # [V, T, C, H, W]

    # deterministic view: consecutive clip frames differ only by toy noise,
    # and the SAME center-crop is applied to all of them
    ev = loader.clip_views(ref, ["ce"], clip_len=4, stride=1, rng=random.Random(1))
    assert ev.shape == (1, 4, 3, 48, 48)
    assert torch.allclose(ev[0, 0], ev[0, 1], atol=0.5)  # same block, same crop

    # short ref: indices clamp to the last frame (pad-by-repetition)
    short = ItemRef("toy_00", "video", 0.0, 0.4, ("toy_00", 0, 3))
    sv = loader.clip_views(short, ["ce"], clip_len=8, stride=1, rng=random.Random(2))
    assert sv.shape == (1, 8, 3, 48, 48)
    assert torch.equal(sv[0, 2], sv[0, 7])  # clamped repeats


def test_clip_slot_composes_through_recipe(tmp_path):
    # unit=clip flows through memory -> recipe -> composer -> [B,V,T,C,H,W]
    cfg = _toy_cfg(tmp_path)
    cfg["views"]["cv"] = {"name": "clip_simclr", "size": 48, "view_idx": 0}
    cfg["recipe"]["slots"] = [
        {
            "name": "clips",
            "source": {"kind": "tier", "memory": "frames", "tier": "short"},
            "n": 2,
            "unit": "clip",
            "clip_len": 4,
            "clip_stride": 1,
            "views": ["cv"],
            "labels_from": "fixed",
            "strict": True,
        },
    ]
    cfg["losses"] = []  # loading-side support only; no clip loss yet
    pipe = egolearn.build(cfg)
    pipe.driver._run_warmup()
    batch = pipe.composer.compose(pipe.recipe, pipe.memories, {}, pipe.ctx)
    assert batch is not None
    assert batch["clips"]["views"].shape == (2, 1, 4, 3, 48, 48)
    assert batch["clips"]["labels"].shape == (2,)


# ---------------------------------------------------------------------------
def _toy_cfg(tmp_path):
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
            "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 64}},
            "heads": {
                "proj": {"encoder": "vision", "_target_": "head:mlp", "sizes": [64, 64, 32]},
                "cls": {"encoder": "vision", "_target_": "head:mlp", "sizes": [64, 64, 8]},
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
            "groups": [{"match": ".*", "lr": 0.01, "weight_decay": 0.0}],
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


def test_toy_pipeline_builds_and_composes(tmp_path):
    pipe = egolearn.build(_toy_cfg(tmp_path))
    drv = pipe.driver
    drv._run_warmup()
    assert len(pipe.memories["frames"].tiers["short"]) > 0
    batch = pipe.composer.compose(pipe.recipe, pipe.memories, {}, pipe.ctx)
    assert batch is not None
    views = batch["curr"]["views"]
    assert views.shape[0] == 4 and views.shape[1] == 2  # [B, V, C, H, W]
    assert batch["curr"]["labels"].shape == (4,)


def test_toy_pipeline_two_training_steps(tmp_path):
    pipe = egolearn.build(_toy_cfg(tmp_path))
    pipe.driver.fit()  # max_steps: 2
    assert pipe.ctx.clock.step == 2
    # the latency ledger measured every stage of those steps
    rep = pipe.driver.timer.report()
    assert "batch.compose" in rep and "data.load" in rep
    assert rep["model.forward"]["count"] == 2


def test_validation_rejects_unknown_slot(tmp_path):
    cfg = _toy_cfg(tmp_path)
    cfg["losses"][0]["slots"] = ["nonexistent"]
    with pytest.raises(egolearn.PipelineError, match="nonexistent"):
        egolearn.build(cfg)


def test_validation_rejects_unknown_view(tmp_path):
    cfg = _toy_cfg(tmp_path)
    cfg["recipe"]["slots"][0]["views"] = ["not_a_view", "v2"]
    with pytest.raises(egolearn.PipelineError, match="not_a_view"):
        egolearn.build(cfg)


def test_validation_rejects_text_loss_without_text_encoder(tmp_path):
    cfg = _toy_cfg(tmp_path)
    cfg["recipe"]["slots"][0]["strict"] = False
    cfg["losses"].append(
        {
            "_target_": "loss:clip_infonce",
            "name": "xm",
            "slot": "curr",
            "vision_head": "proj",
            "weight": 1.0,
        }
    )
    with pytest.raises(egolearn.PipelineError, match="text encoder"):
        egolearn.build(cfg)


# ---------------------------------------------------------------------------
# offline (IID) mode -- the parallel loop: no memory, no admission
# ---------------------------------------------------------------------------
def _offline_cfg(tmp_path):
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
        "model": {
            "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 64}},
            "heads": {"proj": {"encoder": "vision", "_target_": "head:mlp", "sizes": [64, 64, 32]}},
        },
        "losses": [
            {
                "_target_": "loss:ntxent",
                "name": "inst",
                "slots": ["frames"],
                "head": "proj",
                "temperature": 0.1,
                "weight": 1.0,
            },
        ],
        "solver": {
            "opt": "sgd",
            "groups": [{"match": ".*", "lr": 0.01, "weight_decay": 0.0}],
            "schedule": {"constant": True},
        },
        "driver": {
            "mode": "offline",
            "batch": {"slot": "frames", "n": 8, "views": ["v1", "v2"]},
            "epochs": 1,
            "max_steps": 3,
        },
        "hooks": {"log_every": 1},
    }


def test_offline_mode_trains(tmp_path):
    pipe = egolearn.build(_offline_cfg(tmp_path))
    assert pipe.mode == "offline"
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 3  # max_steps wins
    assert pipe.ctx.clock.progress == pytest.approx(3 / 3)
    rep = pipe.driver.timer.report()
    assert "batch.compose" in rep and "model.forward" in rep


def test_offline_mode_is_deterministic_under_seed(tmp_path):
    idx = []
    for _ in range(2):
        pipe = egolearn.build(_offline_cfg(tmp_path))
        refs = pipe.driver._build_index()
        order = list(range(len(refs)))
        pipe.ctx.fresh_rng("offline.shuffle.epoch0").shuffle(order)
        idx.append(order[:10])
    assert idx[0] == idx[1]
    # fresh_rng is one-shot derived randomness: asking again regenerates the
    # SAME order (a cached rng would have advanced) -- this is what keeps a
    # resumed run's epoch shuffle identical to the original's
    pipe = egolearn.build(_offline_cfg(tmp_path))
    a = list(range(50))
    b = list(range(50))
    pipe.ctx.fresh_rng("offline.shuffle.epoch0").shuffle(a)
    pipe.ctx.fresh_rng("offline.shuffle.epoch0").shuffle(b)
    assert a == b


def test_offline_resume_continues_from_snapshot(tmp_path):
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    # run 3 of 5 planned steps, snapshot, resume in a FRESH pipeline
    cfg = _offline_cfg(tmp_path)
    cfg["driver"]["max_steps"] = 3
    pipe1 = egolearn.build(cfg)
    pipe1.driver.fit()
    snap = str(tmp_path / "snap.pth")
    save_snapshot(snap, pipe1.driver)

    cfg2 = _offline_cfg(tmp_path)
    cfg2["driver"]["max_steps"] = 5
    pipe2 = egolearn.build(cfg2)
    load_snapshot(snap, pipe2.driver)
    assert pipe2.ctx.clock.step == 3
    pipe2.driver.fit()  # continues 3 -> 5
    assert pipe2.ctx.clock.step == 5


def test_streaming_resume_replay_rebuilds_state(tmp_path):
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    # train 2 steps, snapshot; fresh pipeline resumes: memories identical,
    # admission state rebuilt by replay, training continues
    pipe1 = egolearn.build(_toy_cfg(tmp_path))
    pipe1.driver.fit()  # max_steps: 2
    admitted = pipe1.ctx.clock.admitted
    mem_sizes = {t: len(tab) for t, tab in pipe1.memories["frames"].tiers.items()}
    snap = str(tmp_path / "snap.pth")
    save_snapshot(snap, pipe1.driver)

    cfg2 = _toy_cfg(tmp_path)
    cfg2["driver"]["max_steps"] = 4
    pipe2 = egolearn.build(cfg2)
    load_snapshot(snap, pipe2.driver)
    assert pipe2.ctx.clock.step == 2
    assert pipe2.ctx.clock.admitted == admitted  # replay hit the mark
    assert {
        t: len(tab) for t, tab in pipe2.memories["frames"].tiers.items()
    } == mem_sizes  # memories from snapshot
    # labeler state rebuilt: composing a batch resolves labels without error
    pipe2.driver.fit()  # continues 2 -> 4
    assert pipe2.ctx.clock.step == 4


def test_offline_mode_rejects_memory_config(tmp_path):
    cfg = _offline_cfg(tmp_path)
    cfg["memories"] = {
        "frames": {
            "tiers": {"short": {"capacity": 8, "update": "fifo"}},
            "admit_to": ["short"],
            "overflow": {},
        }
    }
    with pytest.raises(egolearn.PipelineError, match=r"memory-free|no.*memory|memories"):
        egolearn.build(cfg)


def test_offline_mode_rejects_unknown_view(tmp_path):
    cfg = _offline_cfg(tmp_path)
    cfg["driver"]["batch"]["views"] = ["v1", "nope"]
    with pytest.raises(egolearn.PipelineError, match="nope"):
        egolearn.build(cfg)


# ---------------------------------------------------------------------------
# composed model: encoders x heads x predictors, capabilities, eval compat
# ---------------------------------------------------------------------------
def test_composed_model_two_encoders_and_contract():
    import torch.nn as nn

    from egolearn.model.encoders.image import ToyCNNEncoder
    from egolearn.model.model import ComposedModel

    class TinyText(nn.Module):
        modality = "text"
        out_dim = 8

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 8, padding_idx=0)

        def forward(self, ids, lengths):
            return self.emb(ids).sum(1) / lengths.clamp(min=1).unsqueeze(1).float()

    model = ComposedModel(
        {"vision": ToyCNNEncoder(dim=16), "text": TinyText()},
        heads={"proj": {"encoder": "vision", "module": nn.Linear(16, 4)}},
    )
    assert model.head_encoder("proj") == "vision"
    assert model.head_field("proj") == "pooled"
    caps = model.capabilities()
    assert {"image_representation", "text_representation", "multimodal"} <= caps

    x = torch.rand(2, 3, 32, 32)
    ids = torch.ones(2, 5, dtype=torch.long)
    lens = torch.full((2,), 5, dtype=torch.long)
    # by config name and by modality
    assert model.encode("vision", x).pooled.shape == (2, 16)
    assert torch.equal(model.encode("image", x).pooled, model.encode("vision", x).pooled)
    assert model.encode("text", ids, lens).pooled.shape == (2, 8)
    assert model.apply_head("proj", model.encode("vision", x).pooled).shape == (2, 4)

    contract = model.export_contract()
    assert contract["encoders"] == {"vision": "image", "text": "text"}
    assert "multimodal" in contract["capabilities"]

    # a slot with views must say what they are; no guessing from tensor shape
    import pytest

    with pytest.raises(KeyError, match="declares no modality"):
        model.precompute_feats({"s": {"views": x.unsqueeze(1)}}, torch.device("cpu"))


def test_video_encoder_trains_on_clip_slots(tmp_path):
    # video SSL expressed purely in config: clip unit -> video encoder ->
    # ntxent over the two clip views, end to end through the driver
    cfg = _toy_cfg(tmp_path)
    cfg["views"] = {
        "cv1": {"name": "clip_simclr", "size": 48, "view_idx": 0},
        "cv2": {"name": "clip_simclr", "size": 48, "view_idx": 1},
    }
    cfg["recipe"]["slots"] = [
        {
            "name": "clips",
            "source": {"kind": "tier", "memory": "frames", "tier": "short"},
            "n": 3,
            "unit": "clip",
            "clip_len": 4,
            "clip_stride": 1,
            "views": ["cv1", "cv2"],
            "strict": True,
        },
    ]
    cfg["model"] = {
        "encoders": {"video": {"_target_": "encoder:toy_video", "dim": 32}},
        "heads": {"proj": {"encoder": "video", "_target_": "head:mlp", "sizes": [32, 32, 16]}},
    }
    cfg["losses"] = [
        {
            "_target_": "loss:ntxent",
            "name": "inst",
            "slots": ["clips"],
            "head": "proj",
            "temperature": 0.2,
            "weight": 1.0,
        }
    ]
    pipe = egolearn.build(cfg)
    assert "video_representation" in pipe.model.capabilities()
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2  # max_steps from _toy_cfg


def test_predictor_wiring_and_validation(tmp_path):
    import pytest

    cfg = _toy_cfg(tmp_path)
    cfg["model"]["predictors"] = {
        "pred": {"inputs": ["vision"], "_target_": "predictor:mlp", "sizes": [64, 64, 64]}
    }
    pipe = egolearn.build(cfg)
    assert "prediction" in pipe.model.capabilities()
    out = pipe.model.encode("image", torch.rand(2, 3, 64, 64))
    assert pipe.model.predict("pred", out.pooled).shape == (2, 64)
    # heads and predictors eat tensors; the field choice stays at the call site
    with pytest.raises(TypeError, match="pick a field"):
        pipe.model.predict("pred", out)

    cfg2 = _toy_cfg(tmp_path)
    cfg2["model"]["predictors"] = {
        "pred": {"inputs": ["video"], "_target_": "predictor:mlp", "sizes": [64, 64]}
    }
    with pytest.raises(egolearn.PipelineError, match=r"predictor .* reads encoder"):
        egolearn.build(cfg2)


def test_probe_capability_mismatch_fails_at_build(tmp_path):
    import pytest

    from egolearn.config.registry import PROBES
    from egolearn.evaluation.probe import Probe

    if "needs_text_probe" not in PROBES.names():

        @PROBES.register("needs_text_probe")
        class NeedsTextProbe(Probe):
            name = "needs_text"
            requires_capabilities = ("text_representation",)

            def __init__(self, **kw):
                pass

            def run(self, model, ctx):
                return {}

    cfg = _toy_cfg(tmp_path)  # image-only model
    cfg["probes"] = [{"_target_": "probe:needs_text_probe"}]
    with pytest.raises(egolearn.PipelineError, match="needs capability"):
        egolearn.build(cfg)


def test_frozen_twin_encoder_stays_out_of_the_way(tmp_path):
    # the JEPA EMA-target pattern: a frozen copy of an encoder is carried by
    # the model (so it snapshots) but is excluded from the optimizer and
    # never grabs modality-based routing
    cfg = _toy_cfg(tmp_path)
    cfg["model"]["encoders"] = {
        "vision": {"_target_": "encoder:toy_cnn", "dim": 64},
        "vision_target": {"_target_": "encoder:toy_cnn", "dim": 64, "frozen": True},
    }
    pipe = egolearn.build(cfg)
    tgt = pipe.model.encoders["vision_target"]
    assert not any(p.requires_grad for p in tgt.parameters())
    opt_params = {id(p) for g in pipe.driver.optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in opt_params for p in tgt.parameters())
    # modality routing resolves to the trainable twin
    assert pipe.model.encoder_for("image") is pipe.model.encoders["vision"]
    # train() must not pull the frozen twin out of eval mode (BatchNorm
    # running stats would drift otherwise)
    pipe.model.train()
    assert not tgt.training
    # routing is a construction-time fact: freezing everything for a linear
    # probe must not change which encoder answers for "image"
    pipe.model.requires_grad_(False)
    assert pipe.model.encoder_for("image") is pipe.model.encoders["vision"]
    # ...and a snapshot still carries the frozen weights
    assert any(k.startswith("encoders.vision_target") for k in pipe.model.state_dict())


def test_encoder_output_fields_are_additive():
    # an encoder may return more than the pooled vector (token sequences,
    # spatial maps, extras); pooled-only consumers keep working unchanged
    # and richer consumers name the field they want
    import pytest
    import torch.nn as nn

    from egolearn.model import EncoderOutput
    from egolearn.model.model import ComposedModel

    class RichEncoder(nn.Module):
        modality = "image"
        out_dim = 16

        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, 16, kernel_size=8, stride=8)

        def forward(self, x):
            spatial = self.proj(x)  # [B, 16, H/8, W/8]
            tokens = spatial.flatten(2).transpose(1, 2)  # [B, N, 16]
            return EncoderOutput(
                pooled=tokens.mean(dim=1),
                tokens=tokens,
                spatial=spatial,
                aux={"logit_scale": torch.tensor(4.6)},
            )

    model = ComposedModel(
        {"vision": RichEncoder()},
        heads={"proj": {"encoder": "vision", "module": nn.Linear(16, 4)}},
    )
    x = torch.rand(2, 3, 32, 32)

    out = model.encode("vision", x)
    assert out.pooled.shape == (2, 16)
    assert out.tokens is not None and out.tokens.shape == (2, 16, 16)
    assert out.spatial is not None and out.spatial.shape == (2, 16, 4, 4)
    assert "logit_scale" in out.aux

    # pooled shortcut and shared-feature path see only the pooled field
    batch: dict = {"s": {"views": x.unsqueeze(1), "modality": "image"}}
    model.precompute_feats(batch, torch.device("cpu"))
    feats = batch["s"]["feats"]
    assert isinstance(feats, torch.Tensor)
    assert torch.allclose(feats[:, 0], out.pooled)

    # handing the whole wrapper to a head is refused with a clear message
    # (cast: this call is wrong on purpose)
    from typing import cast

    with pytest.raises(TypeError, match="pick a field"):
        model.apply_head("proj", cast(torch.Tensor, out))
    assert model.apply_head("proj", out.pooled).shape == (2, 4)


def test_head_unknown_encoder_fails_at_build(tmp_path):
    # a head reading a nonexistent encoder is caught even when no loss
    # requires that head (probe-only heads used to slip through)
    import pytest

    cfg = _toy_cfg(tmp_path)
    cfg["model"]["heads"]["orphan"] = {
        "encoder": "backbone",  # no encoder has this name
        "_target_": "head:mlp",
        "sizes": [64, 32, 16],
    }
    with pytest.raises(egolearn.PipelineError, match="reads encoder"):
        egolearn.build(cfg)


def test_slot_modality_without_encoder_warns(tmp_path):
    # a clip slot on an image-only model: no video encoder will ever fill
    # slot["feats"], so the build surfaces a warning instead of leaving the
    # mismatch to explode at the first batch
    cfg = _toy_cfg(tmp_path)
    cfg["views"]["cv"] = {"name": "clip_simclr", "size": 48, "view_idx": 0}
    cfg["recipe"]["slots"].append(
        {
            "name": "clips",
            "source": {"kind": "tier", "memory": "frames", "tier": "short"},
            "n": 2,
            "unit": "clip",
            "clip_len": 4,
            "views": ["cv"],
            "strict": False,
        }
    )
    pipe = egolearn.build(cfg)
    assert any("clips" in w and "video" in w for w in pipe.warnings)


def test_probes_null_key_means_no_probes(tmp_path):
    # `probes:` present but empty (yaml null) disables evaluation, as before
    cfg = _toy_cfg(tmp_path)
    cfg["probes"] = None
    pipe = egolearn.build(cfg)
    assert pipe.probes == []


def test_two_trainable_same_modality_encoders_rejected_at_build(tmp_path):
    import pytest

    cfg = _toy_cfg(tmp_path)
    cfg["model"]["encoders"] = {
        "vision_a": {"_target_": "encoder:toy_cnn", "dim": 64},
        "vision_b": {"_target_": "encoder:toy_cnn", "dim": 64},
    }
    with pytest.raises(egolearn.PipelineError, match="compete for modality"):
        egolearn.build(cfg)


def test_encode_rejects_unknown_names():
    import pytest

    from egolearn.model.encoders.image import ToyCNNEncoder
    from egolearn.model.model import ComposedModel

    model = ComposedModel({"vision": ToyCNNEncoder(dim=16)})
    x = torch.rand(2, 3, 32, 32)
    assert model.encode("vision", x).pooled.shape == (2, 16)
    with pytest.raises(KeyError, match="no encoder named"):
        model.encode("text", torch.ones(2, 5, dtype=torch.long), torch.full((2,), 5))


def test_solver_warns_on_group_matching_nothing(tmp_path):
    cfg = _toy_cfg(tmp_path)
    cfg["solver"]["groups"] = [
        {"match": r"^backbone\.", "lr": 1e-3},  # matches nothing: params are encoders.*
        {"match": ".*", "lr": 1e-4},
    ]
    pipe = egolearn.build(cfg)
    assert any("matched no parameters" in w for w in pipe.warnings)


# ---------------------------------------------------------------------------
# determinism: the three guarantees the framework exists to provide
# ---------------------------------------------------------------------------
def test_same_seed_twice_gives_identical_weights(tmp_path):
    # weight init, augmentations and sampling all pin to the configured seed,
    # so two full runs from the same config end bit-identical (BN running
    # statistics included)
    states = []
    for _ in range(2):
        pipe = egolearn.build(_toy_cfg(tmp_path))
        pipe.driver.fit()  # max_steps: 2
        states.append({k: v.clone() for k, v in pipe.model.state_dict().items()})
    assert states[0].keys() == states[1].keys()
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), f"weights differ at {k}"


def test_resumed_run_matches_uninterrupted(tmp_path):
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    # straight run to 4 steps
    cfg = _toy_cfg(tmp_path)
    cfg["driver"]["max_steps"] = 4
    ref = egolearn.build(cfg)
    ref.driver.fit()

    # same run interrupted at 2, snapshotted, resumed in a fresh process
    # stand-in (fresh build), continued to 4
    p1 = egolearn.build(_toy_cfg(tmp_path))
    p1.driver.fit()  # max_steps: 2
    snap = str(tmp_path / "det_snap.pth")
    save_snapshot(snap, p1.driver)

    cfg2 = _toy_cfg(tmp_path)
    cfg2["driver"]["max_steps"] = 4
    p2 = egolearn.build(cfg2)
    load_snapshot(snap, p2.driver)
    p2.driver.fit()

    resumed = p2.model.state_dict()
    for k, v in ref.model.state_dict().items():
        assert torch.equal(v, resumed[k]), f"resume diverged at {k}"


def test_stream_tail_trains_when_ending_mid_quantum(tmp_path):
    # the toy stream has 160 observations and warmup consumes 20; with an
    # advance quantum larger than the rest of the stream, the final partial
    # quantum must still get its training steps
    cfg = _toy_cfg(tmp_path)
    cfg["driver"]["advance"] = {"kind": "observations", "n": 100000}
    cfg["driver"]["max_steps"] = None
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    assert pipe.ctx.clock.step >= 1  # the tail trained
    assert pipe.driver._exhausted  # and the whole stream was consumed


def test_same_timestamp_observations_keep_channel_order():
    # ties at equal media time surface in channel declaration order: the
    # frame comes before the utterance that ends at the same instant
    from egolearn.data.sources.base import DenseFrameChannel, Session, SparseIntervalChannel

    video = DenseFrameChannel("video", "s", times=[0.0, 1.0], locators=[("f", 0, 1), ("f", 1, 2)])
    text = SparseIntervalChannel("transcript", "s", items=[{"t0": 0.5, "t1": 1.0, "payload": {}}])
    sess = Session("s", {"video": video, "transcript": text})
    seq = [(o.channel, o.t) for o in sess.observations()]
    assert seq == [("video", 0.0), ("video", 1.0), ("transcript", 1.0)]


def test_final_partial_grad_accumulation_is_applied(tmp_path):
    # the stream ends after an odd number of micro-steps with
    # grad_accumulation=2: the pending gradients must reach the optimizer
    # (counted through a recording wrapper -- asserting on p.grad alone
    # cannot tell "applied" from "silently zeroed")
    cfg = _toy_cfg(tmp_path)
    cfg["driver"]["advance"] = {"kind": "observations", "n": 100000}
    cfg["driver"]["max_steps"] = None
    cfg["driver"]["grad_accumulation"] = 2
    pipe = egolearn.build(cfg)

    calls = []
    orig_step = pipe.driver.optimizer.step

    def counting_step(*a, **kw):
        calls.append(1)
        return orig_step(*a, **kw)

    pipe.driver.optimizer.step = counting_step
    pipe.driver.fit()
    assert pipe.driver._exhausted
    assert pipe.ctx.clock.step % 2 == 1  # an odd tail was left pending...
    assert len(calls) == pipe.ctx.clock.step // 2 + 1  # ...and flushed
    assert all(p.grad is None for p in pipe.model.parameters())


def test_mid_window_snapshot_resume_matches_uninterrupted(tmp_path):
    # grad_accumulation=2, snapshot after 3 micro-steps (mid-window: one
    # backward pending).  Pending gradients travel inside the snapshot and
    # no phantom flush runs at the max_steps stop, so the continuation folds
    # micro-steps 3+4 into one optimizer step exactly like the straight run.
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    cfg = _toy_cfg(tmp_path)
    cfg["driver"]["max_steps"] = 4
    cfg["driver"]["grad_accumulation"] = 2
    ref = egolearn.build(cfg)
    ref.driver.fit()

    cfg1 = _toy_cfg(tmp_path)
    cfg1["driver"]["max_steps"] = 3
    cfg1["driver"]["grad_accumulation"] = 2
    p1 = egolearn.build(cfg1)
    p1.driver.fit()
    assert any(p.grad is not None for p in p1.model.parameters())  # mid-window
    snap = str(tmp_path / "ga_snap.pth")
    save_snapshot(snap, p1.driver)

    cfg2 = _toy_cfg(tmp_path)
    cfg2["driver"]["max_steps"] = 4
    cfg2["driver"]["grad_accumulation"] = 2
    p2 = egolearn.build(cfg2)
    load_snapshot(snap, p2.driver)
    p2.driver.fit()

    resumed = p2.model.state_dict()
    for k, v in ref.model.state_dict().items():
        assert torch.equal(v, resumed[k]), f"mid-window resume diverged at {k}"


def test_resume_after_stream_end_adds_nothing(tmp_path):
    # a snapshot taken after the stream ran dry: resuming must not re-admit
    # the final session-end flush or train an extra quantum
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    cfg = _toy_cfg(tmp_path)
    cfg["driver"]["advance"] = {"kind": "observations", "n": 100000}
    cfg["driver"]["max_steps"] = None
    ref = egolearn.build(cfg)
    ref.driver.fit()
    steps, events = ref.ctx.clock.step, ref.ctx.clock.events_emitted
    sizes = {t: len(tab) for t, tab in ref.memories["frames"].tiers.items()}
    snap = str(tmp_path / "end_snap.pth")
    save_snapshot(snap, ref.driver)

    cfg2 = _toy_cfg(tmp_path)
    cfg2["driver"]["advance"] = {"kind": "observations", "n": 100000}
    cfg2["driver"]["max_steps"] = None
    p2 = egolearn.build(cfg2)
    load_snapshot(snap, p2.driver)
    p2.driver.fit()
    assert p2.ctx.clock.step == steps
    assert p2.ctx.clock.events_emitted == events
    assert {t: len(tab) for t, tab in p2.memories["frames"].tiers.items()} == sizes


def test_snapshot_restores_rng_states_after_replay():
    # the resume order is clock -> replay -> RNG states: a replay that
    # consumes named RNGs must not advance them past the snapshot's values
    import tempfile

    from egolearn.core.context import Context
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    class StubDriver:
        def __init__(self, ctx):
            self.ctx = ctx
            self.model = torch.nn.Linear(2, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.memories = {}

        def _resume_replay(self, exhausted: bool = False):
            # replay consumes admission randomness, like the segmenters do
            for _ in range(self.ctx.clock.admitted):
                self.ctx.rng("admission.stub").random()

    # original run: consume the rng 5 times, snapshot
    ctx1 = Context(seed=11, device="cpu")
    d1 = StubDriver(ctx1)
    for _ in range(5):
        ctx1.rng("admission.stub").random()
    ctx1.clock.admitted = 5
    saved_state = ctx1.rng("admission.stub").getstate()

    with tempfile.TemporaryDirectory() as td:
        sp = f"{td}/s.pth"
        save_snapshot(sp, d1)
        ctx2 = Context(seed=11, device="cpu")
        d2 = StubDriver(ctx2)
        load_snapshot(sp, d2)
    # replay consumed 5 draws, but the snapshot state was restored AFTER it
    assert ctx2.rng("admission.stub").getstate() == saved_state


def test_first_step_uses_warmup_lr(tmp_path):
    # the schedule is applied before optimizer.step, so gradient step 1 runs
    # at the warmup multiplier instead of the full base LR
    cfg = _toy_cfg(tmp_path)
    cfg["solver"]["schedule"] = {"warmup_min_steps": 100}
    cfg["driver"]["max_steps"] = 1
    pipe = egolearn.build(cfg)
    base = pipe.driver._base_lrs[0]

    seen = []
    orig_step = pipe.driver.optimizer.step

    def recording_step(*a, **kw):
        seen.append(pipe.driver.optimizer.param_groups[0]["lr"])
        return orig_step(*a, **kw)

    pipe.driver.optimizer.step = recording_step
    pipe.driver.fit()
    assert len(seen) == 1
    assert seen[0] < base  # warmup multiplier, not full LR


def test_an_chunk_covers_final_frame_at_tied_timestamp():
    # SAYCam transcript timestamps are frame-aligned, so EVERY utterance ends
    # exactly on a frame time.  The merge surfaces that frame before the
    # utterance (channel order), so the chunk includes its final step --
    # matching mcl, whose step2first is prebuilt over the whole video.
    from egolearn.core.refs import ItemRef
    from egolearn.data.sources.base import (
        DenseFrameChannel,
        Session,
        SparseIntervalChannel,
    )
    from egolearn.methods.babycl.an_chunks import ANChunkEmitter

    times = [i * 0.2 for i in range(6)]  # 5 fps: steps 0..5
    video = DenseFrameChannel(
        "video", "s", times=times, locators=[("d", i, i + 1) for i in range(6)]
    )
    text = SparseIntervalChannel(
        "transcript",
        "s",
        items=[
            {
                "t0": 0.6,
                "t1": 1.0,
                "payload": {
                    K.TEXT_UTTERANCE: "look at the doggie",
                    K.TEXT_TIMESTAMPS: [0.6, 0.8, 1.0],
                    K.UTTERANCE_NUM: 1,
                },
            }
        ],
    )
    sess = Session("s", {"video": video, "transcript": text})

    em = ANChunkEmitter(fps=5.0)
    em.start_session("s")
    events = []
    for obs in sess.observations():
        if obs.channel in em.subscribes:
            em.ingest(obs)
        events.extend(em.poll())
    em.end_session()
    events.extend(em.poll())

    assert len(events) == 1
    ref: ItemRef = events[0][K.VIDEO_REF]
    # frames at steps 3..5 -> indices [3, 6): the tied final frame included
    assert ref.locator[1:] == (3, 6)
    assert ref.t1 == 1.0


def test_an_chunk_waits_for_rounded_up_final_frame():
    # a final timestamp that rounds UP to a frame step the stream has not
    # reached yet: the chunk must wait for that frame instead of dropping
    # the step (mcl's step2first is prebuilt, so it never drops it)
    from egolearn.core.refs import ItemRef
    from egolearn.data.sources.base import (
        DenseFrameChannel,
        Session,
        SparseIntervalChannel,
    )
    from egolearn.methods.babycl.an_chunks import ANChunkEmitter

    times = [i * 0.2 for i in range(6)]  # steps 0..5
    video = DenseFrameChannel(
        "video", "s", times=times, locators=[("d", i, i + 1) for i in range(6)]
    )
    text = SparseIntervalChannel(
        "transcript",
        "s",
        items=[
            {
                "t0": 0.31,
                "t1": 0.51,  # surfaces between frames 0.4 and 0.6...
                "payload": {
                    K.TEXT_UTTERANCE: "hi",
                    # ...but its last timestamp rounds to step 3 (frame 0.6)
                    K.TEXT_TIMESTAMPS: [0.31, 0.51],
                    K.UTTERANCE_NUM: 2,
                },
            }
        ],
    )
    sess = Session("s", {"video": video, "transcript": text})
    em = ANChunkEmitter(fps=5.0)
    em.start_session("s")
    events = []
    for obs in sess.observations():
        if obs.channel in em.subscribes:
            em.ingest(obs)
        events.extend(em.poll())
    em.end_session()
    events.extend(em.poll())

    assert len(events) == 1
    ref: ItemRef = events[0][K.VIDEO_REF]
    # steps round(0.31*5)=2 and round(0.51*5)=3 -> frames [2, 4)
    assert ref.locator[1:] == (2, 4)
