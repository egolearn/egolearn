"""Core contract tests (no real data, no network, no GPU).

Run: pytest tests/ -q
"""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

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
    got = mem.sample_mix(4, {"short": 0.25, "long": 0.75}, ctx.fresh_rng("compose.0"))
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


def test_validation_rejects_unknown_slot(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
    cfg["losses"][0]["slots"] = ["nonexistent"]
    with pytest.raises(egolearn.PipelineError, match="nonexistent"):
        egolearn.build(cfg)


def test_recipe_rejects_batch_local_targets(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
    cfg["recipe"]["slots"][0]["batch_label"] = True
    with pytest.raises(TypeError, match="batch_label"):
        egolearn.build(cfg)


def test_validation_rejects_unknown_view(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
    cfg["recipe"]["slots"][0]["views"] = ["not_a_view", "v2"]
    with pytest.raises(egolearn.PipelineError, match="not_a_view"):
        egolearn.build(cfg)


def test_validation_rejects_text_loss_without_text_encoder(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
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
def _offline_cfg(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
    for key in ("admission", "memories", "routes", "recipe", "label_stores"):
        del cfg[key]
    del cfg["model"]["heads"]["cls"]
    cfg["losses"] = cfg["losses"][:1]
    cfg["losses"][0]["slots"] = ["frames"]
    cfg["driver"] = {
        "mode": "offline",
        "batch": {"slot": "frames", "n": 8, "views": ["v1", "v2"]},
        "epochs": 1,
        "max_steps": 3,
    }
    return cfg


def test_offline_mode_rejects_memory_config(toy_config, tmp_path):
    cfg = _offline_cfg(toy_config, tmp_path)
    cfg["memories"] = {
        "frames": {
            "tiers": {"short": {"capacity": 8, "update": "fifo"}},
            "admit_to": ["short"],
            "overflow": {},
        }
    }
    with pytest.raises(egolearn.PipelineError, match=r"memory-free|no.*memory|memories"):
        egolearn.build(cfg)


def test_offline_mode_rejects_unknown_view(toy_config, tmp_path):
    cfg = _offline_cfg(toy_config, tmp_path)
    cfg["driver"]["batch"]["views"] = ["v1", "nope"]
    with pytest.raises(egolearn.PipelineError, match="nope"):
        egolearn.build(cfg)


@pytest.mark.parametrize(
    "invalid,prefetch,successful_steps",
    [("nan", False, 1), ("nan", True, 1), ("inf", False, 0), ("empty", True, 0)],
)
def test_offline_invalid_loss_stops_without_advancing_batches(
    toy_config, tmp_path, monkeypatch, invalid, prefetch, successful_steps
):
    cfg = _offline_cfg(toy_config, tmp_path)
    cfg["record_json"] = False  # Failure must not depend on run-record hooks.
    cfg["driver"].update(prefetch=prefetch, grad_accumulation=3)
    cfg["loader"] = {"workers": 1 if prefetch else 0}
    pipe = egolearn.build(cfg)
    driver = pipe.driver
    before = [p.detach().clone() for p in pipe.model.parameters()]
    hook = Mock()
    driver.hooks.append(hook)
    close = Mock(wraps=driver.loader.close)
    monkeypatch.setattr(driver.loader, "close", close)
    step = Mock(wraps=driver.optimizer.step)
    monkeypatch.setattr(driver.optimizer, "step", step)
    loss = pipe.losses[0]
    compute = loss.compute
    calls = 0

    def invalid_loss(*args):
        nonlocal calls
        calls += 1
        # Bound the regression even if a persistent invalid loss would loop forever.
        if calls > successful_steps + 1:
            pytest.fail("offline training continued after an invalid loss")
        if calls <= successful_steps:
            return compute(*args)
        return {} if invalid == "empty" else {"loss": torch.tensor(float(invalid))}

    monkeypatch.setattr(loss, "compute", invalid_loss)
    with pytest.raises(
        RuntimeError,
        match=(
            rf"step {successful_steps}.*non-finite or empty loss.*"
            rf"epoch 0, batch {successful_steps}"
        ),
    ):
        driver.fit()

    assert calls == successful_steps + 1
    assert pipe.ctx.clock.step == pipe.ctx.clock.admitted == successful_steps
    assert pipe.ctx.clock.compose_seq == successful_steps + 1
    assert driver.skipped_steps == 1
    assert driver.nonfinite_steps == int(invalid != "empty")
    hook.on_skip.assert_called_once_with(driver, nonfinite=invalid != "empty")
    hook.on_end.assert_not_called()
    step.assert_not_called()  # Do not flush the partial accumulation window on failure.
    for old, current in zip(before, pipe.model.parameters(), strict=True):
        assert torch.equal(old, current)
    close.assert_called_once_with()
    assert driver.loader._pool is None


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
    with pytest.raises(KeyError, match="no encoder named"):
        model.encode("missing", x)


def test_video_encoder_trains_on_clip_slots(toy_config, tmp_path):
    # video SSL expressed purely in config: clip unit -> video encoder ->
    # ntxent over the two clip views, end to end through the driver
    cfg = toy_config(tmp_path)
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
            "labels_from": "fixed",
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
    pipe.driver._run_warmup()
    batch = pipe.composer.compose(pipe.recipe, pipe.memories, {}, pipe.ctx)
    assert batch is not None
    assert batch["clips"]["views"].shape == (3, 2, 4, 3, 48, 48)
    assert batch["clips"]["labels"].shape == (3,)
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2  # max_steps from the shared toy config


def test_predictor_wiring_and_validation(toy_config, tmp_path):
    import pytest

    cfg = toy_config(tmp_path)
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

    cfg2 = toy_config(tmp_path)
    cfg2["model"]["predictors"] = {
        "pred": {"inputs": ["video"], "_target_": "predictor:mlp", "sizes": [64, 64]}
    }
    with pytest.raises(egolearn.PipelineError, match=r"predictor .* reads encoder"):
        egolearn.build(cfg2)


def test_probe_capability_mismatch_fails_at_build(toy_config, tmp_path):
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

    cfg = toy_config(tmp_path)  # image-only model
    cfg["probes"] = [{"_target_": "probe:needs_text_probe"}]
    with pytest.raises(egolearn.PipelineError, match="needs capability"):
        egolearn.build(cfg)


def test_frozen_twin_encoder_stays_out_of_the_way(toy_config, tmp_path):
    # the JEPA EMA-target pattern: a frozen copy of an encoder is carried by
    # the model (so it snapshots) but is excluded from the optimizer and
    # never grabs modality-based routing
    cfg = toy_config(tmp_path)
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


def test_head_unknown_encoder_fails_at_build(toy_config, tmp_path):
    # a head reading a nonexistent encoder is caught even when no loss
    # requires that head (probe-only heads used to slip through)
    import pytest

    cfg = toy_config(tmp_path)
    cfg["model"]["heads"]["orphan"] = {
        "encoder": "backbone",  # no encoder has this name
        "_target_": "head:mlp",
        "sizes": [64, 32, 16],
    }
    with pytest.raises(egolearn.PipelineError, match="reads encoder"):
        egolearn.build(cfg)


def test_slot_modality_without_encoder_warns(toy_config, tmp_path):
    # a clip slot on an image-only model: no video encoder will ever fill
    # slot["feats"], so the build surfaces a warning instead of leaving the
    # mismatch to explode at the first batch
    cfg = toy_config(tmp_path)
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


def test_two_trainable_same_modality_encoders_rejected_at_build(toy_config, tmp_path):
    import pytest

    cfg = toy_config(tmp_path)
    cfg["model"]["encoders"] = {
        "vision_a": {"_target_": "encoder:toy_cnn", "dim": 64},
        "vision_b": {"_target_": "encoder:toy_cnn", "dim": 64},
    }
    with pytest.raises(egolearn.PipelineError, match="compete for modality"):
        egolearn.build(cfg)


def test_solver_warns_on_group_matching_nothing(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
    cfg["solver"]["groups"] = [
        {"match": r"^backbone\.", "lr": 1e-3},  # matches nothing: params are encoders.*
        {"match": ".*", "lr": 1e-4},
    ]
    pipe = egolearn.build(cfg)
    assert any("matched no parameters" in w for w in pipe.warnings)


# ---------------------------------------------------------------------------
# determinism: the three guarantees the framework exists to provide
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_same_seed_twice_gives_identical_weights(toy_config, tmp_path, mode):
    cfg = _offline_cfg(toy_config, tmp_path) if mode == "offline" else toy_config(tmp_path)
    states = []
    for _ in range(2):
        pipe = egolearn.build(deepcopy(cfg))
        pipe.driver.fit()
        assert pipe.mode == mode
        assert pipe.ctx.clock.step == cfg["driver"]["max_steps"]
        if mode == "offline":
            assert pipe.ctx.clock.progress == pytest.approx(1.0)
        report = pipe.driver.timer.report()
        assert "batch.compose" in report and "data.load" in report
        assert report["model.forward"]["count"] == pipe.ctx.clock.step
        states.append({k: v.clone() for k, v in pipe.model.state_dict().items()})
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key


def test_resumed_run_does_not_overwrite_milestone_snapshots(tmp_path):
    # SnapshotHook's fired-milestones set is per-process, so a resumed run
    # re-crosses milestones the original run already recorded; those files
    # are write-once archives and must survive the resume untouched
    from egolearn.training.hooks import SnapshotHook

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "snapshot_005pct.pth"
    marker.write_bytes(b"original milestone state")

    class _Clock:
        step, progress = 10, 0.2  # far past the 5% milestone

    class _Ctx:
        clock = _Clock()

    class _Driver:
        ctx = _Ctx()

    hook = SnapshotHook(str(run_dir), milestones=[0.05])
    hook.on_step(_Driver(), {})
    assert marker.read_bytes() == b"original milestone state"
    assert 0.05 in hook._done  # marked handled, so it is not re-checked


@pytest.mark.parametrize(
    "mode",
    [
        "streaming",
        "offline",
        "accumulation",
        "prefetch",
        "mid_quantum",
        "mid_quantum_reuse",
        "final_quantum",
        "final_quantum_amp",
    ],
)
def test_resumed_run_matches_uninterrupted(toy_config, tmp_path, mode):
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    cfg = _offline_cfg(toy_config, tmp_path) if mode == "offline" else toy_config(tmp_path)
    cfg["driver"]["max_steps"] = 4
    cut = 2
    if mode == "accumulation":
        cfg["driver"]["grad_accumulation"] = 2
        cut = 3  # Snapshot with one backward pending in the accumulation window.
    elif mode.startswith("mid_quantum"):
        # Snapshot one step into the second quantum; with resample off, the
        # resumed run must rebuild the batch that quantum reuses.
        cfg["driver"]["rate"] = {"steps_per_admit": 3, "resample": mode == "mid_quantum"}
        cfg["driver"]["max_steps"] = 6
        cut = 4
    elif mode.startswith("final_quantum"):
        # The stream is exhausted, but the capped final quantum still owes
        # updates: its partial accumulation window must survive the snapshot.
        cfg["driver"].update(
            advance={"kind": "observations", "n": 100000},
            rate={"steps_per_admit": 7, "resample": True},
            grad_accumulation=2,
            max_steps=7,  # The completed quantum must flush its odd tail.
        )
        if mode == "final_quantum_amp":
            cfg["driver"]["amp"] = {"dtype": "float16"}
        cut = 3
    elif mode == "prefetch":
        cfg = _prefetch_cfg(toy_config, tmp_path, True)
        cut = 4  # Snapshot at a quantum boundary with a prefetched batch in flight.
    ref = egolearn.build(deepcopy(cfg))
    ref.driver.fit()

    interrupted = deepcopy(cfg)
    interrupted["driver"]["max_steps"] = cut
    first = egolearn.build(interrupted)
    first.driver.fit()
    if mode == "accumulation" or mode.startswith("final_quantum"):
        assert any(p.grad is not None for p in first.model.parameters())
    if mode.startswith("final_quantum"):
        assert first.driver._exhausted
        assert first.ctx.clock.quantum_left == 4
    snapshot = str(tmp_path / "resume.pth")
    save_snapshot(snapshot, first.driver)

    resumed = egolearn.build(cfg)
    load_snapshot(snapshot, resumed.driver)
    assert resumed.ctx.clock.step == cut
    assert resumed.ctx.clock.admitted == first.ctx.clock.admitted
    for name, memory in first.memories.items():
        assert {tier: len(table) for tier, table in memory.tiers.items()} == {
            tier: len(table) for tier, table in resumed.memories[name].tiers.items()
        }
    resumed.driver.fit()
    assert resumed.ctx.clock.step == ref.ctx.clock.step == cfg["driver"]["max_steps"]
    assert resumed.ctx.clock.admitted == ref.ctx.clock.admitted
    if mode.startswith("final_quantum"):
        assert all(p.grad is None for p in ref.model.parameters())
        assert all(p.grad is None for p in resumed.model.parameters())
    if mode == "final_quantum_amp":
        assert resumed.driver.scaler.state_dict() == ref.driver.scaler.state_dict()
    for key, value in ref.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key


def test_same_timestamp_observations_keep_channel_order():
    # ties at equal media time surface in channel declaration order: the
    # frame comes before the utterance that ends at the same instant
    from egolearn.data.sources.base import DenseFrameChannel, Session, SparseIntervalChannel

    video = DenseFrameChannel("video", "s", times=[0.0, 1.0], locators=[("f", 0, 1), ("f", 1, 2)])
    text = SparseIntervalChannel("transcript", "s", items=[{"t0": 0.5, "t1": 1.0, "payload": {}}])
    sess = Session("s", {"video": video, "transcript": text})
    seq = [(o.channel, o.t) for o in sess.observations()]
    assert seq == [("video", 0.0), ("video", 1.0), ("transcript", 1.0)]


def test_final_partial_grad_accumulation_is_applied(toy_config, tmp_path):
    # the stream ends after an odd number of micro-steps with
    # grad_accumulation=2: the pending gradients must reach the optimizer
    # (counted through a recording wrapper -- asserting on p.grad alone
    # cannot tell "applied" from "silently zeroed")
    cfg = toy_config(tmp_path)
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
    assert pipe.ctx.clock.step >= 1
    assert pipe.ctx.clock.step % 2 == 1  # an odd tail was left pending...
    assert len(calls) == pipe.ctx.clock.step // 2 + 1  # ...and flushed
    assert all(p.grad is None for p in pipe.model.parameters())


def test_resume_after_stream_end_adds_nothing(toy_config, tmp_path):
    # a snapshot taken after the stream ran dry: resuming must not re-admit
    # the final session-end flush or train an extra quantum
    from egolearn.utils.snapshot import load_snapshot, save_snapshot

    cfg = toy_config(tmp_path)
    cfg["driver"]["advance"] = {"kind": "observations", "n": 100000}
    cfg["driver"]["max_steps"] = None
    ref = egolearn.build(cfg)
    ref.driver.fit()
    steps, events = ref.ctx.clock.step, ref.ctx.clock.events_emitted
    sizes = {t: len(tab) for t, tab in ref.memories["frames"].tiers.items()}
    snap = str(tmp_path / "end_snap.pth")
    save_snapshot(snap, ref.driver)

    cfg2 = toy_config(tmp_path)
    cfg2["driver"]["advance"] = {"kind": "observations", "n": 100000}
    cfg2["driver"]["max_steps"] = None
    p2 = egolearn.build(cfg2)
    load_snapshot(snap, p2.driver)
    p2.driver.fit()
    assert p2.ctx.clock.step == steps
    assert p2.ctx.clock.events_emitted == events
    assert {t: len(tab) for t, tab in p2.memories["frames"].tiers.items()} == sizes


def test_clock_from_an_older_snapshot_owes_no_quantum():
    # snapshots saved before quantum_left existed resume at the next advance
    state = Context(seed=0, device="cpu").state_dict()
    del state["clock"]["quantum_left"]
    ctx = Context(seed=0, device="cpu")
    ctx.clock.quantum_left = 5
    ctx.load_clock(state)
    assert ctx.clock.quantum_left == 0


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


def test_first_step_uses_warmup_lr(toy_config, tmp_path):
    # the schedule is applied before optimizer.step, so gradient step 1 runs
    # at the warmup multiplier instead of the full base LR
    cfg = toy_config(tmp_path)
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


@pytest.mark.parametrize(
    "timestamps, expected",
    [([0.6, 0.8, 1.0], (3, 6)), ([0.31, 0.51], (2, 4))],
    ids=["tied-final-frame", "rounded-up-final-frame"],
)
def test_an_chunk_includes_final_frame(timestamps, expected):
    from egolearn.config.registry import ADMISSION
    from egolearn.data.sources.base import DenseFrameChannel, Session, SparseIntervalChannel
    from egolearn.methods.babycl.an_chunks import ANChunkEmitter

    assert ADMISSION.get("an_chunk_emitter") is ANChunkEmitter
    assert ADMISSION.get("babycl_online_segmenter") is not None
    video = DenseFrameChannel(
        "video",
        "s",
        times=[i * 0.2 for i in range(6)],
        locators=[("d", i, i + 1) for i in range(6)],
    )
    text = SparseIntervalChannel(
        "transcript",
        "s",
        items=[
            {
                "t0": timestamps[0],
                "t1": timestamps[-1],
                "payload": {
                    K.TEXT_UTTERANCE: "hello",
                    K.TEXT_TIMESTAMPS: timestamps,
                    K.UTTERANCE_NUM: 1,
                },
            }
        ],
    )
    emitter = ANChunkEmitter(fps=5.0)
    emitter.start_session("s")
    events = []
    for observation in Session("s", {"video": video, "transcript": text}).observations():
        if observation.channel in emitter.subscribes:
            emitter.ingest(observation)
        events.extend(emitter.poll())
    emitter.end_session()
    events.extend(emitter.poll())
    assert len(events) == 1
    assert events[0][K.VIDEO_REF].locator[1:] == expected
    assert events[0][K.VIDEO_REF].t1 == pytest.approx((expected[1] - 1) * 0.2)


def test_worker_pool_matches_serial_loading(toy_config, tmp_path):
    # the worker count is a throughput knob only: same seed, same batches,
    # whether decode+augment ran in-process or in a spawn pool
    def first_batch(workers):
        cfg = toy_config(tmp_path / f"w{workers}")
        cfg["_imports_"] = ["egolearn.methods.babycl"]
        cfg["views"] = {
            f"v{i + 1}": {"name": "babycl_ms_aug", "size": 64, "view_idx": i} for i in range(2)
        }
        cfg["loader"] = {"workers": workers}
        pipe = egolearn.build(cfg)
        pipe.driver._run_warmup()
        batch = pipe.composer.compose(pipe.recipe, pipe.memories, {}, pipe.ctx)
        pipe.loader.close()
        assert batch is not None
        return batch

    serial, pooled = first_batch(0), first_batch(2)
    assert serial["curr"]["views"].shape == (4, 2, 3, 64, 64)
    assert serial["curr"]["labels"].shape == (4,)
    assert sorted(serial) == sorted(pooled)
    for name in serial:
        assert torch.equal(serial[name]["views"], pooled[name]["views"]), name
        if "labels" in serial[name]:
            assert torch.equal(serial[name]["labels"], pooled[name]["labels"])


def _prefetch_cfg(toy_config, tmp_path, prefetch):
    # quanta of 4 steps (prefetch only ever composes ahead WITHIN a quantum,
    # so steps_per_admit must be > 1 for it to engage at all)
    cfg = toy_config(tmp_path / f"p{int(prefetch)}")
    cfg["loader"] = {"workers": 2}
    cfg["driver"]["rate"] = {"steps_per_admit": 4, "resample": True}
    cfg["driver"]["max_steps"] = 8
    cfg["driver"]["prefetch"] = prefetch
    return cfg


def test_prefetch_matches_serial_streaming(toy_config, tmp_path):
    # prefetch is a pure throughput knob: batches derive from (seed,
    # compose_seq, memory contents), so composing ahead changes nothing
    states = []
    for prefetch in (False, True):
        pipe = egolearn.build(_prefetch_cfg(toy_config, tmp_path, prefetch))
        pipe.driver.fit()
        assert pipe.ctx.clock.step == 8
        states.append({k: v.clone() for k, v in pipe.model.state_dict().items()})
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), f"prefetch diverged at {k}"


def test_prefetch_matches_serial_offline(toy_config, tmp_path):
    states = []
    for prefetch in (False, True):
        cfg = _offline_cfg(toy_config, tmp_path / f"o{int(prefetch)}")
        cfg["loader"] = {"workers": 2}
        cfg["driver"]["prefetch"] = prefetch
        cfg["driver"]["max_steps"] = 4
        pipe = egolearn.build(cfg)
        pipe.driver.fit()
        assert pipe.ctx.clock.step == 4
        states.append({k: v.clone() for k, v in pipe.model.state_dict().items()})
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), f"prefetch diverged at {k}"


def test_prefetch_requires_workers(toy_config, tmp_path):
    cfg = _prefetch_cfg(toy_config, tmp_path, True)
    cfg["loader"] = {"workers": 0}
    with pytest.raises(egolearn.PipelineError, match=r"loader\.workers"):
        egolearn.build(cfg)


def test_prefetch_refused_when_a_loss_mutates_sampling(toy_config, tmp_path):
    from egolearn.config.registry import LOSSES
    from egolearn.model.losses.base import Loss

    if "mutating_test_loss" not in LOSSES.names():

        @LOSSES.register("mutating_test_loss")
        class MutatingTestLoss(Loss):
            # e.g. per-step replay priorities: what the composer would sample
            # changes between steps, so batches cannot be composed ahead
            mutates_sampling_state = True

            def __init__(self, **kw):
                super().__init__("mut")

            def compute(self, batch, model, ctx):
                return {}

    cfg = _prefetch_cfg(toy_config, tmp_path, True)
    cfg["losses"].append({"_target_": "loss:mutating_test_loss"})
    with pytest.raises(egolearn.PipelineError, match="mutates_sampling_state"):
        egolearn.build(cfg)
