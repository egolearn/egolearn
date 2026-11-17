"""The complete offline BabyCL data/loss path, using small synthetic frame files."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.core import keys as K
from egolearn.methods.babycl.offline import build_offline_index
from egolearn.training.hooks import Hook
from egolearn.utils.snapshot import load_snapshot, save_snapshot


@pytest.fixture
def iid_config(tmp_path):
    root = tmp_path / "frames"
    videos, transcript = [], []
    for s in range(2):
        sid = f"S_{s}"
        directory = root / sid
        directory.mkdir(parents=True)
        frames = []
        for i in range(12):
            name = f"{sid}_{i / 5:.2f}.jpg"
            Image.new("RGB", (24, 24), (i * 17, s * 90, 150)).save(directory / name)
            frames.append(name)
        videos.append(
            {
                "video_dir": sid,
                "frames": frames,
                "segments": [{"start": 0, "end": 6}, {"start": 6, "end": 12}],
            }
        )
        # Two contiguous runs, one missing timestamp, and a duplicate annotation.
        utterance = {
            "video_filename": f"{sid}.avi",
            "utterance_num": s * 2,
            "utterance": "red ball",
            "timestamps": [0.0, 0.2, 0.8, 1.0, 9.0],
        }
        transcript.extend([utterance, dict(utterance)])
        transcript.append(
            {
                "video_filename": f"{sid}.avi",
                "utterance_num": s * 2 + 1,
                "utterance": "blue ball",
                "timestamps": [1.8, 2.0, 2.2],
            }
        )
    storyboard = tmp_path / "storyboard.json"
    storyboard.write_text(json.dumps({"fps": 5, "root_dir": "unused", "videos": videos}))
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps(transcript))
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"<pad>": 0, "<unk>": 1, "red": 2, "blue": 3, "ball": 4}))
    return {
        "_imports_": ["egolearn.methods.babycl"],
        "seed": 7,
        "device": "cpu",
        "run_dir": str(tmp_path / "run"),
        "stream": {
            "_target_": "stream:saycam",
            "root_dir": str(root),
            "transcript_json": str(transcript_path),
            "fps": 5.0,
            "include": "S_*",
        },
        "tokenizer": {"vocab_json": str(vocab_path)},
        "views": {
            "v1": {"name": "simclr_strong", "size": 16, "view_idx": 0},
            "v2": {"name": "simclr_strong", "size": 16, "view_idx": 1},
            "pair": {"name": "center_crop", "size": 16},
        },
        "model": {
            "encoders": {
                "vision": {"_target_": "encoder:toy_cnn", "dim": 16},
                "text": {"_target_": "encoder:text_bow", "vocab_size": "auto", "dim": 8},
            },
            "heads": {
                "proj": {"encoder": "vision", "_target_": "head:mlp", "sizes": [16, 16, 8]},
                "segment": {
                    "encoder": "vision",
                    "_target_": "head:linear",
                    "in_dim": 16,
                    "out_dim": 8,
                },
                "embed": {
                    "encoder": "vision",
                    "_target_": "head:linear",
                    "in_dim": 16,
                    "out_dim": 8,
                },
            },
        },
        "losses": [
            {"_target_": "loss:ntxent", "slots": ["segments"], "head": "proj", "weight": 0.5},
            {
                "_target_": "loss:supcon",
                "slots": ["segments"],
                "head": "segment",
                "temperature": 0.2,
                "weight": 0.5,
            },
            {
                "_target_": "loss:clip_infonce",
                "slot": "pairs",
                "vision_head": "embed",
                "weight": 0.5,
            },
        ],
        "solver": {
            "opt": "adamw",
            "groups": [{"match": ".*", "lr": 0.001}],
            "schedule": {"constant": True},
        },
        "driver": {
            "mode": "offline",
            "_target_": "egolearn.methods.babycl.offline.BabyCLOfflineDriver",
            "batch": {
                "storyboard_json": str(storyboard),
                "segment_views": ["v1", "v2"],
                "pair_view": "pair",
                "segments_per_batch": 2,
                "frames_per_segment": 2,
                "pairs_per_batch": 2,
            },
            "epochs": 1,
        },
        "hooks": {"log_every": 1},
    }


def test_index_preserves_alignment_deduplicates_and_covers_frames(iid_config):
    pipe = egolearn.build(iid_config)
    driver = pipe.driver
    refs = driver._build_index()
    assert driver.index.num_frames == 24
    assert len(driver.index.segments) == 4
    assert len(refs) == 6
    assert sorted(p[K.VIDEO_REF].locator[1:] for p in refs) == [
        (0, 2),
        (0, 2),
        (4, 6),
        (4, 6),
        (9, 12),
        (9, 12),
    ]
    order = list(range(len(refs)))
    pipe.ctx.fresh_rng("offline.shuffle.epoch0").shuffle(order)
    consumed = [p for b in range(3) for p in driver._pick(refs, order, b)]
    assert len({(p[K.VIDEO_REF].key, p[K.UTTERANCE_NUM]) for p in consumed}) == 6
    batch = driver._compose(consumed[:2], seq=3)
    assert batch["segments"][K.SLOT_VIEWS].shape == (4, 2, 3, 16, 16)
    assert driver.index.segment_labels == [0, 1, 2, 3]
    label_by_ref = dict(
        zip(
            [s[K.VIDEO_REF].key for s in driver.index.segments],
            driver.index.segment_labels,
            strict=True,
        )
    )
    assert batch["segments"][K.SLOT_LABELS].tolist() == [
        label_by_ref[item[K.VIDEO_REF].key] for item in batch["segments"][K.SLOT_META]
    ]
    assert batch["pairs"][K.SLOT_VIEWS].shape == (2, 1, 3, 16, 16)
    assert batch["pairs"][K.SLOT_TEXT_IDS].shape == (2, 2)
    repeated = driver._compose(consumed[:2], seq=3)
    assert pipe.ctx.clock.compose_seq == 0  # unconsumed prefetch doesn't advance the clock
    for slot in batch:
        assert torch.equal(batch[slot][K.SLOT_VIEWS], repeated[slot][K.SLOT_VIEWS])


def test_all_three_losses_train_both_encoders_and_every_head(iid_config):
    pipe = egolearn.build(iid_config)
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    history = []

    class Capture(Hook):
        def on_step(self, driver, logs):
            history.append(logs)

    pipe.driver.hooks.append(Capture())
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 3
    for logs in history:
        terms = [logs["instance/ntxent"], logs["temporal/supcon"], logs["cross_modal/clip_infonce"]]
        assert all(v > 0 for v in terms)
        assert logs["loss"] == pytest.approx(0.5 * sum(terms))
    after = pipe.model.state_dict()
    for prefix in (
        "encoders.vision.",
        "encoders.text.",
        "heads.proj.",
        "heads.segment.",
        "heads.embed.",
    ):
        assert any(not torch.equal(before[k], v) for k, v in after.items() if k.startswith(prefix))
    assert all(torch.isfinite(v).all() for v in after.values())


def test_global_ids_survive_reordering_duplicate_draws_and_cross_session_merges(iid_config):
    path = Path(iid_config["driver"]["batch"]["storyboard_json"])
    manifest = json.loads(path.read_text())
    gids = [9001, 12, 9001, 2**40]
    for segment, gid in zip(
        [s for v in manifest["videos"] for s in v["segments"]], gids, strict=True
    ):
        segment["global_id"] = gid
    path.write_text(json.dumps(manifest))
    driver = egolearn.build(iid_config).driver
    refs = driver._build_index()
    assert driver.index.segment_labels == gids
    for indices in ([0, 1], [1, 0], [0, 0], [0, 2], [3, 1]):
        batch = driver.composer.compose(
            driver.batch_recipe,
            {},
            {
                "VisualSegment": [driver.index.segments[i] for i in indices],
                "AlignedPair": refs[:2],
            },
            driver.ctx,
            seq=0,
        )
        assert batch["segments"][K.SLOT_LABELS].tolist() == [
            gids[i] for i in indices for _ in range(2)
        ]


def test_implicit_global_ids_do_not_change_when_sessions_are_filtered(iid_config):
    iid_config["stream"]["include"] = "S_1"
    driver = egolearn.build(iid_config).driver
    driver._build_index()
    assert driver.index.segment_labels == [2, 3]


@pytest.mark.parametrize("change", ["missing_segments", "invalid_global_id"])
def test_invalid_unselected_manifest_entries_name_the_session(iid_config, change):
    iid_config["stream"]["include"] = "S_1"
    path = Path(iid_config["driver"]["batch"]["storyboard_json"])
    manifest = json.loads(path.read_text())
    if change == "missing_segments":
        del manifest["videos"][0]["segments"]
    else:
        manifest["videos"][0]["segments"][0]["global_id"] = -1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="session 'S_0'"):
        egolearn.build(iid_config).driver._build_index()


@pytest.mark.parametrize("bad_id", [None, -1, 1.5, True, 2**63])
def test_invalid_or_partial_global_ids_are_rejected(iid_config, bad_id):
    path = Path(iid_config["driver"]["batch"]["storyboard_json"])
    manifest = json.loads(path.read_text())
    for video in manifest["videos"]:
        for segment in video["segments"]:
            segment["global_id"] = 9001
    segment = manifest["videos"][0]["segments"][0]
    if bad_id is None:
        del segment["global_id"]
    else:
        segment["global_id"] = bad_id
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="global_id"):
        egolearn.build(iid_config).driver._build_index()


@pytest.mark.parametrize("identity_source", ["metadata", "provenance"])
def test_resume_rejects_changed_global_segment_labels(iid_config, tmp_path, identity_source):
    from egolearn.utils.input_provenance import input_identity

    pipe = egolearn.build(iid_config)
    pipe.driver._build_index()
    snapshot = str(tmp_path / "snapshot.pth")
    save_snapshot(snapshot, pipe.driver)
    path = Path(iid_config["driver"]["batch"]["storyboard_json"])
    if identity_source == "provenance":
        state = torch.load(snapshot, weights_only=False)
        state["driver"]["metadata"] = None
        _, identity = input_identity(path)
        state["run_provenance"] = {"input_provenance": {"driver.batch.storyboard_json": identity}}
        torch.save(state, snapshot)
    # Compatible model weights can resume using either saved identity source.
    load_snapshot(snapshot, egolearn.build(iid_config).driver)
    manifest = json.loads(path.read_text())
    for video in manifest["videos"]:
        for segment in video["segments"]:
            segment["global_id"] = 9001
    path.write_text(json.dumps(manifest))
    resumed = egolearn.build(iid_config)
    with pytest.raises(ValueError, match="storyboard differs"):
        load_snapshot(snapshot, resumed.driver)


@pytest.mark.parametrize("pairs_only", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_cosine_resume_across_epoch_boundary(iid_config, tmp_path, pairs_only, prefetch):
    cfg = _pairs_only(iid_config) if pairs_only else iid_config
    cfg["driver"].update(epochs=2, prefetch=prefetch)
    cfg["loader"] = {"workers": 1 if prefetch else 0}
    cfg["solver"]["schedule"] = {"warmup_frac": 0.25, "warmup_min_steps": 2, "floor": 0.1}
    path = str(tmp_path / "mid_run.pth")
    full_history: list[tuple[int, float, dict]] = []
    resumed_history: list[tuple[int, float, dict]] = []

    class Capture(Hook):
        def __init__(self, history, snapshot=False):
            self.history, self.snapshot = history, snapshot

        def on_step(self, driver, logs):
            self.history.append(
                (driver.ctx.clock.step, driver.optimizer.param_groups[0]["lr"], logs)
            )
            if self.snapshot and driver.ctx.clock.step == 2:
                save_snapshot(path, driver)

    full = egolearn.build(copy.deepcopy(cfg))
    full.driver.hooks.append(Capture(full_history, snapshot=True))
    full.driver.fit()
    assert full.ctx.clock.step == 6  # three batches per epoch
    assert len({lr for _, lr, _ in full_history}) > 2
    resumed = egolearn.build(copy.deepcopy(cfg))
    load_snapshot(path, resumed.driver)
    assert resumed.ctx.clock.step == 2
    assert resumed.driver.schedule.state_dict() == full.driver.schedule.state_dict()
    resumed.driver.hooks.append(Capture(resumed_history))
    resumed.driver.fit()
    assert resumed_history == full_history[2:]
    for key, value in resumed.model.state_dict().items():
        assert torch.equal(value, full.model.state_dict()[key]), key


@pytest.mark.parametrize(
    "change,match", [("order", "frame order"), ("gap", "gaps/overlap"), ("fps", "fps")]
)
def test_mismatched_storyboard_is_rejected(iid_config, change, match):
    pipe = egolearn.build(iid_config)
    path = Path(iid_config["driver"]["batch"]["storyboard_json"])
    manifest = json.loads(path.read_text())
    if change == "order":
        manifest["videos"][0]["frames"].reverse()
    elif change == "gap":
        manifest["videos"][0]["segments"][1]["start"] = 7
    else:
        manifest["fps"] = 10
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        build_offline_index(pipe.stream, str(path))


def test_missing_tokenizer_and_pair_view_fail_validation(iid_config):
    no_tokenizer = copy.deepcopy(iid_config)
    no_tokenizer.pop("tokenizer")
    no_tokenizer["model"]["encoders"]["text"]["vocab_size"] = 5
    with pytest.raises(egolearn.PipelineError, match="tokenizer"):
        egolearn.build(no_tokenizer)
    iid_config["driver"]["batch"]["pair_view"] = "absent"
    with pytest.raises(egolearn.PipelineError, match="unknown view"):
        egolearn.build(iid_config)


def test_iid_preset_keeps_babycl_losses_without_streaming_components():
    config_dir = str(Path(__file__).resolve().parents[1] / "projects/babycl/configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="saycam_iid"), resolve=True)
        streaming = OmegaConf.to_container(compose(config_name="saycam_default"), resolve=True)
        visual = OmegaConf.to_container(compose(config_name="krishnacam"), resolve=True)
        walkingtours = OmegaConf.to_container(compose(config_name="walkingtours"), resolve=True)
    assert isinstance(cfg, dict)
    assert isinstance(streaming, dict)
    assert cfg["driver"]["mode"] == "offline"
    assert cfg["hooks"]["checkpoint_every"] == 5000
    assert streaming["hooks"]["checkpoint_every"] == 500
    for key in ("model", "losses", "solver", "views", "tokenizer"):
        if key == "solver":
            # only the streaming run reproduces the original's fixed step budget
            assert streaming[key]["schedule"].pop("total_steps") == 26568
        assert cfg[key] == streaming[key], key
    for key in ("memories", "admission", "routes", "recipe", "embedders", "probes"):
        assert not cfg[key], key
    assert [loss["weight"] for loss in cfg["losses"]] == [0.5, 0.5, 0.5]
    assert isinstance(visual, dict)
    assert isinstance(walkingtours, dict)
    for preset in (streaming, visual, walkingtours):
        recipes = [preset["recipe"]]
        if "fallback" in preset["recipe"]:
            recipes.append(preset["recipe"]["fallback"])
        for recipe in recipes:
            slot = next(s for s in recipe["slots"] if s["name"] == "segments")
            assert slot["labels_from"] == "events" and not slot.get("batch_label", False)
        loss = next(loss for loss in preset["losses"] if loss["name"] == "temporal")
        assert loss["_target_"] == "loss:supcon"
        assert "cls" not in preset["model"]["heads"]
        assert preset["model"]["heads"][loss["head"]]["sizes"][-1] == 128


# ---------------------------------------------------------------------------
# pairs-only batches (segments_per_batch=0): CVCL-style image/utterance training,
# the second stage of a "pretrain visually, then align to language" run
# ---------------------------------------------------------------------------
def _pairs_only(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    batch = cfg["driver"]["batch"]
    for key in ("storyboard_json", "segment_views"):
        batch.pop(key)
    batch["segments_per_batch"] = 0
    cfg["losses"] = [loss for loss in cfg["losses"] if loss["_target_"] == "loss:clip_infonce"]
    return cfg


def test_pairs_only_batches_train_the_clip_path_without_a_storyboard(iid_config):
    pipe = egolearn.build(_pairs_only(iid_config))
    assert pipe.driver.slot_names == {"pairs"}
    assert "segments" not in pipe.driver.batch_description
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    history = []

    class Capture(Hook):
        def on_step(self, driver, logs):
            history.append(logs)

    pipe.driver.hooks.append(Capture())
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 3  # 6 utterance chunks / 2 per batch
    assert pipe.driver.index.segments == [] and pipe.driver.index.num_frames == 24
    for logs in history:
        assert set(logs) == {"cross_modal/clip_infonce", "loss"}
        assert logs["loss"] == pytest.approx(0.5 * logs["cross_modal/clip_infonce"])
    after = pipe.model.state_dict()
    for prefix in ("encoders.vision.", "encoders.text.", "heads.embed."):
        assert any(not torch.equal(before[k], v) for k, v in after.items() if k.startswith(prefix))
    # heads no loss consumes stay untouched
    for prefix in ("heads.proj.", "heads.segment."):
        assert all(torch.equal(before[k], v) for k, v in after.items() if k.startswith(prefix))


def test_pairs_only_rejects_visual_losses_and_stray_storyboards(iid_config):
    with_visual = _pairs_only(iid_config)
    with_visual["losses"] = iid_config["losses"]
    with pytest.raises(egolearn.PipelineError, match="requires slot 'segments'"):
        egolearn.build(with_visual)
    stray = _pairs_only(iid_config)
    stray["driver"]["batch"]["storyboard_json"] = iid_config["driver"]["batch"]["storyboard_json"]
    with pytest.raises(ValueError, match="pairs-only"):
        egolearn.build(stray)
