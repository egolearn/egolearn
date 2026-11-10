"""Validate temporal targets under frame shuffling and resumed optimization."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.core import keys as K
from egolearn.data.decoding.base import VideoDecoder
from egolearn.data.sources.video import VideoStream
from egolearn.methods.babycl.temporal_iid import TemporalFrameDriver
from egolearn.training.hooks import Hook
from egolearn.utils.snapshot import load_snapshot, save_snapshot


class SyntheticDecoder(VideoDecoder):
    def probe(self, path):
        return 2.0, 8

    def read_window(self, path, i0, i1, indices=None):
        return [Image.new("RGB", (64, 64), (20 * (i0 + i), 80, 130)) for i in indices]


@pytest.fixture
def temporal_driver(tmp_path, toy_config):
    sessions = ["a.mp4", "b.mp4"]
    document = {
        "format_version": 1,
        "num_classes": 8,
        "sessions": {
            sid: {"num_frames": 8, "fps": 2.0, "segments": [[0, 3, 4], [3, 8, 1]]}
            for sid in sessions
        },
    }
    path = tmp_path / "temporal.json"
    path.write_text(json.dumps(document))

    def build(prefetch=False):
        config = toy_config(tmp_path, dim=8)
        config["loader"] = {"workers": 1 if prefetch else 0}
        pipe = egolearn.build(config)
        stream = VideoStream(str(tmp_path), SyntheticDecoder(), sessions=sessions)
        pipe.loader.stream = stream
        return TemporalFrameDriver(
            temporal_manifest=str(path),
            num_classes=8,
            stream=stream,
            loader=pipe.loader,
            model=pipe.model,
            losses=[pipe.losses[1]],
            optimizer=pipe.driver.optimizer,
            schedule=pipe.driver.schedule,
            ctx=pipe.ctx,
            slot="curr",
            n=4,
            views=["v1", "v2"],
            max_steps=6,
            prefetch=prefetch,
        )

    return build, path, document


def test_shuffled_frames_retain_targets_across_batch_boundaries(temporal_driver):
    build, _, _ = temporal_driver
    driver = build()
    refs = driver._build_index()
    assert len(refs) == 16
    order = list(range(16))
    driver.ctx.fresh_rng("offline.shuffle.epoch0").shuffle(order)
    seen: set[tuple] = set()
    for b in range(4):
        chosen = driver._pick(refs, order, b)
        batch = driver._compose(chosen, seq=b)["curr"]
        assert batch[K.SLOT_LABELS].tolist() == [4 if r.locator[1] < 3 else 1 for r in chosen]
        seen.update(r.key for r in chosen)
    assert len(seen) == 16
    assert driver.ctx.clock.compose_seq == 0
    driver.loader.close()


@pytest.mark.parametrize("prefetch", [False, True])
def test_temporal_updates_and_snapshot_resume(temporal_driver, tmp_path, prefetch):
    build, _, _ = temporal_driver
    driver = build(prefetch)
    before = copy.deepcopy(driver.model.state_dict())
    snapshot = str(tmp_path / "resume.pth")

    class Save(Hook):
        def on_step(self, driver, logs):
            assert logs["loss"] == pytest.approx(logs["temp/tcl"])
            if driver.ctx.clock.step == 2:
                save_snapshot(snapshot, driver)

    driver.hooks.append(Save())
    driver.fit()
    assert driver.ctx.clock.step == 6
    assert driver.nonfinite_steps == driver.skipped_steps == 0
    for prefix in ["encoders.vision.", "heads.cls."]:
        assert any(
            not torch.equal(v, before[k])
            for k, v in driver.model.state_dict().items()
            if k.startswith(prefix)
        )
    resumed = build(prefetch)
    load_snapshot(snapshot, resumed)
    resumed.fit()
    for key, value in driver.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key


@pytest.mark.parametrize(
    "change", ["gap", "overlap", "label", "count", "fps", "sessions", "classes"]
)
def test_incompatible_temporal_manifest_fails(temporal_driver, change):
    build, path, document = temporal_driver
    entry = document["sessions"]["a.mp4"]
    if change == "gap":
        entry["segments"][1][0] = 4
    elif change == "overlap":
        entry["segments"][1][0] = 2
    elif change == "label":
        entry["segments"][0][2] = 8
    elif change == "count":
        entry["num_frames"] = 7
        entry["segments"][-1][1] = 7
    elif change == "fps":
        entry["fps"] = 3.0
    elif change == "sessions":
        del document["sessions"]["b.mp4"]
    else:
        document["num_classes"] = 9
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="temporal"):
        build()._build_index()


@pytest.mark.parametrize("manifest_count,video_count", [(0, 0), (8, 0), (0, 8)])
def test_empty_sessions_are_validated(temporal_driver, manifest_count, video_count):
    build, path, document = temporal_driver
    entry = document["sessions"]["a.mp4"]
    entry["num_frames"] = manifest_count
    if manifest_count == 0:
        entry["segments"] = []
    path.write_text(json.dumps(document))
    driver = build()
    driver.stream._meta["a.mp4"] = {"fps": 2.0, "num_frames": video_count}
    try:
        if manifest_count != video_count:
            with pytest.raises(ValueError, match="temporal metadata"):
                driver._build_index()
        else:
            refs = driver._build_index()
            assert len(refs) == 8
            assert {ref.session for ref in refs} == {"b.mp4"}
    finally:
        driver.loader.close()


def test_resume_rejects_changed_temporal_targets_before_loading_weights(temporal_driver, tmp_path):
    build, path, document = temporal_driver
    original = build()
    with torch.no_grad():
        next(original.model.parameters()).add_(1)
    snapshot = str(tmp_path / "resume.pth")
    save_snapshot(snapshot, original)
    original.loader.close()
    document["sessions"]["a.mp4"]["segments"][0][2] = 2
    path.write_text(json.dumps(document))
    resumed = build()
    before = copy.deepcopy(resumed.model.state_dict())
    try:
        with pytest.raises(ValueError, match="temporal manifest"):
            load_snapshot(snapshot, resumed)
        for key, value in before.items():
            assert torch.equal(value, resumed.model.state_dict()[key]), key
    finally:
        resumed.loader.close()


@pytest.mark.parametrize("identity", ["matching", "changed", "missing"])
def test_legacy_snapshot_requires_matching_temporal_provenance(temporal_driver, tmp_path, identity):
    build, path, _ = temporal_driver
    original = build()
    snapshot = str(tmp_path / "legacy.pth")
    save_snapshot(snapshot, original)
    original.loader.close()
    state = torch.load(snapshot, weights_only=False)
    del state["driver"]["metadata"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {"driver.batch.temporal_manifest": {"sha256": digest}}
    if identity == "changed":
        inputs["driver.batch.temporal_manifest"]["sha256"] = "0" * 64
    elif identity == "missing":
        inputs = {}
    state["run_provenance"] = {"input_provenance": inputs}
    torch.save(state, snapshot)
    resumed = build()
    try:
        if identity == "matching":
            load_snapshot(snapshot, resumed)
        else:
            with pytest.raises(ValueError, match="temporal manifest"):
                load_snapshot(snapshot, resumed)
    finally:
        resumed.loader.close()


@pytest.mark.parametrize("method,head", [("simclr", "proj128"), ("tc", "cls")])
def test_iid_configs_have_only_the_requested_objective(method, head):
    configs = Path(__file__).resolve().parents[1] / "projects/babycl/configs"
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        cfg = compose(
            config_name=f"krishnacam_iid_{method}",
            overrides=["temporal_classes=8"] if method == "tc" else [],
        )
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    assert list(plain["model"]["heads"]) == [head]
    assert len(plain["losses"]) == 1
    assert plain["losses"][0]["weight"] == 1.0
    assert plain["driver"]["mode"] == "offline"
    assert plain["driver"]["batch"]["n"] == 128
    assert plain["driver"]["max_steps"] == 11900
    assert plain["admission"] == [] and plain["memories"] is None


def test_temporal_manifest_is_hashed_in_run_inputs(temporal_driver):
    from egolearn.utils.input_provenance import prepare_inputs

    _, path, _ = temporal_driver
    config = {"driver": {"batch": {"temporal_manifest": str(path)}}}
    inputs = prepare_inputs(config)
    assert (
        inputs["driver.batch.temporal_manifest"]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
