"""Exercise the default Storyboard objectives with a synthetic frame stream."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import egolearn
from egolearn.core import keys as K
from egolearn.model.losses import NTXent, SupervisedContrastive
from egolearn.training.hooks import Hook


def test_default_storyboard_trains_simclr_and_replay_supcon(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config_dir = Path(__file__).resolve().parents[1] / "projects/storyboard/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="saycam"), resolve=True)
    assert isinstance(cfg, dict)
    cfg.update(
        device="cpu",
        run_dir=str(tmp_path / "run"),
        loader={"workers": 0},
        hooks={},
        stream={
            "_target_": "stream:toy",
            "n_sessions": 1,
            "blocks_per_session": 6,
            "block_len": 4,
            "img_size": 16,
        },
    )
    for view in cfg["views"].values():
        view["size"] = 16
    cfg["admission"][1]["class_length"] = 4
    cfg["admission"][2].update(
        window_frames=4, window_stride=1, class_length=4, prototype_samples=2
    )
    tiers = cfg["memories"]["frames"]["tiers"]
    tiers["short"].update(capacity=4, admit_stride=1)
    tiers["long"]["capacity"] = 8
    for slot in cfg["recipe"]["slots"]:
        slot["n"] = 2 if slot["name"] == "curr" else 4
    cfg["model"]["encoders"]["vision"] = {"_target_": "encoder:toy_cnn", "dim": 16}
    for head in cfg["model"]["heads"].values():
        head["sizes"][:2] = [16, 16]
    cfg["driver"]["advance"]["n"] = 4
    cfg["driver"].update(max_steps=2, prefetch=False)

    pipe = egolearn.build(cfg)
    instance, temporal = pipe.losses
    assert isinstance(instance, NTXent) and instance.slots == ["curr", "replay"]
    assert isinstance(temporal, SupervisedContrastive) and temporal.slots == ["replay"]
    assert instance.temperature == temporal.temperature == 0.1
    assert instance.weight == temporal.weight == 1.0
    assert set(pipe.model.heads) == {"proj128", "segment128"}

    labels_seen = []
    compose_batch = pipe.driver.composer.compose

    def check_labels(*args, **kwargs):
        batch = compose_batch(*args, **kwargs)
        if batch is not None:
            replay = batch["replay"]
            expected = [
                pipe.label_stores["storyboard"].resolve(item[K.VIDEO_REF].key)
                for item in replay[K.SLOT_META]
            ]
            assert replay[K.SLOT_LABELS].tolist() == expected
            labels_seen.append(expected)
        return batch

    monkeypatch.setattr(pipe.driver.composer, "compose", check_labels)
    history = []

    class RecordLosses(Hook):
        def on_step(self, driver, logs):
            history.append(logs)

    pipe.driver.hooks.append(RecordLosses())
    before = {name: value.clone() for name, value in pipe.model.state_dict().items()}
    pipe.driver.fit()
    assert pipe.ctx.clock.step == len(history) == len(labels_seen) == 2
    assert pipe.driver.nonfinite_steps == 0
    for logs in history:
        assert logs["temporal/supcon"] > 0
        assert logs["loss"] == pytest.approx(logs["instance/ntxent"] + logs["temporal/supcon"])
    after = pipe.model.state_dict()
    for prefix in ("encoders.vision.", "heads.proj128.", "heads.segment128."):
        assert any(
            not torch.equal(before[name], value)
            for name, value in after.items()
            if name.startswith(prefix)
        )
    assert all(torch.isfinite(value).all() for value in after.values())
