"""Static-image transfer preserves video features, checkpoint identity and probe math."""

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from egolearn.evaluation import evaluate, evaluate_checkpoint
from egolearn.evaluation.linear_probe import LinearProbe
from egolearn.methods.vjepa2 import VJEPA2Encoder
from egolearn.methods.vjepa2.evaluation import StaticClipEncoder, VJEPA2ImageBenchmark
from egolearn.model.build import build_model
from egolearn.model.model import ComposedModel


def test_static_clip_matches_explicit_video_and_preserves_spatial_tokens():
    encoder = VJEPA2Encoder(dim=32, depth=1, heads=2, patch=8, tubelet=2).eval()
    images = torch.randn(2, 3, 16, 16)
    actual = StaticClipEncoder(encoder, 4)(images)
    expected = encoder(images[:, None].repeat(1, 4, 1, 1, 1))
    torch.testing.assert_close(actual.tokens, expected.tokens)
    torch.testing.assert_close(actual.pooled, expected.tokens.mean(1))
    assert actual.tokens.shape == (2, 8, 32)
    with pytest.raises(ValueError, match="divisible"):
        StaticClipEncoder(encoder, 3)
    with pytest.raises(ValueError, match="shape"):
        StaticClipEncoder(encoder)(images[:, None])


@pytest.fixture
def image_probe(tmp_path):
    split: dict[str, Any] = {"classes": ["red", "blue"], "train": [], "test": []}
    for label, color in enumerate(((240, 10, 10), (10, 10, 240))):
        for index in range(4):
            image = tmp_path / f"{label}-{index}.png"
            Image.new("RGB", (16, 20), color).save(image)
            split["train" if index < 2 else "test"].append({"image": str(image), "label": label})
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(split))
    config = {
        "encoders": {
            name: {
                "_target_": "encoder:vjepa2",
                "dim": 32,
                "depth": 1,
                "heads": 2,
                "patch": 8,
                "tubelet": 2,
                **({"frozen": True} if name == "video_target" else {}),
            }
            for name in ("video", "video_target")
        }
    }
    model = build_model(config, imports=["egolearn.methods.vjepa2"])
    spec = {
        "_target_": "benchmark:vjepa2_image",
        "name": "objects",
        "probe": "linear",
        "num_frames": 4,
        "probe_kwargs": {"manifest": str(manifest), "size": 16, "batch_size": 2, "max_iter": 20},
    }
    return model, spec


def test_wrapped_probe_matches_direct_image_probe_without_mutating_model(image_probe):
    model, spec = image_probe
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    gradients = [p.requires_grad for p in model.parameters()]
    calls = []
    handle = model.encoders["video_target"].register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape)
    )
    online = model.encoders["video"].register_forward_pre_hook(
        lambda *_: pytest.fail("image evaluation used the online encoder instead of EMA")
    )
    result = evaluate(model, benchmarks=[spec], record_json=None)
    online.remove()
    handle.remove()
    assert calls and all(shape[1:] == (4, 3, 16, 16) for shape in calls)
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == gradients
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    direct = ComposedModel({"image": StaticClipEncoder(model.encoders["video_target"], 4)})
    expected = evaluate(
        direct,
        benchmarks=[LinearProbe(name="objects", **spec["probe_kwargs"])],
        record_json=None,
    )
    assert result.results == expected.results
    assert result.configs["objects"]["probe_kwargs"]["l2"] == 0.001
    assert result.metadata["objects"]["n_test"] == 4
    assert result.metadata["objects"]["encoder_frozen"] is True
    assert result.configs["objects"]["probe_kwargs"]["manifest"] == "split.json"
    json.dumps(result.to_dict(), allow_nan=False)


def test_checkpoint_image_evaluation_records_original_checkpoint(image_probe, tmp_path):
    model, spec = image_probe
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "model_config": model.construction_config}, checkpoint)
    result = evaluate_checkpoint(
        checkpoint, benchmarks=[spec], record_json=tmp_path / "record.json"
    )
    assert result.status == "complete"
    assert result.checkpoint is not None
    assert result.checkpoint["filename"] == "model.pth"
    assert result.configs["objects"]["encoder"] == "video_target"
    assert result.configs["objects"]["num_frames"] == 4
    assert result.model_config == model.construction_config
    record = json.loads((tmp_path / "record.json").read_text())
    assert record["status"] == "completed"


def test_image_eval_recipe_resolves_explicit_protocol(monkeypatch):
    for key in [
        "LABELED_S_ROOT",
        "LABELED_S_SPLIT",
        "BABYVIEW_LINEAR_MANIFEST",
        "SHAPE_BIAS_MANIFEST",
    ]:
        monkeypatch.setenv(key, f"/data/{key.lower()}")
    directory = Path(__file__).resolve().parents[1] / "projects/vjepa2/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        config = OmegaConf.to_container(
            compose(config_name="image_evals", overrides=["checkpoint=/models/final.pth"]),
            resolve=True,
        )
    assert isinstance(config, dict)
    for module in config["_imports_"]:
        importlib.import_module(module)
    assert [s["probe"] for s in config["benchmarks"]] == [
        "labeled_s_linear",
        "linear",
        "shape_bias",
    ]
    for spec in config["benchmarks"]:
        assert spec["encoder"] == "video_target"
        assert spec["num_frames"] == 16
        assert spec["probe_kwargs"]["size"] == 128
        benchmark = VJEPA2ImageBenchmark(**{k: v for k, v in spec.items() if k != "_target_"})
        assert benchmark.configuration()["probe"] == spec["probe"]


def test_static_probe_rejects_text_and_projection_heads():
    with pytest.raises(ValueError, match="supported"):
        VJEPA2ImageBenchmark("bad", "four_afc", {})
    with pytest.raises(ValueError, match="without a head"):
        VJEPA2ImageBenchmark("bad", "linear", {"head": "proj"})
