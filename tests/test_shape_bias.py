from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

import egolearn.evaluation.benchmarks.shape_bias.stimuli as shape_bias_stimuli
from egolearn.evaluation.benchmarks.shape_bias import (
    ShapeBiasProbe,
    prepare_shape_bias_stimuli,
    score_shape_bias,
    validate_stimulus_manifest,
)
from egolearn.evaluation.benchmarks.shape_bias.stimuli import sha256_file
from egolearn.evaluation.evaluator import _configured_benchmark
from egolearn.model.encoders.base import EncoderOutput


def _make_source_assets(root: Path, monkeypatch) -> None:
    checksums: dict[str, str] = {}
    for shape_index in range(16):
        path = root / "novel-masks" / f"shape{shape_index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10 + shape_index, 20, 100 + shape_index, 200), fill="black")
        image.save(path)
        checksums[f"novel-masks/{path.name}"] = sha256_file(path)
    for texture_index in range(16):
        path = root / "brodatz-textures" / f"texture{texture_index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.indices((300, 300)).sum(axis=0)
        values = ((values + texture_index * 7) % 32 * 8).astype(np.uint8)
        Image.fromarray(values).convert("RGB").save(path)
        checksums[f"brodatz-textures/{path.name}"] = sha256_file(path)
    monkeypatch.setattr(shape_bias_stimuli, "ASSET_SHA256", checksums)


def test_stimulus_generation_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    _make_source_assets(assets, monkeypatch)
    first = prepare_shape_bias_stimuli(assets, tmp_path / "first", seed=1)
    second = prepare_shape_bias_stimuli(assets, tmp_path / "second", seed=1)

    assert first["n_stimuli"] == 256
    assert first["n_triplets"] == 57_600
    assert len(first["shapes"]) == len(first["textures"]) == 16
    assert [record["sha256"] for record in first["stimuli"]] == [
        record["sha256"] for record in second["stimuli"]
    ]
    assert all(0 <= record["offset_x"] <= 44 for record in first["stimuli"])
    assert all(0 <= record["offset_y"] <= 44 for record in first["stimuli"])
    validate_stimulus_manifest(first, tmp_path / "first")

    with (tmp_path / "first" / "stimulus_manifest.json").open(encoding="utf-8") as handle:
        assert json.load(handle) == first


def test_shape_only_embeddings_have_complete_shape_bias() -> None:
    embeddings = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    metrics = score_shape_bias(embeddings).metrics
    assert metrics["n_triplets"] == 4
    assert metrics["shape_bias"] == 1.0
    assert metrics["ties"] == 0


def test_texture_only_embeddings_have_zero_shape_bias() -> None:
    embeddings = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    metrics = score_shape_bias(embeddings).metrics
    assert metrics["shape_bias"] == 0.0
    assert metrics["ties"] == 0


def test_ties_count_as_texture_decisions() -> None:
    metrics = score_shape_bias(np.ones((2, 2, 3))).metrics
    assert metrics["shape_bias"] == 0.0
    assert metrics["texture_bias"] == 1.0
    assert metrics["ties"] == 4


def test_vectorized_scorer_matches_independent_loop() -> None:
    embeddings = np.random.default_rng(7).normal(size=(4, 5, 11))
    vectorized = score_shape_bias(embeddings).metrics
    normalized = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    margins = []
    for shape_index in range(4):
        for texture_index in range(5):
            anchor = normalized[shape_index, texture_index]
            for other_texture in range(5):
                if other_texture == texture_index:
                    continue
                shape_cosine = float(anchor @ normalized[shape_index, other_texture])
                for other_shape in range(4):
                    if other_shape != shape_index:
                        texture_cosine = float(anchor @ normalized[other_shape, texture_index])
                        margins.append(shape_cosine - texture_cosine)
    expected = np.asarray(margins)
    assert vectorized["n_triplets"] == len(expected)
    assert vectorized["shape_decisions"] == int(np.sum(expected > 0))
    assert vectorized["ties"] == int(np.sum(expected == 0))
    assert np.isclose(vectorized["mean_cosine_margin"], expected.mean(), atol=1e-15)


class _FakeModel:
    def __init__(self) -> None:
        self.next_index = 0

    def encode(self, name, inputs):
        assert name == "image"
        indices = torch.arange(
            self.next_index,
            self.next_index + len(inputs),
            device=inputs.device,
            dtype=torch.float32,
        )
        self.next_index += len(inputs)
        return EncoderOutput(pooled=torch.stack([torch.ones_like(indices), indices + 1], dim=1))


def test_probe_writes_compact_artifact(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    _make_source_assets(assets, monkeypatch)
    stimulus_root = tmp_path / "stimuli"
    prepare_shape_bias_stimuli(assets, stimulus_root, seed=1)
    output_path = tmp_path / "shape_bias.json"
    probe, config = _configured_benchmark(
        {
            "_target_": "probe:shape_bias",
            "name": "shape_bias",
            "manifest": str(stimulus_root / "stimulus_manifest.json"),
            "batch_size": 31,
            "output_json": str(output_path),
        }
    )
    assert isinstance(probe, ShapeBiasProbe)
    assert config["manifest"] == "stimulus_manifest.json"
    assert config["output_json"] == output_path.name
    assert str(tmp_path) not in json.dumps(config)

    metrics = probe.run(_FakeModel(), SimpleNamespace(device=torch.device("cpu")))

    assert metrics["shape_bias_n"] == 57_600.0
    assert output_path.is_file()
    with output_path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert artifact["n_stimuli"] == 256
    assert artifact["n_triplets"] == 57_600
    assert artifact["shape_decisions"] + artifact["texture_decisions"] == 57_600
    assert len(artifact["anchors"]) == 256
    metadata = probe.metadata()
    assert metadata["n_images"] == 256
    assert metadata["n_triplets"] == 57600
    assert metadata["manifest_sha256"] == artifact["manifest_sha256"]
    # Metadata identifies the bytes actually loaded, even if the file changes later.
    with probe.manifest_path.open("a") as handle:
        handle.write("\n")
    assert probe.metadata() == metadata


def test_shape_bias_runs_through_static_clip_wrapper(tmp_path: Path, monkeypatch) -> None:
    from egolearn.evaluation import evaluate
    from egolearn.methods.vjepa2 import VJEPA2Encoder
    from egolearn.methods.vjepa2.evaluation import VJEPA2ImageBenchmark
    from egolearn.model.model import ComposedModel

    assets = tmp_path / "assets"
    _make_source_assets(assets, monkeypatch)
    stimulus_root = tmp_path / "stimuli"
    prepare_shape_bias_stimuli(assets, stimulus_root, seed=1)
    model = ComposedModel(
        {"video_target": VJEPA2Encoder(dim=32, depth=1, heads=2, patch=8, tubelet=2)},
        frozen=("video_target",),
    )
    benchmark = VJEPA2ImageBenchmark(
        name="shape_bias",
        probe="shape_bias",
        num_frames=4,
        probe_kwargs={"manifest": str(stimulus_root / "stimulus_manifest.json"), "size": 16},
    )
    result = evaluate(model, benchmarks=[benchmark], record_json=None)
    assert result.results["shape_bias"]["shape_bias_n"] == 57600
    assert result.metadata["shape_bias"]["n_images"] == 256
    assert result.metadata["shape_bias"]["encoder_frozen"] is True
    assert result.metadata["shape_bias"]["manifest_sha256"] == sha256_file(
        stimulus_root / "stimulus_manifest.json"
    )


def test_packaged_eval_config_uses_hosted_stimuli(tmp_path: Path, monkeypatch) -> None:
    import sys

    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from egolearn.evaluation.evaluator import _configured_benchmark

    assets = tmp_path / "assets"
    _make_source_assets(assets, monkeypatch)
    prepare_shape_bias_stimuli(assets, tmp_path / "repo", seed=1)
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="shape_bias_eval"), resolve=True)
    assert isinstance(cfg, dict)
    spec = cfg["benchmarks"][0]
    prefix = "hf://datasets/egolearn/shape-bias@v1/"
    assert spec["manifest"] == prefix + "stimulus_manifest.json"
    monkeypatch.setattr(
        sys.modules["egolearn.config.instantiate"],
        "resolve",
        lambda reference: tmp_path / "repo" / reference.removeprefix(prefix),
    )

    probe, config = _configured_benchmark(spec)

    assert isinstance(probe, ShapeBiasProbe)
    assert len(probe.records) == 256
    assert config["manifest"] == spec["manifest"]
