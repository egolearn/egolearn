"""Checkpoint evaluation without training data, and benchmark lifecycle guarantees."""

import hashlib
import json
import random
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.config.registry import BENCHMARKS, ENCODERS, PROBES, STREAMS
from egolearn.evaluation import Benchmark, evaluate, evaluate_checkpoint, load_model
from egolearn.evaluation.benchmarks.imagenet_linear import benchmark as imagenet
from egolearn.evaluation.linear_sgd_probe import LinearSGDProbe
from egolearn.evaluation.probe import Probe
from egolearn.model.build import build_model
from egolearn.model.encoders.image import ToyCNNEncoder
from egolearn.training.hooks import CheckpointHook
from egolearn.utils.snapshot import save_snapshot


class WeightBenchmark(Benchmark):
    name = "weights"
    requires_capabilities = ("image_representation",)

    def run(self, model, ctx):
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        return {"sum": sum(float(value.sum()) for value in model.state_dict().values())}


class RandomBenchmark(Benchmark):
    def __init__(self, name="random", fail=False):
        self.name = name
        self.fail = fail

    def run(self, model, ctx):
        value = random.random() + float(np.random.random()) + float(torch.rand(()))  # noqa: NPY002
        if self.fail:
            raise RuntimeError("evaluation failed")
        return {"random": value, "derived": float(ctx.derive_seed("sample") % 100)}


@pytest.fixture
def model_config():
    return {
        "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 8}},
        "heads": {
            "proj": {"encoder": "vision", "_target_": "head:linear", "in_dim": 8, "out_dim": 4}
        },
    }


@pytest.fixture
def toy_pipeline(tmp_path):
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="toy_streaming")
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    plain["run_dir"] = str(tmp_path)
    return egolearn.build(plain)


@pytest.fixture
def imagefolder(tmp_path):
    root = tmp_path / "images"
    for split in ("train", "val"):
        for label, color in enumerate(((230, 20, 20), (20, 20, 230))):
            directory = root / split / f"class_{label}"
            directory.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (24, 28), color).save(directory / f"{index}.png")
    return root


@pytest.fixture
def legacy_text_checkpoint(tmp_path, model_config):
    vocab = {"<pad>": 0, "<unk>": 1, "red": 2, "ball": 3}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    config = {
        "model": deepcopy(model_config),
        "_imports_": ["egolearn.methods.babycl"],
        "vocab_path": str(vocab_path),
        "tokenizer": {"vocab_json": "${vocab_path}", "normalize": False},
    }
    config["model"]["encoders"]["text"] = {
        "_target_": "encoder:text_bow",
        "dim": 8,
        "vocab_size": "auto",
    }
    model = build_model(config["model"], vocab_size=len(vocab), imports=config["_imports_"])
    path = tmp_path / "legacy_text.pth"
    # Older checkpoints record weights and routing, but no construction config.
    torch.save({"model": model.state_dict(), "export_contract": model.export_contract()}, path)
    return path, config, model


@pytest.mark.parametrize("hub_path", [False, True])
@pytest.mark.parametrize("snapshot", [False, True])
def test_checkpoint_round_trip_without_building_training_pipeline(
    toy_pipeline, tmp_path, monkeypatch, snapshot, hub_path
):
    pipe = toy_pipeline
    pipe.ctx.clock.step = 17
    path = tmp_path / "checkpoint.pth"
    if snapshot:
        save_snapshot(str(path), pipe.driver)
    else:
        CheckpointHook(str(tmp_path))._save(pipe.driver, path.name)
    expected = evaluate(pipe.model, benchmarks=[WeightBenchmark()]).results

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation attempted to construct or resume training")

    monkeypatch.setattr(egolearn, "build", forbidden)
    monkeypatch.setattr(STREAMS.get("toy"), "__init__", forbidden)
    monkeypatch.setattr("egolearn.utils.snapshot.load_snapshot", forbidden)
    monkeypatch.setattr("egolearn.training.solver.build_optimizer", forbidden)
    checkpoint = path
    if hub_path:
        import huggingface_hub

        def download(*, repo_id, repo_type, revision):
            assert (repo_id, repo_type, revision) == ("org/model", "model", "v1")
            return str(tmp_path)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
        checkpoint = "hf://org/model@v1/checkpoint.pth"
    result = evaluate_checkpoint(checkpoint, benchmarks=[WeightBenchmark()])
    assert result.results == expected
    assert result.checkpoint is not None
    assert result.checkpoint["step"] == 17
    assert result.checkpoint["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.model_config == pipe.model.construction_config


def test_legacy_checkpoint_uses_only_model_related_config(tmp_path, model_config, monkeypatch):
    model = build_model(model_config)
    path = tmp_path / "legacy.pth"
    torch.save({"model": model.state_dict()}, path)
    with pytest.raises(ValueError, match="supply model_config"):
        load_model(path)

    config_path = tmp_path / "config.yaml"
    monkeypatch.delenv("EGOLEARN_TEST_MISSING_DATA", raising=False)
    config_path.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "model": model_config,
                    "stream": {"root": "${oc.env:EGOLEARN_TEST_MISSING_DATA}"},
                    "probes": [{"_target_": "probe:does_not_exist"}],
                }
            )
        )
    )
    loaded = load_model(path, model_config=config_path)
    assert all(
        torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()
    )

    mismatched = deepcopy(model_config)
    mismatched["heads"]["proj"]["out_dim"] = 5
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_model(path, model_config=mismatched)


def test_legacy_checkpoint_resolves_auto_vocab_from_hydra_tokenizer(
    legacy_text_checkpoint, tmp_path, monkeypatch
):
    path, config, model = legacy_text_checkpoint
    monkeypatch.delenv("EGOLEARN_TEST_MISSING_DATA", raising=False)
    config["stream"] = {"root": "${oc.env:EGOLEARN_TEST_MISSING_DATA}"}
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)

    loaded = load_model(path, model_config=config_path)

    assert loaded.construction_config is not None
    assert loaded.construction_config["model"]["encoders"]["text"]["vocab_size"] == 4
    assert loaded.export_contract() == model.export_contract()
    assert loaded.state_dict().keys() == model.state_dict().keys()
    assert all(
        torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()
    )
    assert not loaded.training


@pytest.mark.parametrize("null_tokenizer", [False, True])
def test_legacy_checkpoint_auto_vocab_requires_tokenizer(legacy_text_checkpoint, null_tokenizer):
    path, config, _ = legacy_text_checkpoint
    if null_tokenizer:
        config["tokenizer"] = None
    else:
        del config["tokenizer"]

    with pytest.raises(
        ValueError, match="vocab_size=auto needs a tokenizer or a resolved vocabulary size"
    ):
        load_model(path, model_config=config)


def test_checkpoint_rejects_different_routing_with_matching_weights(tmp_path, model_config):
    config = deepcopy(model_config)
    config["encoders"]["target"] = {"_target_": "encoder:toy_cnn", "dim": 8, "frozen": True}
    model = build_model(config)
    path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.construction_config,
            "export_contract": model.export_contract(),
        },
        path,
    )
    mismatched = deepcopy(config)
    mismatched["heads"]["proj"]["encoder"] = "target"
    other = build_model(mismatched)
    # Strict weight loading accepts this model, but its head reads a different encoder.
    other.load_state_dict(model.state_dict(), strict=True)
    assert other.export_contract() != model.export_contract()

    with pytest.raises(
        ValueError, match="model_config does not match the checkpoint's export contract"
    ):
        load_model(path, model_config=mismatched)


def test_checkpoint_loading_skips_pretrained_initialization(tmp_path, monkeypatch):
    def encoder(pretrained=False):
        assert pretrained is False, "checkpoint loading requested pretrained weights"
        return ToyCNNEncoder(dim=8)

    monkeypatch.setitem(ENCODERS._items, "test_pretrained", encoder)
    config = {"encoders": {"vision": {"_target_": "encoder:test_pretrained", "pretrained": True}}}
    model = build_model(config, initialize_pretrained=False)
    path = tmp_path / "pretrained.pth"
    torch.save({"model": model.state_dict(), "model_config": model.construction_config}, path)
    loaded = load_model(path)
    assert loaded.state_dict().keys() == model.state_dict().keys()


def test_checkpoint_records_resolved_vocab_and_frozen_routing(tmp_path, model_config):
    config = deepcopy(model_config)
    config["encoders"]["target"] = {"_target_": "encoder:toy_cnn", "dim": 8, "frozen": True}
    config["encoders"]["text"] = {"_target_": "encoder:text_bow", "dim": 8, "vocab_size": "auto"}
    config["predictors"] = {
        "pred": {"_target_": "predictor:mlp", "inputs": ["vision"], "sizes": [8, 8]}
    }
    model = build_model(config, vocab_size=11, imports=["egolearn.methods.babycl"])
    assert model.construction_config is not None
    assert model.construction_config["model"]["encoders"]["text"]["vocab_size"] == 11
    assert config["encoders"]["text"]["vocab_size"] == "auto"
    path = tmp_path / "resolved.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.construction_config,
            "export_contract": model.export_contract(),
        },
        path,
    )
    loaded = load_model(path)
    assert loaded.export_contract() == model.export_contract()
    assert all(
        torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()
    )
    assert not any(p.requires_grad for p in loaded.encoders["target"].parameters())


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_preserves_rng_modes_weights_and_gradients(model_config, fail):
    model = build_model(model_config).train()
    model.encoders["vision"].eval()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    weights = deepcopy(model.state_dict())
    grad = parameter.grad.clone()
    py_state, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),  # noqa: NPY002
        torch.get_rng_state(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluate(model, benchmarks=[RandomBenchmark(fail=True)], seed=23)
    else:
        first = evaluate(model, benchmarks=[RandomBenchmark()], seed=23)
        second = evaluate(model, benchmarks=[RandomBenchmark("other"), RandomBenchmark()], seed=23)
        assert first.results["random"] == second.results["random"]
    assert modes == [module.training for module in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    assert torch.equal(grad, parameter.grad)
    assert random.getstate() == py_state
    current_np = np.random.get_state()  # noqa: NPY002
    assert isinstance(np_state, tuple) and isinstance(current_np, tuple)
    assert np_state[0] == current_np[0] and np.array_equal(np_state[1], current_np[1])
    assert np_state[2:] == current_np[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


def test_benchmark_validation_precedes_dataset_preparation(model_config, monkeypatch):
    model = build_model(model_config)
    benchmark = imagenet.ImageNetLinearBenchmark("imagenet", "missing")
    benchmark.requires_capabilities = ("text_representation",)

    def forbidden():
        raise AssertionError("prepared incompatible benchmark")

    monkeypatch.setattr(benchmark, "prepare", forbidden)
    with pytest.raises(ValueError, match="text_representation"):
        evaluate(model, benchmarks=[benchmark])
    with pytest.raises(ValueError, match="unique"):
        evaluate(model, benchmarks=[benchmark, benchmark])
    with pytest.raises(ValueError, match="at least one"):
        evaluate(model, benchmarks=[])


def test_imagenet_is_lazy_reuses_split_and_matches_shared_probe(
    imagefolder, model_config, monkeypatch
):
    calls = []
    build_split = imagenet.build_split

    def counted(*args, **kwargs):
        calls.append(1)
        return build_split(*args, **kwargs)

    monkeypatch.setattr(imagenet, "build_split", counted)
    settings: dict = dict(
        name="imagenet", expected_classes=2, size=16, workers=0, batch_size=4, epochs=2
    )
    benchmark = imagenet.ImageNetLinearBenchmark(data_root=str(imagefolder), **settings)
    assert calls == []
    assert not isinstance(benchmark, Probe)
    model = build_model(model_config)
    first = evaluate(model, benchmarks=[benchmark], seed=7)
    # The protocol seed is independent of the standalone runner seed.
    second = evaluate(model, benchmarks=[benchmark], seed=19)
    assert len(calls) == 1
    assert first.results == second.results
    assert first.metadata["imagenet"] == {
        "classes": ["class_0", "class_1"],
        "n_train": 4,
        "n_test": 4,
    }
    assert first.configs["imagenet"]["view"] == "imagenet_center_crop"

    settings.pop("expected_classes")
    probe = LinearSGDProbe(split=benchmark.split, **settings)
    assert evaluate(model, benchmarks=[probe], seed=31).results == first.results
    assert "imagenet_linear_sgd" not in PROBES.names()
    assert BENCHMARKS.get("imagenet_linear_sgd") is imagenet.ImageNetLinearBenchmark


def test_training_pipeline_rejects_standalone_benchmarks_without_scanning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("training scanned ImageNet")

    monkeypatch.setattr(imagenet, "build_split", forbidden)
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="toy_streaming"), resolve=True)
    assert isinstance(cfg, dict)
    cfg["probes"] = [
        {"_target_": "benchmark:imagenet_linear_sgd", "name": "imagenet", "data_root": "missing"}
    ]
    with pytest.raises(egolearn.PipelineError, match="standalone benchmarks"):
        egolearn.build(cfg)


def test_existing_probe_configs_work_in_standalone_runner(tmp_path, imagefolder, model_config):
    split = imagenet.build_split(imagefolder, expected_classes=2)
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "classes": split.classes,
                "train": [{"image": item.image, "label": item.label} for item in split.train],
                "test": [{"image": item.image, "label": item.label} for item in split.test],
            }
        )
    )
    result = evaluate(
        build_model(model_config),
        benchmarks=[
            {
                "_target_": "probe:linear_sgd",
                "name": "linear",
                "manifest": str(manifest),
                "workers": 0,
                "epochs": 1,
                "size": 16,
                "batch_size": 4,
            }
        ],
    )
    assert result.configs["linear"]["view"] == "imagenet_center_crop"
    output = tmp_path / "probe.json"
    result.save_json(output)
    assert json.loads(output.read_text())["results"] == result.results


def test_imagenet_cli_runs_from_outside_repository(tmp_path, imagefolder, model_config):
    model = build_model(model_config)
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "model_config": model.construction_config}, checkpoint)
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "egolearn.evaluate",
            "--config-name",
            "imagenet_linear_eval",
            f"checkpoint={checkpoint}",
            f"data_root={imagefolder}",
            f"output_json={output}",
            "device=cpu",
            "benchmarks.0.expected_classes=2",
            "benchmarks.0.workers=0",
            "benchmarks.0.size=16",
            "benchmarks.0.batch_size=4",
            "benchmarks.0.epochs=1",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert set(result["results"]["imagenet_linear"]) == {"top1", "train_top1_running"}
    assert result["configs"]["imagenet_linear"]["epochs"] == 1
    assert result["versions"]["imagenet_linear"] == "1"
    assert result["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
