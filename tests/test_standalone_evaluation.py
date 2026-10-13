"""Checkpoint evaluation without training data, and benchmark lifecycle guarantees."""

import hashlib
import json
import random
import subprocess
import sys
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

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
from egolearn.evaluation.evaluator import _configured_benchmark
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

        def download(*, repo_id, repo_type, revision, filename):
            assert (repo_id, repo_type, revision) == ("org/model", "model", "v1")
            cache = tmp_path / "snapshots" / ("a" * 40)
            cache.mkdir(parents=True)
            (cache / filename).symlink_to(path)
            return str(cache / filename)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        checkpoint = "hf://org/model@v1/checkpoint.pth"
    result = evaluate_checkpoint(
        checkpoint, benchmarks=[WeightBenchmark()], trusted_snapshot=snapshot
    )
    assert result.results == expected
    assert result.checkpoint is not None
    if hub_path:
        assert result.checkpoint["commit"] == "a" * 40
        assert result.checkpoint["reference"] == checkpoint
        assert result.checkpoint["repo"] == "org/model"
    assert result.checkpoint["filename"] == path.name
    assert str(tmp_path) not in json.dumps(result.checkpoint)
    assert result.checkpoint["step"] == 17
    assert result.checkpoint["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.model_config == pipe.model.construction_config


@pytest.mark.parametrize("probe_name", ["vtwt", "baby_winoground", "four_afc"])
@pytest.mark.parametrize("path_kind", ["absolute", "relative", "hub"])
def test_probe_configs_record_portable_paths(tmp_path, monkeypatch, probe_name, path_kind):
    (tmp_path / "labels.csv").write_text("image,positive_phrase,negative_phrase\n")
    (tmp_path / "vocab.json").write_text('{"<unk>": 0}')
    (tmp_path / "trials.json").write_text('{"trials": []}')
    monkeypatch.chdir(tmp_path)
    hub_prefix = "hf://org/benchmark@v1/"
    monkeypatch.setattr(
        sys.modules["egolearn.config.instantiate"],
        "resolve",
        lambda reference: tmp_path / reference.removeprefix(hub_prefix),
    )
    paths = {"vocab_json": "vocab.json"}
    if probe_name == "four_afc":
        target = "egolearn.evaluation.four_afc.FourAFCProbe"
        paths["manifest"] = "trials.json"
    else:
        class_name = "VTWTProbe" if probe_name == "vtwt" else "BabyWinogroundProbe"
        target = f"egolearn.evaluation.benchmarks.{probe_name}.{class_name}"
        paths["csv_path"] = "labels.csv"
        if probe_name == "vtwt":
            paths["image_dir"] = "images"
        else:
            paths.update(pos_dir="positive", neg_dir="negative")
        paths["output_json"] = "scores.json"
    arguments = {
        key: str(tmp_path / name)
        if path_kind == "absolute"
        else hub_prefix + name
        if path_kind == "hub"
        else f"./{name}"
        for key, name in paths.items()
    }
    spec = {"_target_": target, "name": probe_name, "vision_head": "embed", **arguments}
    original = deepcopy(spec)

    benchmark, config = _configured_benchmark(spec)

    assert spec == original
    assert benchmark.name == probe_name
    assert config["size"] == 224  # constructor defaults are still recorded
    for key, name in paths.items():
        assert config[key] == (name if path_kind == "absolute" else arguments[key])
        if key.endswith("_dir"):
            expected = f"./{name}" if path_kind == "relative" else str(tmp_path / name)
            assert getattr(benchmark, key) == expected
    assert str(tmp_path) not in json.dumps(config)


@pytest.mark.parametrize("via_mapping", [False, True], ids=["instance", "mapping"])
def test_saved_benchmark_configs_preserve_structured_settings(
    tmp_path, monkeypatch, model_config, via_mapping
):
    SourcePair = namedtuple("SourcePair", ["train", "test"])
    settings = {
        "classes": ["/m/01g317", "/a/cat", "/b/cat"],
        "options": ("relative/data.json", "hf://org/data@v1/manifest.json", None, 7, True),
        "pair": SourcePair("train", "test"),
    }
    original = deepcopy(settings)

    class ConfiguredBenchmark(Benchmark):
        def __init__(self, settings, output_dir=tmp_path / "output"):
            self.settings = settings
            self.output_dir = output_dir

        def configuration(self):
            return {"settings": self.settings, "output_dir": self.output_dir}

        def run(self, model, ctx):
            assert self.settings == original
            assert self.output_dir == tmp_path / "output"
            return {"score": 1.0}

    monkeypatch.setitem(BENCHMARKS._items, "test_portable_config", ConfiguredBenchmark)
    spec = (
        {"_target_": "benchmark:test_portable_config", "settings": settings}
        if via_mapping
        else ConfiguredBenchmark(settings)
    )
    result = evaluate(build_model(model_config), benchmarks=[spec])
    destination = tmp_path / "results.json"
    result.save_json(destination)
    recorded = json.loads(destination.read_text())["configs"]["benchmark"]

    assert recorded["settings"] == {
        "classes": original["classes"],
        "options": list(original["options"]),
        "pair": ["train", "test"],
    }
    assert recorded["output_dir"] == "output"
    assert str(tmp_path) not in destination.read_text()
    assert settings == original


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
    assert benchmark.split is not None
    assert first.metadata["imagenet"] == {
        "classes": ["class_0", "class_1"],
        "n_train": 4,
        "n_test": 4,
        "split_sha256": benchmark.split.metadata()["split_sha256"],
    }
    assert first.configs["imagenet"]["view"] == "imagenet_center_crop"
    assert first.configs["imagenet"]["data_root"] == imagefolder.name
    assert str(imagefolder.parent) not in json.dumps(first.to_dict())

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


@pytest.mark.parametrize("probe_type", ["linear", "linear_sgd"])
@pytest.mark.parametrize("hub_path", [False, True])
def test_existing_probe_configs_work_in_standalone_runner(
    tmp_path, imagefolder, model_config, probe_type, hub_path, monkeypatch
):
    split = imagenet.build_split(imagefolder, expected_classes=2)
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "classes": ["/m/01g317", "/m/0bt9lr"],
                "train": [{"image": item.image, "label": item.label} for item in split.train],
                "test": [{"image": item.image, "label": item.label} for item in split.test],
            }
        )
    )
    reference = "hf://org/benchmark@v1/split.json" if hub_path else str(manifest)
    monkeypatch.setattr(
        sys.modules["egolearn.config.instantiate"], "resolve", lambda reference: manifest
    )
    result = evaluate(
        build_model(model_config),
        benchmarks=[
            {
                "_target_": f"probe:{probe_type}",
                "name": "linear",
                "manifest": reference,
                **({"workers": 0, "epochs": 1} if probe_type == "linear_sgd" else {"max_iter": 3}),
                "view": "imagenet_center_crop",
                "size": 16,
                "batch_size": 4,
            }
        ],
    )
    assert result.configs["linear"]["view"] == "imagenet_center_crop"
    assert result.configs["linear"]["manifest"] == (reference if hub_path else manifest.name)
    assert result.configs["linear"]["split"]["classes"] == ["/m/01g317", "/m/0bt9lr"]
    assert result.configs["linear"]["split"]["classes"] == result.metadata["linear"]["classes"]
    assert str(tmp_path) not in json.dumps(result.to_dict())
    assert (
        result.metadata["linear"]["manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    output = tmp_path / "probe.json"
    result.save_json(output)
    assert json.loads(output.read_text())["results"] == result.results


@pytest.mark.parametrize("entrypoint", ["module", "console"])
def test_imagenet_cli_runs_from_outside_repository(tmp_path, imagefolder, model_config, entrypoint):
    model = build_model(model_config)
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "model_config": model.construction_config}, checkpoint)
    output = tmp_path / "result.json"
    command = (
        [sys.executable, "-m", "egolearn.evaluate", "--config-name", "imagenet_linear_eval"]
        if entrypoint == "module"
        else [str(Path(sys.executable).parent / "egolearn"), "eval", "imagenet_linear_eval"]
    )
    subprocess.run(
        [
            *command,
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
    assert result["configs"]["imagenet_linear"]["data_root"] == imagefolder.name
    assert str(tmp_path) not in json.dumps(result)
    assert result["versions"]["imagenet_linear"] == "1"
    assert result["status"] == "complete"
    assert result["failed_benchmark"] is None
    assert result["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


@pytest.mark.parametrize("trust", [None, False, True], ids=["default", "untrusted", "trusted"])
def test_eval_entry_point_records_snapshot_trust_outcome(
    tmp_path, toy_pipeline, monkeypatch, trust
):
    from pickle import UnpicklingError

    from egolearn.evaluate import main

    # Populate replay with Python event objects that need explicit snapshot trust.
    toy_pipeline.driver.max_steps = 1
    toy_pipeline.driver.fit()
    checkpoint = tmp_path / "snapshot.pth"
    save_snapshot(str(checkpoint), toy_pipeline.driver)
    monkeypatch.setitem(BENCHMARKS._items, "test_weights", WeightBenchmark)
    output = tmp_path / "evaluation.json"
    record_path = tmp_path / "record.json"
    config = {
        "checkpoint": str(checkpoint),
        "benchmarks": [{"_target_": "benchmark:test_weights"}],
        "output_json": str(output),
        "record_json": str(record_path),
    }
    if trust is not None:
        config["trusted_snapshot"] = trust
    cfg = OmegaConf.create(config)
    if trust:
        main.__wrapped__(cfg)
        result = json.loads(output.read_text())
        record = json.loads(record_path.read_text())
        assert result["status"] == "complete"
        assert record["status"] == "completed"
        assert record["metrics"] == {"weights/sum": result["results"]["weights"]["sum"]}
        assert record["checkpoint"] == result["checkpoint"]
    else:
        with pytest.raises(UnpicklingError):
            main.__wrapped__(cfg)
        assert not output.exists()
        record = json.loads(record_path.read_text())
        assert record["status"] == "failed"
        assert record["error_type"] == "UnpicklingError"


@pytest.mark.parametrize("stage", ["prepare", "run", "metadata"])
def test_failed_benchmark_keeps_completed_results_and_checkpoint(
    tmp_path, model_config, monkeypatch, stage
):
    from egolearn.evaluation import EvaluationError

    model = build_model(model_config)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": model.state_dict(), "model_config": model.construction_config}, path)
    failed = RandomBenchmark("failed")
    cause = ValueError("broken benchmark")

    def fail(*args):
        raise cause

    monkeypatch.setattr(failed, stage, fail)
    with pytest.raises(EvaluationError) as caught:
        evaluate_checkpoint(path, benchmarks=[WeightBenchmark(), failed])
    error = caught.value
    assert error.benchmark == "failed"
    assert error.__cause__ is cause
    result = error.partial_result
    assert result.checkpoint is not None
    assert result.checkpoint["filename"] == path.name
    for field in (result.results, result.configs, result.versions, result.metadata):
        assert set(field) == {"weights"}
    assert set(result.code_revision) == {"commit", "dirty"}
    assert result.status == "partial"
    assert result.failed_benchmark == "failed"
    output = tmp_path / "partial.json"
    result.save_json(output)
    assert json.loads(output.read_text())["results"] == result.results


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("completed_first", [False, True])
def test_nonfinite_metric_preserves_serializable_partial_results(
    tmp_path, model_config, monkeypatch, value, completed_first
):
    from egolearn.evaluation import EvaluationError

    failed = RandomBenchmark("nonfinite")
    monkeypatch.setattr(failed, "run", lambda model, ctx: {"valid": 1.0, "invalid": value})
    with pytest.raises(EvaluationError) as caught:
        evaluate(
            build_model(model_config),
            benchmarks=[*([WeightBenchmark()] if completed_first else []), failed],
        )
    assert isinstance(caught.value.__cause__, ValueError)
    assert "metric 'invalid' must be finite" in str(caught.value.__cause__)
    result = caught.value.partial_result
    assert result.status == "partial" and result.failed_benchmark == "nonfinite"
    for field in (result.results, result.configs, result.versions, result.metadata):
        assert set(field) == ({"weights"} if completed_first else set())
    output = tmp_path / "partial.json"
    result.save_json(output)
    assert json.loads(output.read_text()) == result.to_dict()


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_mps_rng_on_success_and_failure(model_config, monkeypatch, fail):
    from egolearn.evaluation import EvaluationError

    # Emulate the MPS RNG on CPU CI, including torch.manual_seed's MPS side effect.
    state = torch.tensor([11], dtype=torch.uint8)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: state.clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda value: state.copy_(value))
    monkeypatch.setattr(torch.mps, "manual_seed", lambda seed: state.fill_(seed))
    model = build_model(model_config)
    if fail:
        with pytest.raises(EvaluationError):
            evaluate(model, benchmarks=[RandomBenchmark(fail=True)], seed=23)
    else:
        evaluate(model, benchmarks=[RandomBenchmark()], seed=23)
    assert state.item() == 11


@pytest.mark.parametrize("completed_first", [False, True])
@pytest.mark.parametrize("record_enabled", [False, True])
def test_cli_writes_partial_results_and_exits_with_failure(
    tmp_path, imagefolder, model_config, completed_first, record_enabled
):
    model = build_model(model_config)
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "model_config": model.construction_config}, checkpoint)
    benchmark = {
        "_target_": "benchmark:imagenet_linear_sgd",
        "name": "finished",
        "data_root": str(imagefolder),
        "expected_classes": 2,
        "size": 16,
        "workers": 0,
        "batch_size": 4,
        "epochs": 1,
    }
    output = tmp_path / "partial.json"
    record_path = tmp_path / "record.json"
    OmegaConf.save(
        {
            "checkpoint": str(checkpoint),
            "output_json": str(output),
            "record_json": str(record_path) if record_enabled else None,
            "_imports_": ["egolearn.evaluation.benchmarks.imagenet_linear"],
            "benchmarks": [
                *([benchmark] if completed_first else []),
                {**benchmark, "name": "failed", "data_root": "missing"},
            ],
        },
        tmp_path / "partial.yaml",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "egolearn.evaluate",
            "--config-path",
            str(tmp_path),
            "--config-name",
            "partial",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "FileNotFoundError" in completed.stderr
    result = json.loads(output.read_text())
    assert set(result["results"]) == ({"finished"} if completed_first else set())
    assert result["checkpoint"]["filename"] == checkpoint.name
    assert result["status"] == "partial"
    assert result["failed_benchmark"] == "failed"
    if record_enabled:
        record = json.loads(record_path.read_text())
        assert record["status"] == "failed"
        assert record["error_type"] == "EvaluationError"
        assert record["wall_seconds"] >= 0
    else:
        assert not record_path.exists()


@pytest.mark.parametrize("probe_type", ["linear", "linear_sgd"])
def test_constructed_probes_record_protocol_and_split(
    imagefolder, model_config, tmp_path, probe_type
):
    from egolearn.evaluation.linear_probe import LinearProbe

    split = imagenet.build_split(imagefolder, expected_classes=2)
    common: dict = dict(name="linear", split=split, size=16, batch_size=4)
    probe: LinearProbe | LinearSGDProbe
    if probe_type == "linear":
        probe = LinearProbe(**common, l2=0.03, max_iter=3)
    else:
        probe = LinearSGDProbe(**common, seed=19, workers=0, epochs=1, lr=0.1)
    result = evaluate(build_model(model_config), benchmarks=[probe], seed=31)
    config = result.configs["linear"]
    assert config["size"] == 16
    assert config["batch_size"] == 4
    if probe_type == "linear":
        assert config["l2"] == 0.03 and config["max_iter"] == 3
    else:
        assert config["seed"] == 19 and config["lr"] == 0.1
        assert config["larc_trust_coefficient"] == 0.001
    assert result.metadata["linear"] == split.metadata()
    assert result.metadata["linear"]["n_train"] == 4
    assert str(imagefolder) not in json.dumps(result.metadata)
    result.save_json(tmp_path / "result.json")


def test_evaluation_error_keeps_partial_results_through_pickling(model_config):
    import pickle

    from egolearn.evaluation import EvaluationError

    with pytest.raises(EvaluationError) as caught:
        evaluate(
            build_model(model_config), benchmarks=[WeightBenchmark(), RandomBenchmark(fail=True)]
        )
    restored = pickle.loads(pickle.dumps(caught.value))
    assert restored.benchmark == "random"
    assert restored.partial_result.to_dict() == caught.value.partial_result.to_dict()
    assert str(restored) == str(caught.value)
