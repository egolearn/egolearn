"""Check accumulated fine-tuning updates and deferred metric validation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from egolearn.evaluation import Benchmark, EvaluationError, evaluate, evaluate_checkpoint
from egolearn.evaluation import full_finetune as ft
from egolearn.evaluation.benchmarks.imagenet_full_finetune import ImageNetFullFinetuneBenchmark
from egolearn.model.build import build_model


class TinyEncoder(nn.Sequential):
    out_dim = 8


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    encoder = TinyEncoder(nn.Flatten(), nn.Linear(12, 8))
    model = ft.Classifier(encoder, 3, "default", 0.0)
    images = torch.randn(11, 3, 2, 2)
    labels = torch.arange(11) % 3
    config = {
        "device": "cpu",
        "amp": False,
        "accumulation": 2,
        "batch_size": 3,
        "mixup": 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1,
        "clip_grad": 1.0,
        "warmup_epochs": 0,
        "epochs": 10,
        "lr": 0.01,
    }
    return model, images, labels, config


def test_epoch_streams_do_not_overlap_between_adjacent_run_seeds():
    from egolearn.evaluation.linear_probe import LabeledImage

    items = [LabeledImage(f"unused-{index}.png", index % 2) for index in range(32)]
    orders = []
    for seed, epoch in ((0, 1), (1, 0), (0, 1)):
        loader = ft.make_loader(
            items,
            None,
            {
                "seed": seed,
                "batch_size": 4,
                "workers": 0,
                "device": "cpu",
            },
            training=True,
            epoch=epoch,
        )
        assert loader.generator.initial_seed() == ft.epoch_seed(seed, epoch)
        orders.append(list(loader.sampler))
    assert orders[0] == orders[2]
    assert orders[0] != orders[1]


def test_layer_decay_keeps_final_normalization_at_the_head_learning_rate():
    class Encoder(nn.Module):
        out_dim = 4

        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
            self.pos_embed = nn.Parameter(torch.zeros(1, 2, 4))
            self.patch_embed = nn.Linear(3, 4)
            self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.norm = nn.LayerNorm(4)

    model = ft.Classifier(Encoder(), 2, "default", 0.0)
    assignments = {
        id(parameter): (group["lr_scale"], group["weight_decay"])
        for group in ft.parameter_groups(model, 0.1, 0.5)
        for parameter in group["params"]
    }
    params = dict(model.named_parameters())
    for name, expected in {
        "encoder.cls_token": (0.125, 0.0),
        "encoder.pos_embed": (0.125, 0.0),
        "encoder.patch_embed.weight": (0.125, 0.1),
        "encoder.blocks.0.weight": (0.25, 0.1),
        "encoder.blocks.1.weight": (0.5, 0.1),
        "encoder.norm.weight": (1.0, 0.0),
        "head.weight": (1.0, 0.1),
    }.items():
        assert assignments[id(params[name])] == expected


def test_deferred_metrics_preserve_updates_and_partial_batch(setup):
    model, images, labels, config = setup
    results = []
    states = []
    for deferred in (False, True):
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.SGD(ft.parameter_groups(candidate, 0.0, 1.0), lr=config["lr"])
        loader = DataLoader(TensorDataset(images, labels), batch_size=config["batch_size"])
        reports: list[tuple] = []
        metrics = ft.train_epoch(
            candidate,
            loader,
            optimizer,
            {**config, "defer_metrics": deferred},
            0,
            lambda *args, reports=reports: reports.append(args),
        )
        metrics.update(ft.accuracy(candidate, loader, {**config, "defer_metrics": deferred}))
        assert reports and isinstance(reports[0][2], float)
        results.append(metrics)
        states.append(copy.deepcopy(candidate.state_dict()))
    assert results[0] == results[1]
    assert results[0]["n_train"] == 11
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    # Explicit effective batches of 6 and 5 check partial-group normalization.
    optimizer = torch.optim.SGD(ft.parameter_groups(model, 0.0, 1.0), lr=config["lr"])
    for index, (start, end) in enumerate(((0, 6), (6, 11))):
        for group in optimizer.param_groups:
            group["lr"] = ft.learning_rate(config, (index + 1) / 2)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            model(images[start:end]), labels[start:end], label_smoothing=0.1
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[0][key], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("failure", ["loss", "gradient", "zero_gradient"])
def test_invalid_update_is_rejected_before_optimizer_step(setup, failure):
    model, images, labels, config = setup
    original = copy.deepcopy(model.state_dict())
    if failure == "loss":
        images[0].fill_(float("nan"))
    else:
        for parameter in model.parameters():
            parameter.register_hook(
                lambda grad: torch.full_like(grad, float("inf") if failure == "gradient" else 0)
            )
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    optimizer = torch.optim.SGD(ft.parameter_groups(model, 0.0, 1.0), lr=config["lr"])
    expected = RuntimeError if failure == "zero_gradient" else FloatingPointError
    with pytest.raises(expected):
        ft.train_epoch(model, loader, optimizer, {**config, "defer_metrics": True}, 0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_deferred_validation_rejects_nonfinite_logits(setup):
    model, images, labels, config = setup
    images[-1].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        ft.accuracy(
            model,
            DataLoader(TensorDataset(images, labels), batch_size=3),
            {**config, "defer_metrics": True},
        )


def test_kernel_preparation_preserves_checkpoint_keys(setup):
    model, images, _, config = setup
    state = copy.deepcopy(model.state_dict())
    config = {**config, "channels_last": True}
    assert ft.prepare_model(model, config) is model
    transferred = ft.move_images(images, config)
    assert transferred.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(model(images), model(transferred))
    model.load_state_dict(state, strict=True)


@pytest.fixture
def finetune_case(tmp_path):
    torch.set_num_threads(1)
    construction = {
        "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 8}},
        "heads": {"projection": {"encoder": "vision", "_target_": "head:mlp", "sizes": (8, 8)}},
    }
    model = build_model(construction)
    checkpoint = tmp_path / "input.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "export_contract": model.export_contract(),
            "model_config": {"model": construction},
        },
        checkpoint,
    )
    root = tmp_path / "images"
    for split in ("train", "val"):
        for label, color in enumerate(((200, 20, 20), (20, 20, 200))):
            directory = root / split / str(label)
            directory.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (24, 24), color).save(directory / f"{index}.png")
    spec = {
        "_target_": "benchmark:imagenet_full_finetune",
        "name": "finetune",
        "data_root": str(root),
        "expected_classes": 2,
        "encoder": "vision",
        "epochs": 2,
        "warmup_epochs": 0,
        "batch_size": 2,
        "accumulation": 2,
        "workers": 0,
        "lr": 0.001,
        "weight_decay": 0.01,
        "size": 16,
        "randaugment": False,
        "erase_probability": 0.0,
        "defer_metrics": True,
        "channels_last": True,
    }
    return model, checkpoint, spec


def benchmark_from(spec):
    return ImageNetFullFinetuneBenchmark(**{k: v for k, v in spec.items() if k != "_target_"})


def test_finetune_resume_matches_uninterrupted_and_checks_provenance(finetune_case, tmp_path):
    model, _, spec = finetune_case
    benchmark = benchmark_from(spec)
    benchmark.prepare()
    assert benchmark.split is not None
    config = {**benchmark.settings, "device": "cpu"}
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    expected = ft.fit(model, benchmark.split, config, output_dir=full_dir)
    with pytest.raises(ft.FineTunePaused):
        ft.fit(model, benchmark.split, config, output_dir=resumed_dir, max_epochs=1)
    with pytest.raises(ValueError, match="provenance changed"):
        ft.fit(model, benchmark.split, {**config, "accumulation": 1}, output_dir=resumed_dir)
    actual = ft.fit(model, benchmark.split, config, output_dir=resumed_dir)
    assert expected == actual
    full = torch.load(full_dir / "latest.pth", weights_only=True)
    resumed = torch.load(resumed_dir / "latest.pth", weights_only=True)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert ft.fit(model, benchmark.split, config, output_dir=resumed_dir) == actual
    # Even a completed output must reject a different evaluated encoder.
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match="provenance changed"):
        ft.fit(model, benchmark.split, config, output_dir=resumed_dir)


def test_registered_finetune_preserves_suite_model_and_rng(finetune_case, monkeypatch):
    import random

    import numpy as np

    model, _, spec = finetune_case
    model.train()
    first = next(model.parameters())
    first.requires_grad_(False)
    first.grad = torch.ones_like(first)
    state = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()  # noqa: NPY002
    autotuning = torch.backends.cudnn.benchmark

    class Unchanged(Benchmark):
        name = "unchanged"

        def run(self, candidate, ctx):
            for key, value in candidate.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            assert torch.backends.cudnn.benchmark == autotuning
            return {"ok": 1.0}

    # The packaged default selects the image modality, even when its name is vision.
    result = evaluate(
        model, benchmarks=[{**spec, "encoder": "image"}, Unchanged()], record_json=None
    )
    assert result.results["finetune"]["encoder_parameter_changed"] == 1
    assert result.results["finetune"]["n_test"] == 6
    assert result.results["unchanged"]["ok"] == 1
    assert result.versions["finetune"] == "1"
    assert len(result.metadata["finetune"]["split_sha256"]) == 64
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == flags
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    current_numpy = np.random.get_state()  # noqa: NPY002
    assert isinstance(current_numpy, tuple) and isinstance(numpy_rng, tuple)
    assert current_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(current_numpy[1], numpy_rng[1])
    assert current_numpy[2:] == numpy_rng[2:]

    # A failure after mutating the copy must still restore the caller's state.
    def fail(candidate, *args, **kwargs):
        next(candidate.parameters()).data.add_(10)
        raise RuntimeError("training failure")

    monkeypatch.setattr(ft, "train_epoch", fail)
    with pytest.raises(EvaluationError):
        evaluate(model, benchmarks=[{**spec, "cudnn_benchmark": not autotuning}], record_json=None)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert torch.backends.cudnn.benchmark == autotuning
    assert torch.equal(torch.get_rng_state(), rng)
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == flags


def test_finetune_checkpoint_evaluation_records_protocol_and_inputs(finetune_case, tmp_path):
    _, checkpoint, spec = finetune_case
    record_path = tmp_path / "record.json"
    result = evaluate_checkpoint(checkpoint, benchmarks=[spec], record_json=record_path)
    record = json.loads(record_path.read_text())
    assert record["status"] == "completed"
    assert record["evaluation"]["results"] == result.results
    assert record["input_provenance"]["checkpoint"]["sha256"] == ft.digest(checkpoint)
    assert record["input_provenance"]["benchmarks.0.data_root"]["kind"] == "directory"
    assert result.configs["finetune"]["accumulation"] == 2
    assert result.configs["finetune"]["compile_mode"] is None
    assert result.metadata["finetune"]["classes"] == ["0", "1"]
    assert str(tmp_path) not in json.dumps(result.to_dict())
    # Registered evaluation also accepts the standard safe model export.
    from egolearn.export import export_checkpoint

    exported = export_checkpoint(checkpoint, tmp_path / "export")
    safe_result = evaluate_checkpoint(exported, benchmarks=[spec], record_json=None)
    assert safe_result.results == result.results


def test_packaged_finetune_config_is_lazy_and_uses_suite_seed():
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from egolearn.config.instantiate import instantiate

    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(
            config_name="imagenet_full_finetune_eval",
            overrides=[
                "data_root=/missing/images",
                "checkpoint=/missing/model.pth",
                "seed=23",
            ],
        )
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    benchmark = instantiate(plain["benchmarks"][0])
    assert isinstance(benchmark, ImageNetFullFinetuneBenchmark)
    assert benchmark.split is None
    assert benchmark.settings["seed"] == 23
    with pytest.raises(ValueError, match="warmup"):
        ImageNetFullFinetuneBenchmark("bad", "/missing/images", epochs=1)


def test_full_finetune_runs_in_pinned_suite(finetune_case, tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from egolearn.commands import bench
    from egolearn.hub import sha256_file, write_checksums

    _, checkpoint, spec = finetune_case
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    root = Path(spec["data_root"])
    inventory = write_checksums(root)
    suite_path = tmp_path / "suite.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "version": 1,
                "name": "finetune",
                "entry": "eval",
                "config": "imagenet_full_finetune_eval",
                "seeds": [3, 5],
                "inputs": {
                    "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                    "data_root": {"path": str(root), "sha256": sha256_file(inventory)},
                },
                "overrides": [
                    "device=cpu",
                    *[
                        f"benchmarks.0.{key}={json.dumps(value)}"
                        for key, value in spec.items()
                        if key not in ("_target_", "name", "data_root")
                    ],
                ],
                "metrics": {"imagenet_full_finetune/top1": "max"},
            }
        ),
        suite_path,
    )
    manifest = bench.plan_suite(suite_path, tmp_path / "suite", [])
    assert bench.run_suite(manifest, False, []) == 0
    for seed in (3, 5):
        record = json.loads((manifest.parent / f"seed-{seed}/record.json").read_text())
        assert record["status"] == "completed"
        assert record["evaluation"]["configs"]["imagenet_full_finetune"]["seed"] == seed
        assert record["evaluation"]["metadata"]["imagenet_full_finetune"]["n_test"] == 6
        assert record["inputs"]["data_root"]["sha256"] == sha256_file(inventory)


def test_benchmark_windows_exclude_warmup_and_include_batch_work(
    monkeypatch, finetune_case, tmp_path
):
    bench_spec = importlib.util.spec_from_file_location(
        "bench_finetune",
        Path(__file__).resolve().parents[1] / "scripts/bench/bench_imagenet_finetune.py",
    )
    assert bench_spec is not None and bench_spec.loader is not None
    bench = importlib.util.module_from_spec(bench_spec)
    bench_spec.loader.exec_module(bench)
    model, checkpoint, spec = finetune_case
    config_path = tmp_path / "throughput.yaml"
    config_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "device": "cuda",
                "benchmarks": [spec],
            }
        )
    )
    benchmark, source, config, identity = bench.load_config(config_path)
    assert isinstance(benchmark, ImageNetFullFinetuneBenchmark)
    assert benchmark.split is None
    assert config["accumulation"] == spec["accumulation"]
    assert identity["sha256"] == ft.digest(checkpoint)
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)
    now = [0.0]

    class Loader:
        batch_size = 4
        dataset = range(40)

        def __len__(self):
            return 10

        def __iter__(self):
            now[0] += 50.0  # Worker startup, outside measured windows.
            for _ in range(10):
                now[0] += 1.0
                yield None

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: now[0])
    timed = bench.TimedLoader(Loader(), warmup=1, window=2, accumulation=2)
    for _ in timed:
        now[0] += 4.0  # Forward/backward/update work after the loader yields.
    assert timed.warmup_seconds == 60.0
    assert len(timed.windows) == 2
    for window in timed.windows:
        assert window == {
            "seconds": 20.0,
            "images": 16,
            "images_per_second": 0.8,
            "loader_next_seconds": 4.0,
        }
