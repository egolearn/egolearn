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

from egolearn.model.build import build_model

spec = importlib.util.spec_from_file_location(
    "full_finetune", Path(__file__).resolve().parents[1] / "scripts/eval/full_finetune.py"
)
assert spec is not None and spec.loader is not None
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


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


def test_finetune_resume_matches_uninterrupted_and_rejects_changed_config(tmp_path):
    torch.set_num_threads(1)
    construction = {"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 8}}}
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
    config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ft.digest(checkpoint),
        "encoder": "vision",
        "pooling": "default",
        "drop_path": 0.0,
        "dataset": {
            "kind": "imagenet",
            "root": str(root),
            "classes": 2,
            "train_size": 6,
            "test_size": 6,
        },
        "device": "cpu",
        "amp": False,
        "seed": 0,
        "epochs": 2,
        "warmup_epochs": 0,
        "batch_size": 2,
        "accumulation": 2,
        "workers": 0,
        "lr": 0.001,
        "weight_decay": 0.01,
        "layer_decay": 1.0,
        "size": 16,
        "crop_min": 0.08,
        "randaugment": False,
        "erase_probability": 0.0,
        "mixup": 0.8,
        "cutmix": 1.0,
        "smoothing": 0.1,
        "clip_grad": 1.0,
        "eval_view": "imagenet_center_crop",
        "eval_every": 1,
        "defer_metrics": True,
        "channels_last": True,
        "output_dir": str(tmp_path / "uninterrupted"),
    }
    assert ft.run(config) == 0
    resumed_config = {**config, "output_dir": str(tmp_path / "resumed")}
    assert ft.run(resumed_config, max_epochs=1) == 75
    with pytest.raises(ValueError, match="different configuration"):
        ft.run({**resumed_config, "accumulation": 1})
    assert ft.run(resumed_config) == 0
    full = torch.load(tmp_path / "uninterrupted/latest.pth", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pth", weights_only=True)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert json.loads((tmp_path / "resumed/results.json").read_text())["status"] == "complete"


def test_benchmark_windows_exclude_warmup_and_include_batch_work(monkeypatch):
    bench_spec = importlib.util.spec_from_file_location(
        "bench_finetune",
        Path(__file__).resolve().parents[1] / "scripts/bench/bench_imagenet_finetune.py",
    )
    assert bench_spec is not None and bench_spec.loader is not None
    bench = importlib.util.module_from_spec(bench_spec)
    bench_spec.loader.exec_module(bench)
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
