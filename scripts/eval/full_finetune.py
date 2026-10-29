"""Fine-tune a checkpoint's image encoder on a fixed classification split."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.ops import StochasticDepth
from torchvision.transforms import v2

from egolearn.config.registry import VIEWS
from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD
from egolearn.evaluation.benchmarks.imagenet_linear.benchmark import build_split
from egolearn.evaluation.benchmarks.labeled_s_linear.probe import load_split_text, parse_split
from egolearn.evaluation.linear_probe import ClassificationSplit
from egolearn.evaluation.linear_sgd import _ImageDataset
from egolearn.model.build import build_model


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, classes: int, pooling: str, drop_path: float):
        super().__init__()
        self.encoder = encoder
        self.fc_norm = nn.Identity()
        if pooling == "mae_global":
            if not all(hasattr(encoder, name) for name in ("blocks", "patch_embed", "norm")):
                raise ValueError("mae_global pooling requires an MAE encoder")
            # Normalize after averaging raw patch tokens, as in MAE fine-tuning.
            encoder.norm = nn.Identity()
            encoder.pooling = "mean"
            self.fc_norm = nn.LayerNorm(encoder.out_dim, eps=1e-6)
            for index, block in enumerate(encoder.blocks):
                probability = drop_path * index / max(1, len(encoder.blocks) - 1)
                block.dropout1 = StochasticDepth(probability, "row")
                block.dropout2 = StochasticDepth(probability, "row")
        elif pooling != "default":
            raise ValueError(f"unknown pooling {pooling}")
        self.head = nn.Linear(encoder.out_dim, classes)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)
        self.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if not isinstance(features, torch.Tensor):
            features = features.pooled
        return self.head(self.fc_norm(features))


def load_classifier(config: dict, classes: int) -> tuple[Classifier, dict]:
    path = config["checkpoint"]
    checkpoint_hash = digest(path)
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    construction = state["model_config"]
    model = build_model(
        construction["model"],
        imports=construction.get("_imports_", []),
        initialize_pretrained=False,
    )
    if state.get("export_contract") != model.export_contract():
        raise ValueError("checkpoint model contract mismatch")
    model.load_state_dict(state["model"], strict=True)
    encoder = model.encoders[config.get("encoder", "vision")]
    classifier = Classifier(encoder, classes, config["pooling"], config["drop_path"])
    return classifier, {
        "checkpoint": path,
        "checkpoint_sha256": checkpoint_hash,
        "pretraining_step": state.get("step"),
        "construction": construction,
        "trainable_parameters": sum(p.numel() for p in classifier.parameters()),
    }


def parameter_groups(model: Classifier, weight_decay: float, layer_decay: float) -> list[dict]:
    depth = len(model.encoder.blocks) if hasattr(model.encoder, "blocks") else 0
    groups: dict[tuple[int, bool], dict] = {}
    for name, parameter in model.named_parameters():
        layer = depth + 1
        if depth and name.startswith("encoder."):
            layer = 0
            if name.startswith("encoder.blocks."):
                layer = int(name.split(".")[2]) + 1
        decay = parameter.ndim > 1 and not name.endswith(("cls_token", "pos_embed"))
        group = groups.setdefault(
            (layer, decay),
            {
                "params": [],
                "weight_decay": weight_decay if decay else 0.0,
                "lr_scale": layer_decay ** (depth + 1 - layer),
            },
        )
        group["params"].append(parameter)
    return list(groups.values())


def load_split(config: dict) -> ClassificationSplit:
    dataset = config["dataset"]
    if dataset["kind"] == "imagenet":
        split = build_split(dataset["root"], expected_classes=dataset["classes"])
    else:
        if digest(dataset["manifest"]) != dataset["manifest_sha256"]:
            raise ValueError("split manifest SHA-256 changed")
        if dataset["kind"] == "labeled_s":
            text, _, _ = load_split_text(dataset["manifest"])
            split = parse_split(text, dataset["root"])
        elif dataset["kind"] == "manifest":
            split = ClassificationSplit.from_mapping(
                json.loads(Path(dataset["manifest"]).read_text())
            )
        else:
            raise ValueError("unknown dataset kind")
    if (len(split.train), len(split.test), len(split.classes)) != (
        dataset["train_size"],
        dataset["test_size"],
        dataset["classes"],
    ):
        raise ValueError("dataset counts differ from the pinned protocol")
    train_paths = {item.image for item in split.train}
    if train_paths.intersection(item.image for item in split.test):
        raise ValueError("training and held-out images overlap")
    return split


def split_identity(split: ClassificationSplit) -> dict:
    sha = hashlib.sha256(json.dumps(split.classes).encode())
    for name in ("train", "test"):
        sha.update(name.encode())
        for item in getattr(split, name):
            sha.update(json.dumps([item.image, item.label]).encode() + b"\n")
    return {
        "classes": list(split.classes),
        "membership_sha256": sha.hexdigest(),
        "n_train": len(split.train),
        "n_test": len(split.test),
    }


def train_transform(config: dict):
    transforms = [
        T.RandomResizedCrop(
            config["size"],
            scale=(config["crop_min"], 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
    ]
    if config["randaugment"]:
        transforms.append(
            T.RandAugment(num_ops=2, magnitude=9, interpolation=T.InterpolationMode.BICUBIC)
        )
    transforms.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if config["erase_probability"]:
        transforms.append(T.RandomErasing(p=config["erase_probability"], value="random"))
    return T.Compose(transforms)


def learning_rate(config: dict, progress: float) -> float:
    warmup = config["warmup_epochs"]
    if progress < warmup:
        return config["lr"] * progress / warmup
    fraction = (progress - warmup) / (config["epochs"] - warmup)
    return config["lr"] * 0.5 * (1.0 + math.cos(math.pi * min(1.0, fraction)))


def make_loader(items, transform, config: dict, *, training: bool, epoch: int = 0):
    return DataLoader(
        _ImageDataset(items, transform),
        batch_size=config["batch_size"],
        shuffle=training,
        num_workers=config["workers"],
        pin_memory=config["device"].startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"] + epoch),
        persistent_workers=False,
        drop_last=False,
    )


def prepare_model(model: nn.Module, config: dict) -> nn.Module:
    """Place the model and enable the requested kernel options."""
    torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", False)
    model.to(config["device"])
    if config.get("channels_last", False):
        model.to(memory_format=torch.channels_last)
    mode = config.get("compile_mode")
    if mode is not None:
        # In-place compilation preserves checkpoint keys and encoder/head attributes.
        model.compile(mode=mode)
    return model


def move_images(images: torch.Tensor, config: dict) -> torch.Tensor:
    """Transfer images in the model's configured memory format."""
    memory_format = (
        torch.channels_last if config.get("channels_last", False) else torch.contiguous_format
    )
    return images.to(config["device"], non_blocking=True, memory_format=memory_format)


@torch.no_grad()
def accuracy(model, loader, config: dict) -> dict:
    model.eval()
    deferred = config.get("defer_metrics", False)
    correct = torch.zeros((), dtype=torch.int64, device=config["device"]) if deferred else 0
    top5 = torch.zeros_like(correct) if deferred else 0
    finite = torch.ones((), dtype=torch.bool, device=config["device"])
    total = 0
    for images, labels in loader:
        images = move_images(images, config)
        labels = labels.to(config["device"], non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["amp"]):
            logits = model(images).float()
        if deferred:
            finite &= torch.isfinite(logits).all()
        elif not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite validation logits")
        hits = (logits.argmax(1) == labels).sum()
        hits5 = (
            (logits.topk(min(5, logits.shape[1]), dim=1).indices == labels[:, None]).any(1).sum()
        )
        correct += hits if deferred else int(hits)
        top5 += hits5 if deferred else int(hits5)
        total += labels.numel()
    if deferred:
        if not finite:
            raise FloatingPointError("nonfinite validation logits")
        correct, top5 = int(correct), int(top5)
    return {"top1": correct / total, "top5": top5 / total, "n_test": total}


def train_epoch(model, loader, optimizer, config: dict, epoch: int, progress_callback=None) -> dict:
    model.train()
    mixing = []
    if config["mixup"]:
        mixing.append(v2.MixUp(alpha=config["mixup"], num_classes=model.head.out_features))
    if config["cutmix"]:
        mixing.append(v2.CutMix(alpha=config["cutmix"], num_classes=model.head.out_features))
    accum = config["accumulation"]
    deferred = config.get("defer_metrics", False)
    total = 0
    correct = torch.zeros((), dtype=torch.int64, device=config["device"]) if deferred else 0
    loss_sum = torch.zeros((), dtype=torch.float64, device=config["device"]) if deferred else 0.0
    finite_loss = torch.ones((), dtype=torch.bool, device=config["device"])
    first_parameter = next(model.encoder.parameters())
    before = first_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(loader):
        images = move_images(images, config)
        labels = labels.to(config["device"], non_blocking=True)
        targets = labels
        if mixing:
            images, targets = mixing[int(torch.randint(len(mixing), (1,)))](images, labels)
        lr = learning_rate(config, epoch + (step + 1) / len(loader))
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        # Weight a partial final accumulation group by its actual sample count.
        start = (step // accum) * accum * loader.batch_size
        group_size = min(accum * loader.batch_size, len(loader.dataset) - start)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config["amp"]):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), targets, label_smoothing=config["smoothing"])
        if deferred:
            finite_loss &= torch.isfinite(loss.detach())
        elif not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        (loss * labels.numel() / group_size).backward()
        if (step + 1) % accum == 0 or step + 1 == len(loader):
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config["clip_grad"], error_if_nonfinite=not deferred
            )
            if deferred:
                # One host transfer per update; reject invalid updates before optimizer.step.
                loss_ok, grad_ok, zero_grad = torch.stack(
                    [finite_loss, torch.isfinite(grad_norm), grad_norm == 0]
                ).tolist()
                if not loss_ok or not grad_ok:
                    raise FloatingPointError("nonfinite training loss or gradient")
                finite_loss.fill_(True)
            else:
                zero_grad = float(grad_norm) == 0.0
            if zero_grad:
                raise RuntimeError("all parameter gradients are zero")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += (loss.detach().double() if deferred else float(loss.detach())) * labels.numel()
        hits = (logits.detach().argmax(1) == labels).sum()
        correct += hits if deferred else int(hits)
        total += labels.numel()
        if progress_callback is not None and (step == 0 or (step + 1) % 100 == 0):
            progress_callback(step + 1, len(loader), float(loss_sum) / total)
    changed = not torch.equal(before, first_parameter.detach())
    if epoch == 0 and not changed:
        raise RuntimeError("first encoder parameter did not update during fine-tuning")
    return {
        "train_loss": float(loss_sum) / total,
        "train_top1_running": int(correct) / total,
        "encoder_parameter_changed": changed,
        "n_train": total,
        "lr": lr,
    }


def run(config: dict, *, max_epochs: int | None = None, max_seconds: float | None = None) -> int:
    if config["batch_size"] < 1 or config["accumulation"] < 1:
        raise ValueError("batch_size and accumulation must be positive")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("warmup must be shorter than training")
    if config["amp"] and not config["device"].startswith("cuda"):
        raise ValueError("BF16 AMP requires CUDA")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(config, out, max_epochs=max_epochs, max_seconds=max_seconds)


def run_locked(config: dict, out: Path, *, max_epochs, max_seconds) -> int:
    started = time.monotonic()
    torch.set_num_threads(config.get("threads", 1))
    seed_all(config["seed"])
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text()) != config:
            raise ValueError("output directory belongs to a different configuration")
    else:
        write_json(out / "config.json", config)
    if (out / "results.json").exists():
        print(f"Already complete: {out}", flush=True)
        return 0
    split = load_split(config)
    model, provenance = load_classifier(config, len(split.classes))
    prepare_model(model, config)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config["weight_decay"], config["layer_decay"]), lr=config["lr"]
    )
    import torchvision

    import egolearn

    provenance.update(
        dataset=split_identity(split),
        config_sha256=fingerprint,
        runner_sha256=digest(__file__),
        package_file=egolearn.__file__,
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name() if config["amp"] else None,
    )
    provenance_path = out / "provenance.json"
    start_epoch = 0
    history = []
    if (out / "latest.pth").exists():
        state = torch.load(out / "latest.pth", map_location="cpu", weights_only=True)
        if state["config_sha256"] != fingerprint:
            raise ValueError("resume configuration mismatch")
        previous = json.loads(provenance_path.read_text())
        for key in ("dataset", "runner_sha256", "torch", "torchvision", "package_file"):
            if previous[key] != provenance[key]:
                raise ValueError(f"resume provenance changed: {key}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        history = state["history"]
        del state
    else:
        write_json(provenance_path, provenance)
    eval_loader = make_loader(
        split.test, VIEWS.get(config["eval_view"])(size=config["size"]), config, training=False
    )
    final_epoch = (
        min(config["epochs"], start_epoch + max_epochs) if max_epochs else config["epochs"]
    )
    for epoch in range(start_epoch, final_epoch):
        seed_all(config["seed"] + epoch)
        loader = make_loader(
            split.train, train_transform(config), config, training=True, epoch=epoch
        )
        epoch_started = time.monotonic()

        def report(batch, batches, loss, epoch=epoch, epoch_started=epoch_started):
            value = {
                "state": "running",
                "epoch": epoch + 1,
                "epochs": config["epochs"],
                "batch": batch,
                "batches": batches,
                "loss": loss,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "epoch_seconds": time.monotonic() - epoch_started,
            }
            write_json(out / "status.json", value)
            print(json.dumps(value), flush=True)

        metrics = train_epoch(model, loader, optimizer, config, epoch, report)
        if (epoch + 1) % config["eval_every"] == 0 or epoch + 1 == config["epochs"]:
            metrics.update(accuracy(model, eval_loader, config))
        metrics.update(epoch=epoch + 1, seconds=time.monotonic() - epoch_started)
        history.append(metrics)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config_sha256": fingerprint,
            "history": history,
        }
        torch.save(state, out / "latest.pth.tmp")
        (out / "latest.pth.tmp").replace(out / "latest.pth")
        write_json(out / "history.json", {"epochs": history})
        print(json.dumps(metrics), flush=True)
        if max_seconds and time.monotonic() - started >= max_seconds:
            break
    complete = bool(history) and history[-1]["epoch"] == config["epochs"]
    status = {
        "state": "complete" if complete else "paused",
        "epoch": history[-1]["epoch"],
        "epochs": config["epochs"],
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(out / "status.json", status)
    if complete:
        write_json(
            out / "results.json",
            {
                "status": "complete",
                "metrics": history[-1],
                "provenance": provenance,
                "config": config,
            },
        )
    return 0 if complete else 75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    try:
        raise SystemExit(run(config, max_epochs=args.max_epochs, max_seconds=args.max_seconds))
    except Exception as error:
        out = Path(config["output_dir"])
        if out.is_dir():
            write_json(
                out / "failure.json",
                {"error": repr(error), "job_id": os.environ.get("SLURM_JOB_ID")},
            )
        raise


if __name__ == "__main__":
    main()
