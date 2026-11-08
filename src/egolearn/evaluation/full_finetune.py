"""Fine-tune a checkpoint's image encoder on a fixed classification split."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.ops import StochasticDepth
from torchvision.transforms import v2

from egolearn.config.registry import VIEWS
from egolearn.core.context import _stable_seed
from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD
from egolearn.evaluation.evaluator import _isolated_evaluation_rng
from egolearn.evaluation.image_data import ImageDataset
from egolearn.evaluation.linear_probe import ClassificationSplit
from egolearn.model.model import ComposedModel


def validate_config(config: dict) -> None:
    """Reject invalid fine-tuning settings before enumerating data."""
    for key in ("size", "batch_size", "accumulation", "epochs", "eval_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["workers"]) is not int or config["workers"] < 0:
        raise ValueError("workers must be a nonnegative integer")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("warmup must be shorter than training")
    for key in ("lr", "clip_grad", "layer_decay", "crop_min"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "mixup", "cutmix", "erase_probability", "smoothing", "drop_path"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("crop_min", "layer_decay", "erase_probability", "smoothing"):
        if config[key] > 1:
            raise ValueError(f"{key} must not exceed one")
    if config["drop_path"] >= 1:
        raise ValueError("drop_path must be less than one")
    if config["pooling"] not in ("default", "mae_global"):
        raise ValueError("pooling must be default or mae_global")
    if config["drop_path"] and config["pooling"] != "mae_global":
        raise ValueError("drop_path requires mae_global pooling")
    if (
        config.get("device") is not None
        and config["amp"]
        and not config["device"].startswith("cuda")
    ):
        raise ValueError("BF16 AMP requires CUDA")


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def epoch_seed(seed: int, epoch: int) -> int:
    """Derive independent epoch streams for each fine-tuning run seed."""
    return _stable_seed(seed, f"evaluation.full_finetune:epoch:{epoch}")


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
        out_dim = int(cast(Any, encoder).out_dim)
        self.fc_norm: nn.Module = nn.Identity()
        if pooling == "mae_global":
            if not all(hasattr(encoder, name) for name in ("blocks", "patch_embed", "norm")):
                raise ValueError("mae_global pooling requires an MAE encoder")
            # Normalize after averaging raw patch tokens, as in MAE fine-tuning.
            mae_encoder = cast(Any, encoder)
            mae_encoder.norm = nn.Identity()
            mae_encoder.pooling = "mean"
            self.fc_norm = nn.LayerNorm(out_dim, eps=1e-6)
            blocks = mae_encoder.blocks
            for index, block in enumerate(blocks):
                probability = drop_path * index / max(1, len(blocks) - 1)
                block.dropout1 = StochasticDepth(probability, "row")
                block.dropout2 = StochasticDepth(probability, "row")
        elif pooling != "default":
            raise ValueError(f"unknown pooling {pooling}")
        self.head = nn.Linear(out_dim, classes)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)
        self.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if not isinstance(features, torch.Tensor):
            features = features.pooled
        return self.head(self.fc_norm(features))


def select_encoder(model: ComposedModel, selector: str) -> nn.Module:
    """Resolve an encoder name or modality using the model's normal routing."""
    encoder = (
        model.encoders[selector] if selector in model.encoders else model.encoder_for(selector)
    )
    if encoder is None:
        raise KeyError(f"model has no encoder {selector!r}")
    if encoder.modality != "image":
        raise ValueError("full fine-tuning requires an image encoder")
    return encoder


def make_classifier(model: ComposedModel, classes: int, config: dict) -> Classifier:
    """Copy the selected encoder so adaptation cannot change the evaluated model."""
    encoder = deepcopy(select_encoder(model, config["encoder"]))
    return Classifier(encoder, classes, config["pooling"], config["drop_path"])


def encoder_identity(model: ComposedModel, encoder: str) -> dict:
    """Identify starting weights and construction before fine-tuning."""
    sha = hashlib.sha256()
    module = select_encoder(model, encoder)
    for name, tensor in module.state_dict().items():
        sha.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        sha.update(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return {
        "encoder_sha256": sha.hexdigest(),
        "construction": model.construction_config,
        "encoder_type": f"{type(module).__module__}.{type(module).__qualname__}",
    }


def parameter_groups(model: Classifier, weight_decay: float, layer_decay: float) -> list[dict]:
    depth = len(getattr(model.encoder, "blocks", ()))
    groups: dict[tuple[int, bool], dict] = {}
    for name, parameter in model.named_parameters():
        layer = depth + 1
        if depth and name.startswith("encoder."):
            if name.startswith("encoder.blocks."):
                layer = int(name.split(".")[2]) + 1
            elif name.startswith("encoder.patch_embed.") or name in {
                "encoder.cls_token",
                "encoder.pos_embed",
            }:
                layer = 0
            # Final normalization and the classifier keep the full learning rate.
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
        ImageDataset(items, transform),
        batch_size=config["batch_size"],
        shuffle=training,
        num_workers=config["workers"],
        pin_memory=config["device"].startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(epoch_seed(config["seed"], epoch)),
        persistent_workers=False,
        drop_last=False,
    )


def prepare_model(model: nn.Module, config: dict) -> nn.Module:
    """Place the model and enable the requested kernel options."""
    torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", False)
    model.to(config["device"])
    if config.get("channels_last", False):
        # PyTorch accepts memory_format here, but its Module.to overloads omit it.
        model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
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
    top5 = torch.zeros((), dtype=torch.int64, device=config["device"]) if deferred else 0
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


class FineTunePaused(RuntimeError):
    """Signal an incomplete run whose last completed epoch can be resumed."""


def fit(
    model: ComposedModel,
    split: ClassificationSplit,
    config: dict,
    *,
    output_dir: str | Path | None = None,
    max_epochs: int | None = None,
    max_seconds: float | None = None,
) -> dict[str, float]:
    """Fine-tune a private encoder copy and return final-epoch classification scores.

    An output directory enables epoch-boundary resume. Resume requires identical
    starting weights, construction, configuration, split membership and runtime.
    Split hashes identify paths and labels, not image contents. Execution budgets
    raise FineTunePaused after saving an epoch rather than returning partial scores.
    """
    validate_config(config)
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    previous_autotuning = torch.backends.cudnn.benchmark
    try:
        with (
            (out / ".lock").open("a") if out is not None else nullcontext() as lock,
            _isolated_evaluation_rng(config["seed"]),
            torch.inference_mode(False),
            torch.enable_grad(),
        ):
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _fit_locked(model, split, config, out, max_epochs, max_seconds)
    finally:
        torch.backends.cudnn.benchmark = previous_autotuning


def _fit_locked(source, split, config, out, max_epochs, max_seconds) -> dict[str, float]:
    import torchvision

    from egolearn.evaluation import image_data

    started = time.monotonic()
    provenance = {
        **encoder_identity(source, config["encoder"]),
        "dataset": split.metadata(),
        "config": config,
        "engine_sha256": digest(__file__),
        "image_data_sha256": digest(image_data.__file__),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
    }
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    if out is not None:
        provenance_path = out / "provenance.json"
        if provenance_path.exists():
            if json.loads(provenance_path.read_text()) != provenance:
                raise ValueError("fine-tuning resume provenance changed")
        else:
            if any((out / name).exists() for name in ("latest.pth", "results.json")):
                raise ValueError("fine-tuning resume provenance is missing")
            write_json(provenance_path, provenance)
    seed_all(config["seed"])
    model = make_classifier(source, len(split.classes), config)
    prepare_model(model, config)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config["weight_decay"], config["layer_decay"]), lr=config["lr"]
    )
    start_epoch = 0
    history = []
    if out is not None and (out / "latest.pth").exists():
        state = torch.load(out / "latest.pth", map_location="cpu", weights_only=True)
        if state["provenance_sha256"] != fingerprint:
            raise ValueError("fine-tuning checkpoint provenance changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        history = state["history"]
        del state
    eval_loader = make_loader(
        split.test, VIEWS.get(config["eval_view"])(size=config["size"]), config, training=False
    )
    final_epoch = (
        min(config["epochs"], start_epoch + max_epochs) if max_epochs else config["epochs"]
    )
    for epoch in range(start_epoch, final_epoch):
        seed_all(epoch_seed(config["seed"], epoch))
        loader = make_loader(
            split.train, train_transform(config), config, training=True, epoch=epoch
        )
        epoch_started = time.monotonic()

        def report(batch, batches, loss, epoch=epoch, epoch_started=epoch_started):
            if out is not None:
                write_json(
                    out / "status.json",
                    {
                        "state": "running",
                        "epoch": epoch + 1,
                        "epochs": config["epochs"],
                        "batch": batch,
                        "batches": batches,
                        "loss": loss,
                        "epoch_seconds": time.monotonic() - epoch_started,
                    },
                )

        metrics = train_epoch(model, loader, optimizer, config, epoch, report)
        if (epoch + 1) % config["eval_every"] == 0 or epoch + 1 == config["epochs"]:
            metrics.update(accuracy(model, eval_loader, config))
        metrics.update(epoch=epoch + 1)
        history.append(metrics)
        if out is not None:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "provenance_sha256": fingerprint,
                "history": history,
            }
            torch.save(state, out / "latest.pth.tmp")
            (out / "latest.pth.tmp").replace(out / "latest.pth")
            write_json(out / "history.json", {"epochs": history})
        if max_seconds and time.monotonic() - started >= max_seconds:
            break
    complete = bool(history) and history[-1]["epoch"] == config["epochs"]
    if out is not None:
        write_json(
            out / "status.json",
            {
                "state": "complete" if complete else "paused",
                "epoch": history[-1]["epoch"],
                "epochs": config["epochs"],
            },
        )
    if not complete:
        raise FineTunePaused("fine-tuning paused at an epoch boundary; rerun to resume")
    metrics = {key: float(value) for key, value in history[-1].items()}
    if out is not None:
        write_json(out / "results.json", {"metrics": metrics, "provenance_sha256": fingerprint})
    return metrics
