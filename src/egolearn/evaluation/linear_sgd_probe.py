"""Stochastic-view SGD linear decoding on a frozen image encoder.

The encoder is frozen, but features are recomputed from a stochastic image
view every epoch while a randomly initialized linear classifier is trained
with SGD, LARC, cross-entropy, and a cosine learning-rate schedule.  It is a
separate probe from :mod:`egolearn.evaluation.linear_probe`, whose fixed
features and full-batch L-BFGS solve answer a different experimental question.

Hyperparameters: 10 epochs, 112-pixel crops, batch size 1,024,
L2-normalized features, SGD at learning rate 3, momentum 0.9,
no weight decay, and unclipped LARC with trust coefficient 0.001.

Evaluation uses a registered ``view`` (default: ``imagenet_center_crop``).
Match both ``view`` and ``size`` when comparing evaluation preprocessing
with the fixed-feature probe; the classifier training protocols still differ.
``train_top1_running`` reports the final epoch's accuracy accumulated from
pre-update predictions on stochastic training views, rather than post-fit
accuracy on deterministic training views.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler

from egolearn.config.registry import PROBES, VIEWS
from egolearn.core import keys as K
from egolearn.data.views import IMAGENET_MEAN, IMAGENET_STD
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage
from egolearn.evaluation.probe import Probe


class _ImageDataset(Dataset):
    def __init__(self, items: Sequence[LabeledImage], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int | tuple[int, int]) -> tuple[torch.Tensor, int]:
        seed = None
        if isinstance(index, tuple):
            index, seed = index
        item = self.items[index]
        with Image.open(item.image) as image:
            converted = image.convert("RGB")
            if seed is None:
                tensor = self.transform(converted)
            else:
                with _isolated_rng(seed):
                    tensor = self.transform(converted)
        return tensor, item.label


class _SeededSampler(Sampler[tuple[int, int]]):
    """Attach an epoch-and-item-derived augmentation seed to each index."""

    def __init__(
        self,
        sampler: Sampler[int],
        derive_seed: Callable[[str], int],
        namespace: str,
        size: int,
    ):
        self.sampler = sampler
        self.derive_seed = derive_seed
        self.namespace = namespace
        self.size = size
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        epoch = self.epoch
        self.epoch += 1
        for index in self.sampler:
            seed = self.derive_seed(f"{self.namespace}:epoch:{epoch}:item:{index}")
            yield index, seed

    def __len__(self) -> int:
        return self.size


class _L2Normalize(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value, p=2, dim=1)


# Adapted from NVIDIA Apex's LARC implementation (BSD-3-Clause):
# https://github.com/NVIDIA/apex/blob/e13873debc4699d39c6861074b9a3b2a02327f92/apex/parallel/LARC.py
# The following upstream license applies to the _LARC implementation below:
# https://github.com/NVIDIA/apex/blob/e13873debc4699d39c6861074b9a3b2a02327f92/LICENSE
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
class _LARC:
    """Minimal Apex-compatible LARC wrapper used by the reference protocol."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        self.optimizer = optimizer
        self.trust_coefficient = float(trust_coefficient)
        self.eps = float(eps)

    @torch.no_grad()
    def step(self) -> None:
        weight_decays: list[float] = []
        try:
            for group in self.optimizer.param_groups:
                weight_decay = float(group.get("weight_decay", 0.0))
                weight_decays.append(weight_decay)
                group["weight_decay"] = 0.0
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    parameter_norm = torch.linalg.vector_norm(parameter)
                    gradient_norm = torch.linalg.vector_norm(parameter.grad)
                    if parameter_norm.item() == 0.0 or gradient_norm.item() == 0.0:
                        continue
                    local_lr = (
                        self.trust_coefficient
                        * parameter_norm
                        / (gradient_norm + weight_decay * parameter_norm + self.eps)
                    )
                    if weight_decay:
                        parameter.grad.add_(parameter, alpha=weight_decay)
                    parameter.grad.mul_(local_lr)
            self.optimizer.step()
        finally:
            for group, weight_decay in zip(self.optimizer.param_groups, weight_decays, strict=True):
                group["weight_decay"] = weight_decay


def _train_transform(size: int):
    return T.Compose(
        [
            T.RandomResizedCrop(
                size,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _cosine_lr(base_lr: float, epoch: int, epochs: int) -> float:
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))


@contextmanager
def _isolated_rng(seed: int):
    """Make stochastic decoding repeatable without advancing the training RNG."""
    python_state = random.getstate()
    cpu_state = torch.Generator().manual_seed(int(seed)).get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(cpu_state)
            random.seed(int(seed))
            yield
    finally:
        random.setstate(python_state)


@PROBES.register("linear_sgd")
class LinearSGDProbe(Probe):
    """Train a linear classifier by SGD while keeping the encoder frozen."""

    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        manifest: str | None = None,
        encoder: str = "image",
        head: str | None = None,
        size: int = 112,
        batch_size: int = 1024,
        workers: int = 16,
        epochs: int = 10,
        lr: float = 3.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        larc_trust_coefficient: float | None = 0.001,
        l2_normalize: bool = True,
        seed: int = 0,
        split: ClassificationSplit | None = None,
        view: str = "imagenet_center_crop",
    ):
        self.name = name
        if (manifest is None) == (split is None):
            raise ValueError("linear SGD probe needs exactly one of manifest or split")
        if split is None:
            import json

            assert manifest is not None
            with open(manifest) as file:
                split = ClassificationSplit.from_mapping(json.load(file))
        if size <= 0 or batch_size <= 0 or epochs <= 0:
            raise ValueError("linear SGD size, batch_size, and epochs must be positive")
        if workers < 0:
            raise ValueError("linear SGD workers must be >= 0")
        if lr < 0.0 or weight_decay < 0.0:
            raise ValueError("linear SGD lr and weight_decay must be >= 0")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("linear SGD momentum must be in [0, 1)")
        if larc_trust_coefficient is not None and larc_trust_coefficient <= 0.0:
            raise ValueError("linear SGD LARC trust coefficient must be positive")

        self.split = split
        self.classes = list(split.classes)
        self.encoder = encoder
        self.head = head
        self.size = int(size)
        self.transform = VIEWS.get(view)(size=self.size)
        self.batch_size = int(batch_size)
        self.workers = int(workers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.larc_trust_coefficient = larc_trust_coefficient
        self.l2_normalize = bool(l2_normalize)
        self.seed = int(seed)

    def _loader(
        self,
        items,
        transform,
        shuffle: bool,
        generator,
        derive_seed: Callable[[str], int],
        split_name: str,
        device: torch.device,
    ) -> DataLoader:
        dataset = _ImageDataset(items, transform)
        base_sampler: Sampler[int]
        if shuffle:
            base_sampler = RandomSampler(dataset, generator=generator)
        else:
            base_sampler = SequentialSampler(dataset)
        namespace = f"evaluation.linear_sgd:{self.seed}:{split_name}"
        sampler = _SeededSampler(
            base_sampler,
            derive_seed,
            namespace,
            len(dataset),
        )
        # DataLoader consumes its generator to seed workers.  Keep that bookkeeping
        # separate from the generator that defines the shuffled sample order.
        worker_generator = torch.Generator().manual_seed(derive_seed(f"{namespace}:workers"))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.workers,
            pin_memory=device.type == "cuda",
            generator=worker_generator,
            multiprocessing_context="spawn" if self.workers > 0 else None,
        )

    def _features(self, model, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = model.encode(self.encoder, images).pooled
            if self.head is not None:
                features = model.apply_head(self.head, features)
        # Keep the trainable decoder fp32 even when the frozen encoder uses reduced precision.
        return features.detach().float()

    def _feature_dim(self, model) -> int:
        if self.head is None:
            encoder = (
                model.encoders[self.encoder]
                if self.encoder in model.encoders
                else model.encoder_for(self.encoder)
            )
            if encoder is None:
                raise KeyError(f"model has no encoder {self.encoder!r}")
            return int(encoder.out_dim)

        linears = [
            module for module in model.heads[self.head].modules() if isinstance(module, nn.Linear)
        ]
        if not linears:
            raise ValueError(
                f"cannot infer output width of head {self.head!r}; linear SGD supports "
                "heads containing at least one nn.Linear"
            )
        return int(linears[-1].out_features)

    def _accuracy(self, loader: DataLoader, model, decoder: nn.Module, device) -> float:
        correct = total = 0
        decoder.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = decoder(self._features(model, images))
                correct += int((logits.argmax(dim=1) == labels).sum())
                total += labels.numel()
        return correct / total

    def run(self, model, ctx) -> dict[str, float]:
        was_training = model.training
        model.eval()
        loader_generator = torch.Generator().manual_seed(self.seed)

        try:
            with _isolated_rng(self.seed):
                train_loader = self._loader(
                    self.split.train,
                    _train_transform(self.size),
                    True,
                    loader_generator,
                    ctx.derive_seed,
                    "train",
                    device=ctx.device,
                )
                eval_loader = self._loader(
                    self.split.test,
                    self.transform,
                    False,
                    loader_generator,
                    ctx.derive_seed,
                    "test",
                    device=ctx.device,
                )

                layers: list[nn.Module] = []
                if self.l2_normalize:
                    layers.append(_L2Normalize())
                layers.append(nn.Linear(self._feature_dim(model), len(self.classes)))
                decoder = nn.Sequential(*layers).to(ctx.device)
                optimizer = torch.optim.SGD(
                    decoder.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
                larc = (
                    _LARC(optimizer, self.larc_trust_coefficient)
                    if self.larc_trust_coefficient is not None
                    else None
                )

                train_top1_running = 0.0
                top1 = 0.0
                for epoch in range(self.epochs):
                    epoch_lr = _cosine_lr(self.lr, epoch, self.epochs)
                    for group in optimizer.param_groups:
                        group["lr"] = epoch_lr

                    decoder.train()
                    correct = total = 0
                    for images, labels in train_loader:
                        images = images.to(ctx.device, non_blocking=True)
                        labels = labels.to(ctx.device, non_blocking=True)
                        features = self._features(model, images)
                        with torch.enable_grad():
                            logits = decoder(features)
                            loss = F.cross_entropy(logits, labels)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            if larc is None:
                                optimizer.step()
                            else:
                                larc.step()
                        correct += int((logits.detach().argmax(dim=1) == labels).sum())
                        total += labels.numel()
                    train_top1_running = correct / total
                    top1 = self._accuracy(eval_loader, model, decoder, ctx.device)
                    print(
                        f"[linear-sgd {self.name}] epoch={epoch + 1}/{self.epochs} "
                        f"lr={epoch_lr:.6g} train_top1_running={train_top1_running:.6f} "
                        f"top1={top1:.6f}",
                        flush=True,
                    )
        finally:
            model.train(was_training)

        return {"top1": top1, "train_top1_running": train_top1_running}
