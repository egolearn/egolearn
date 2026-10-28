"""Reuse frozen-image probes with explicit static-clip V-JEPA evaluation."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn

from egolearn.config.registry import BENCHMARKS, PROBES
from egolearn.core import keys as K
from egolearn.evaluation.benchmark import Benchmark
from egolearn.evaluation.evaluator import EvaluationContext
from egolearn.model.encoders.base import EncoderOutput
from egolearn.model.model import ComposedModel


class StaticClipEncoder(nn.Module):
    """Expose a video encoder's pooled representation for repeated still images."""

    modality = "image"

    def __init__(self, encoder: nn.Module, num_frames: int = 16):
        super().__init__()
        if getattr(encoder, "modality", None) != "video":
            raise ValueError("static clips require a video encoder")
        tubelet = getattr(encoder, "tubelet", 1)
        if num_frames <= 0 or num_frames % tubelet:
            raise ValueError("num_frames must be positive and divisible by the encoder tubelet")
        self.encoder = encoder
        self.num_frames = num_frames
        self.out_dim = encoder.out_dim

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("static-clip inputs must have shape [B, 3, H, W]")
        clips = images.unsqueeze(1).expand(-1, self.num_frames, -1, -1, -1)
        return EncoderOutput.wrap(self.encoder(clips))


@BENCHMARKS.register("vjepa2_image")
class VJEPA2ImageBenchmark(Benchmark):
    """Apply an existing image probe to one explicitly selected video encoder.

    Only the input representation changes: dataset splits, preprocessing and
    scoring remain owned by the existing probe. The outer evaluator freezes
    the source model and restores its modes/gradient flags afterward.
    """

    version = "1"
    requires_capabilities = (K.CAP_VIDEO_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        probe: str,
        probe_kwargs: dict[str, Any],
        encoder: str = "video_target",
        num_frames: int = 16,
    ):
        if probe not in {"labeled_s_linear", "linear", "shape_bias"}:
            raise ValueError("supported image probes are labeled_s_linear, linear and shape_bias")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if probe_kwargs.get("head") is not None or probe_kwargs.get("encoder", "image") != "image":
            raise ValueError("static-clip probes use pooled image features without a head")
        self.name = name
        self.probe_name = probe
        self.encoder = encoder
        self.num_frames = num_frames
        self.probe_kwargs = dict(probe_kwargs)
        self.probe_kwargs["name"] = name
        self._probe_class = PROBES.get(probe)
        self._probe: Benchmark | None = None
        self._encoder_frozen: bool | None = None

    def configuration(self) -> dict[str, Any]:
        arguments = inspect.signature(self._probe_class).bind(**self.probe_kwargs)
        arguments.apply_defaults()
        kwargs = dict(arguments.arguments)
        for key in ("data_root", "split_file", "manifest", "output_json"):
            if kwargs.get(key) is not None:
                kwargs[key] = Path(kwargs[key]).name
        return {
            "name": self.name,
            "probe": self.probe_name,
            "probe_kwargs": kwargs,
            "encoder": self.encoder,
            "num_frames": self.num_frames,
            "representation": "static_clip_pooled",
            "probe_version": str(self._probe_class.version),
        }

    def prepare(self) -> None:
        if self._probe is None:
            self._probe = self._probe_class(**self.probe_kwargs)
        self._probe.prepare()

    def metadata(self) -> dict[str, Any]:
        assert self._probe is not None
        return {**self._probe.metadata(), "encoder_frozen": self._encoder_frozen}

    def run(self, model: ComposedModel, ctx: EvaluationContext) -> dict[str, float]:
        if self.encoder not in model.encoders:
            raise ValueError(f"checkpoint has no encoder named {self.encoder!r}")
        adapter = StaticClipEncoder(model.encoders[self.encoder], self.num_frames)
        self._encoder_frozen = self.encoder in model.frozen_names()
        self.prepare()
        image_model = ComposedModel({"image": adapter})
        # Do not call train()/eval() on the shared encoder: the outer evaluator
        # already sets its mode and owns restoring the original state.
        image_model.training = adapter.training = False
        assert self._probe is not None
        return self._probe.run(image_model, ctx)
