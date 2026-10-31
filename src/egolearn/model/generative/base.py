"""What every generative component promises the rest of the framework.

A generative component produces one output modality from inputs of one or
more modalities.  As with encoders, the framework asks questions answered by
class attributes rather than by inheritance:

  input_modalities  which inputs it reads: any of "text", "image", "video"
  output_modality   what it produces; the composed model gains
                    "<output_modality>_generation" for carrying it

and calls one method, batched over requests:

    generate(texts=None, images=None, videos=None) -> list

Element i of every given list belongs to request i, and the lists must have
the same length.  ``texts[i]`` is the request's text, ``images[i]`` a
sequence of PIL images and ``videos[i]`` a sequence of Video, which the
component places before the text.  A text-producing component returns one
string per request.

Video arrives as frames already sampled by the caller, with their media
times, so the benchmark and its configuration decide what a model sees and
every component sees the same thing.  How the frames reach the model (one
video input with its frame rate, or images labelled with their times) is the
component's configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class Video:
    """Frames sampled from one video, in time order.

    ``timestamps`` are the frames' media times in seconds from the start of the
    video (or clip) the caller read; ``fps`` is the rate they were sampled at,
    which is what a model that encodes time from the frame rate must be told.
    """

    frames: tuple[Image.Image, ...]
    timestamps: tuple[float, ...]
    fps: float

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("a Video needs at least one frame")
        if len(self.timestamps) != len(self.frames):
            raise ValueError(f"{len(self.frames)} frames but {len(self.timestamps)} timestamps")
        if any(b < a for a, b in zip(self.timestamps, self.timestamps[1:], strict=False)):
            raise ValueError("Video timestamps must be non-decreasing")
        if not self.fps > 0:
            raise ValueError(f"Video fps must be positive, got {self.fps}")


class GenerativeComponent(nn.Module):
    """Base class that checks the inputs of ``generate`` against the declared modalities.

    Components must be nn.Modules, since the composed model holds them in a ModuleDict;
    subclassing this one is optional.
    """

    input_modalities: tuple[str, ...] = ("text",)
    output_modality: str = "text"

    def generate(
        self,
        texts: Sequence[str] | None = None,
        images: Sequence[Sequence[Image.Image]] | None = None,
        videos: Sequence[Sequence[Video]] | None = None,
    ) -> list[Any]:
        raise NotImplementedError

    def check_inputs(
        self,
        texts: Sequence[str] | None,
        images: Sequence[Sequence[Image.Image]] | None,
        videos: Sequence[Sequence[Video]] | None,
    ) -> int:
        """The batch size, after checking that every given input is declared and aligned."""
        given = {"text": texts, "image": images, "video": videos}
        sizes = set()
        for modality, value in given.items():
            if value is None:
                continue
            if modality not in self.input_modalities:
                raise ValueError(
                    f"{type(self).__name__} does not read {modality} inputs "
                    f"(input_modalities={self.input_modalities})"
                )
            if isinstance(value, str):
                raise TypeError(f"{modality} inputs must be a sequence with one entry per request")
            sizes.add(len(value))
        if not sizes:
            raise ValueError("generate needs at least one input")
        if len(sizes) > 1:
            raise ValueError(f"inputs disagree on the number of requests: {sorted(sizes)}")
        return sizes.pop()
