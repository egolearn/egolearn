"""generative:openai_chat -- a client for OpenAI-compatible chat servers such as vLLM.

Send Ego4D-derived frames (EgoSchema, Ego4D NLQ) only to servers the lab runs
itself, such as a vLLM deployment of Qwen2.5-VL; the Ego4D license does not
allow handing its videos to third-party APIs.  The API key, when the server
needs one, comes from the environment variable named by ``api_key_env``.

Each request holds the request's images, then its videos, then its text, as
one user message (after an optional system message).  ``video_input`` decides
how a Video reaches the server:

* ``video``: one video part per Video, as vLLM's ``video/jpeg`` data URL (the
  frames as comma-separated base64 JPEGs), with the sampling rate passed as
  ``mm_processor_kwargs.fps`` so that models which encode absolute time from
  the frame rate (Qwen2.5-VL) see true times.  This form is vLLM-specific.
* ``frames``: every frame as an image part after a text part
  ``"Frame at <t> s:"``, for any server that accepts images.

Decoding is greedy by default.  With ``cache_dir`` set, replies are stored as
JSON files named by the SHA-256 of the full request body: the model name,
every message part (frames and images included as JPEG bytes), the sampling
parameters and the processor arguments.  Any change to the prompt, the frames,
their resolution, timestamps or frame rate, or the decoding settings therefore
misses the cache, while a rerun of the same evaluation reads it.  The key holds the
model name, not its weights: serving other weights under the same name needs another
``cache_dir``.  Empty replies are not cached.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from egolearn.config.registry import GENERATIVE
from egolearn.model.generative.base import GenerativeComponent, Video

if TYPE_CHECKING:
    from PIL import Image

VIDEO_INPUTS = ("video", "frames")


@GENERATIVE.register("openai_chat")
class OpenAIChat(GenerativeComponent):
    """Text replies from a chat-completions endpoint; see the module docstring."""

    input_modalities = ("text", "image", "video")
    output_modality = "text"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        video_input: str = "video",
        system_prompt: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        jpeg_quality: int = 90,
        cache_dir: str | None = None,
        concurrency: int = 8,
        timeout: float = 600.0,
        max_retries: int = 3,
    ):
        super().__init__()
        if video_input not in VIDEO_INPUTS:
            raise ValueError(f"video_input must be one of {VIDEO_INPUTS}, got {video_input!r}")
        if max_tokens < 1 or concurrency < 1 or max_retries < 0:
            raise ValueError("max_tokens and concurrency must be >= 1, max_retries >= 0")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
        api_key = "EMPTY"  # vLLM accepts any key unless started with --api-key
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env, "")
            if not api_key:
                raise ValueError(f"environment variable {api_key_env} (the API key) is not set")
        try:
            import openai
        except ImportError as error:
            raise ImportError(
                "generative:openai_chat needs the openai package: install egolearn[qa]"
            ) from error
        self.model = model
        self.video_input = video_input
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.jpeg_quality = jpeg_quality
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.concurrency = concurrency
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def generate(
        self,
        texts: Sequence[str] | None = None,
        images: Sequence[Sequence[Image.Image]] | None = None,
        videos: Sequence[Sequence[Video]] | None = None,
    ) -> list[str]:
        count = self.check_inputs(texts, images, videos)
        bodies = [
            self.request_body(
                None if texts is None else texts[i],
                () if images is None else images[i],
                () if videos is None else videos[i],
            )
            for i in range(count)
        ]
        if count == 1 or self.concurrency == 1:
            return [self._reply(body) for body in bodies]
        with ThreadPoolExecutor(min(self.concurrency, count)) as pool:
            return list(pool.map(self._reply, bodies))

    # ---- requests ----------------------------------------------------------------------------
    def request_body(
        self, text: str | None, images: Sequence[Image.Image], videos: Sequence[Video]
    ) -> dict[str, Any]:
        """The chat-completions request for one generation, as sent to the server."""
        content: list[dict[str, Any]] = [self._image_part(image) for image in images]
        for video in videos:
            if self.video_input == "video":
                frames = ",".join(self._jpeg_base64(frame) for frame in video.frames)
                content.append(
                    {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{frames}"}}
                )
            else:
                for frame, time in zip(video.frames, video.timestamps, strict=True):
                    content.append({"type": "text", "text": f"Frame at {time:.1f} s:"})
                    content.append(self._image_part(frame))
        if text is not None:
            content.append({"type": "text", "text": text})
        messages: list[dict[str, Any]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if videos and self.video_input == "video":
            rates = [float(video.fps) for video in videos]
            body["mm_processor_kwargs"] = {"fps": rates[0] if len(rates) == 1 else rates}
        return body

    def _jpeg_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _image_part(self, image: Image.Image) -> dict[str, Any]:
        url = f"data:image/jpeg;base64,{self._jpeg_base64(image)}"
        return {"type": "image_url", "image_url": {"url": url}}

    # ---- replies -----------------------------------------------------------------------------
    def _reply(self, body: dict[str, Any]) -> str:
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = None if self.cache_dir is None else self.cache_dir / key[:2] / f"{key}.json"
        if path is not None and path.is_file():
            return str(json.loads(path.read_text())["reply"])
        request = dict(body)
        extra = {}
        if "mm_processor_kwargs" in request:
            extra["mm_processor_kwargs"] = request.pop("mm_processor_kwargs")
        response = self._client.chat.completions.create(**request, extra_body=extra or None)
        choice = response.choices[0]
        reply = choice.message.content or ""
        if path is not None and reply:  # an empty reply is not kept, so a rerun asks again
            usage = response.usage.model_dump() if response.usage is not None else None
            entry = {
                "reply": reply,
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "usage": usage,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}-{os.urandom(4).hex()}")
            temporary.write_text(json.dumps(entry) + "\n")
            os.replace(temporary, path)
        return reply
