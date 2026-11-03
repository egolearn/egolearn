"""Generative components: the contract, the composed model, openai_chat and checkpoint-free eval."""

import base64
import io
import json

import pytest
import torch.nn as nn
from PIL import Image

from egolearn.core import keys as K
from egolearn.evaluation import Benchmark, evaluate_checkpoint
from egolearn.model import ComposedModel, build_model
from egolearn.model.generative import GenerativeComponent, Video


class Echo(GenerativeComponent):
    input_modalities = ("text", "video")
    output_modality = "text"

    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, texts=None, images=None, videos=None):
        count = self.check_inputs(texts, images, videos)
        self.calls.append((texts, videos))
        return [f"echo {i}" for i in range(count)]


def _frames(count, size=(8, 6)):
    return tuple(Image.new("RGB", size, (10 * i, 0, 0)) for i in range(count))


def test_generative_only_model_has_text_generation_and_routes_by_name_or_output():
    echo = Echo()
    model = ComposedModel({}, generative={"vlm": echo})
    assert model.capabilities() == frozenset({K.CAP_TEXT_GENERATION})
    assert model.generate("vlm", texts=["a", "b"]) == ["echo 0", "echo 1"]
    assert model.generate("text", texts=["a"]) == ["echo 0"]
    with pytest.raises(KeyError, match="no generative component"):
        model.generate("image", texts=["a"])
    assert model.export_contract()["generative"] == {
        "vlm": {"inputs": ["text", "video"], "output": "text"}
    }


def test_encoder_only_contract_is_unchanged():
    model = build_model({"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 8}}})
    assert "generative" not in model.export_contract()
    assert K.CAP_TEXT_GENERATION not in model.capabilities()


def test_composed_model_rejects_ambiguous_or_incomplete_generative_components():
    with pytest.raises(ValueError, match="at least one encoder or generative"):
        ComposedModel({})
    with pytest.raises(ValueError, match="both produce 'text'"):
        ComposedModel({}, generative={"a": Echo(), "b": Echo()})
    with pytest.raises(ValueError, match="lacks the 'input_modalities'"):
        ComposedModel({}, generative={"bad": nn.Identity()})

    class Painter(Echo):
        output_modality = "image"

    with pytest.raises(ValueError, match="only text generation"):
        ComposedModel({}, generative={"painter": Painter()})


def test_check_inputs_validates_modalities_and_alignment():
    echo = Echo()
    video = Video(frames=_frames(2), timestamps=(0.0, 1.0), fps=1.0)
    assert echo.check_inputs(["q"], None, [(video,)]) == 1
    with pytest.raises(ValueError, match="does not read image"):
        echo.check_inputs(["q"], [()], None)
    with pytest.raises(ValueError, match="disagree"):
        echo.check_inputs(["q", "r"], None, [(video,)])
    with pytest.raises(TypeError, match="one entry per request"):
        echo.check_inputs("q", None, None)
    with pytest.raises(ValueError, match="at least one input"):
        echo.check_inputs(None, None, None)


def test_video_validates_frames_and_times():
    with pytest.raises(ValueError, match="at least one frame"):
        Video(frames=(), timestamps=(), fps=1.0)
    with pytest.raises(ValueError, match="2 frames but 1"):
        Video(frames=_frames(2), timestamps=(0.0,), fps=1.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        Video(frames=_frames(2), timestamps=(1.0, 0.0), fps=1.0)
    with pytest.raises(ValueError, match="positive"):
        Video(frames=_frames(1), timestamps=(0.0,), fps=0.0)


class _NeedsText(Benchmark):
    name = "needs_text"
    requires_capabilities = (K.CAP_TEXT_GENERATION,)

    def run(self, model, ctx):
        return {"replies": float(len(model.generate("text", texts=["a", "b", "c"])))}


def _chat_config(url, **overrides):
    vlm = {"_target_": "generative:openai_chat", "base_url": url, "model": "test-vlm", **overrides}
    return {"model": {"generative": {"vlm": vlm}}}


def test_evaluation_without_checkpoint_builds_model_from_config(chat_server):
    pytest.importorskip("openai")
    result = evaluate_checkpoint(
        None,
        model_config=_chat_config(chat_server.url),
        benchmarks=[_NeedsText()],
        record_json=None,
    )
    assert result.results == {"needs_text": {"replies": 3.0}}
    assert result.checkpoint is None and result.model_config is not None
    assert result.model_config["model"]["generative"]["vlm"]["model"] == "test-vlm"
    assert len(chat_server.requests) == 3


def test_evaluation_needs_a_checkpoint_or_model_config():
    with pytest.raises(ValueError, match="needs a checkpoint, or a model_config"):
        evaluate_checkpoint(None, benchmarks=[_NeedsText()], record_json=None)


def test_benchmark_requiring_generation_rejects_an_encoder_model(tmp_path):
    from egolearn.evaluation import evaluate

    model = build_model({"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 8}}})
    with pytest.raises(ValueError, match="needs capabilities \\['text_generation'\\]"):
        evaluate(model, benchmarks=[_NeedsText()], record_json=None)


# ---- generative:openai_chat ----------------------------------------------------------------


def _chat(url, **kwargs):
    pytest.importorskip("openai")
    from egolearn.model.generative import OpenAIChat

    return OpenAIChat(base_url=url, model="test-vlm", **kwargs)


def _decode(data_url):
    header, _, payload = data_url.partition(",")
    assert header == "data:image/jpeg;base64"
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def test_video_input_sends_one_video_with_its_sampling_rate():
    chat = _chat("http://unused/v1", system_prompt="Be brief.")
    video = Video(frames=_frames(3), timestamps=(0.0, 2.0, 4.0), fps=0.5)
    body = chat.request_body("Which?", (), (video,))
    system, user = body["messages"]
    assert system == {"role": "system", "content": "Be brief."}
    video_part, text_part = user["content"]
    assert text_part == {"type": "text", "text": "Which?"}
    assert video_part["video_url"]["url"].startswith("data:video/jpeg;base64,")
    frames = video_part["video_url"]["url"].removeprefix("data:video/jpeg;base64,").split(",")
    assert len(frames) == 3
    assert body["mm_processor_kwargs"] == {"fps": 0.5}
    assert body["temperature"] == 0.0 and body["max_tokens"] == 128


def test_frames_input_labels_every_frame_with_its_time():
    chat = _chat("http://unused/v1", video_input="frames")
    video = Video(frames=_frames(2, size=(16, 12)), timestamps=(0.0, 7.25), fps=0.25)
    body = chat.request_body("Which?", (), (video,))
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url", "text"]
    assert parts[0]["text"] == "Frame at 0.0 s:" and parts[2]["text"] == "Frame at 7.2 s:"
    assert _decode(parts[1]["image_url"]["url"]).size == (16, 12)
    assert "mm_processor_kwargs" not in body


def test_generate_returns_replies_in_order_and_caches_by_request(chat_server, tmp_path):
    chat_server.reply = lambda body: body["messages"][-1]["content"][-1]["text"].upper()
    chat = _chat(chat_server.url, cache_dir=str(tmp_path / "cache"), concurrency=4)
    texts = [f"q{i}" for i in range(6)]
    assert chat.generate(texts=texts) == [t.upper() for t in texts]
    assert len(chat_server.requests) == 6
    assert chat.generate(texts=texts) == [t.upper() for t in texts]
    assert len(chat_server.requests) == 6  # all from the cache
    entries = list((tmp_path / "cache").rglob("*.json"))
    assert len(entries) == 6
    assert json.loads(entries[0].read_text())["finish_reason"] == "stop"

    video = Video(frames=_frames(2), timestamps=(0.0, 1.0), fps=1.0)
    chat.generate(texts=["q0"], videos=[(video,)])
    slower = Video(frames=video.frames, timestamps=(0.0, 2.0), fps=0.5)
    chat.generate(texts=["q0"], videos=[(slower,)])
    assert len(chat_server.requests) == 8  # a new frame rate is a new request
    assert chat_server.requests[-1]["body"]["mm_processor_kwargs"] == {"fps": 0.5}
    resampled = Video(frames=video.frames, timestamps=video.timestamps, fps=1.0, sampling={"x": 1})
    chat.generate(texts=["q0"], videos=[(resampled,)])
    assert len(chat_server.requests) == 9  # other sampling settings, another cache entry


def test_empty_replies_are_not_cached(chat_server, tmp_path):
    chat_server.reply = lambda body: ""
    chat = _chat(chat_server.url, cache_dir=str(tmp_path / "cache"))
    assert chat.generate(texts=["q"]) == [""] and chat.generate(texts=["q"]) == [""]
    assert len(chat_server.requests) == 2


def test_api_key_comes_from_the_named_environment_variable(chat_server, monkeypatch):
    with pytest.raises(ValueError, match="EGOLEARN_TEST_KEY"):
        _chat(chat_server.url, api_key_env="EGOLEARN_TEST_KEY")
    monkeypatch.setenv("EGOLEARN_TEST_KEY", "secret-123")
    _chat(chat_server.url, api_key_env="EGOLEARN_TEST_KEY").generate(texts=["q"])
    headers = {k.lower(): v for k, v in chat_server.requests[-1]["headers"].items()}
    assert headers["authorization"] == "Bearer secret-123"


def test_openai_chat_rejects_bad_settings():
    with pytest.raises(ValueError, match="video_input"):
        _chat("http://unused/v1", video_input="clip")
    with pytest.raises(ValueError, match="jpeg_quality"):
        _chat("http://unused/v1", jpeg_quality=0)
