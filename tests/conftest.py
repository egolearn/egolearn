"""Shared synthetic training configurations and video QA fixtures."""

import json
import threading
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest


@pytest.fixture
def toy_config():
    def build(tmp_path, dim: int = 64):
        return {
            "run_dir": str(tmp_path / "run"),
            "seed": 7,
            "device": "cpu",
            "stream": {
                "_target_": "stream:toy",
                "n_sessions": 1,
                "blocks_per_session": 4,
                "block_len": 40,
                "img_size": 64,
            },
            "views": {
                "v1": {"name": "simclr_strong", "size": 64, "view_idx": 0},
                "v2": {"name": "simclr_strong", "size": 64, "view_idx": 1},
            },
            "label_stores": ["fixed"],
            "admission": [
                {"_target_": "admission:frame_emitter", "subsample": 1},
                {
                    "_target_": "admission:fixed_window_labeler",
                    "class_length": 40,
                    "label_store": "fixed",
                },
            ],
            "memories": {
                "frames": {
                    "tiers": {
                        "short": {"capacity": 30, "update": "fifo"},
                        "long": {"capacity": 60, "update": "reservoir"},
                    },
                    "admit_to": ["short", "long"],
                    "overflow": {},
                }
            },
            "routes": {"RawFrame": "frames"},
            "recipe": {
                "slots": [
                    {
                        "name": "curr",
                        "source": {"kind": "tier", "memory": "frames", "tier": "short"},
                        "n": 4,
                        "unit": "frame",
                        "views": ["v1", "v2"],
                        "labels_from": "fixed",
                        "strict": True,
                    },
                ]
            },
            "model": {
                "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": dim}},
                "heads": {
                    "proj": {"encoder": "vision", "_target_": "head:mlp", "sizes": [dim, dim, 32]},
                    "cls": {"encoder": "vision", "_target_": "head:mlp", "sizes": [dim, dim, 8]},
                },
            },
            "losses": [
                {
                    "_target_": "loss:ntxent",
                    "name": "inst",
                    "slots": ["curr"],
                    "head": "proj",
                    "temperature": 0.1,
                    "weight": 1.0,
                },
                {
                    "_target_": "loss:tcl",
                    "name": "temp",
                    "slots": ["curr"],
                    "head": "cls",
                    "num_classes": 8,
                    "weight": 1.0,
                },
            ],
            "solver": {
                "opt": "sgd",
                "groups": [{"match": ".*", "lr": 0.01, "weight_decay": 0.0}],
                "schedule": {"constant": True},
            },
            "driver": {
                "advance": {"kind": "observations", "n": 8},
                "rate": {"steps_per_admit": 1, "resample": True},
                "warmup": {"admitted": 20},
                "max_steps": 2,
            },
            "hooks": {"log_every": 1},
        }

    return build


class ChatServer:
    """A stand-in OpenAI-compatible chat server on localhost.

    Every request is recorded (headers and JSON body); ``reply(body)`` gives the
    assistant's text, by default "A".
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.reply = lambda body: "A"
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                payload = json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": server.reply(body)},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def chat_server(monkeypatch):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    server = ChatServer()
    yield server
    server.close()


def write_counter_video(path, seconds: float, fps: int = 10, size=(64, 48)):
    """An H.264 video whose frame i is one flat color that encodes i (see counter_index).

    Keyframes come every 2 s, so reads more than 2 s apart seek."""
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 2 * fps
        for i in range(round(seconds * fps)):
            pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            pixels[..., 0], pixels[..., 1] = 16 * (i % 16), 16 * ((i // 16) % 16)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def counter_index(image) -> int:
    """The frame index a counter-video frame encodes (exact for the first 256 frames)."""
    pixels = np.asarray(image.convert("RGB"), dtype=float)
    return 16 * round(pixels[..., 1].mean() / 16) + round(pixels[..., 0].mean() / 16)


@pytest.fixture
def counter_video():
    return write_counter_video


@pytest.fixture
def frame_index():
    return counter_index
