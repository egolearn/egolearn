"""ToyStream: a synthetic video source with known ground truth, for tests.

It exists so that admission, memory, and the training driver can be
unit-tested against known event boundaries without touching real data.
Each session is a sequence of solid-color blocks (`block_len` frames per
color); an utterance naming the color is emitted once per block.  Ground
truth: segment boundaries == color changes, and each utterance aligns with
its block.

Everything is generated deterministically in memory; `open_frame` draws each
image on demand instead of reading from disk.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from egolearn.config.registry import STREAMS
from egolearn.core import keys as K
from egolearn.core.refs import ItemRef

from . import base as S

_PALETTE = [
    ("red", (200, 40, 40)),
    ("green", (40, 170, 60)),
    ("blue", (50, 80, 200)),
    ("yellow", (210, 200, 50)),
    ("purple", (140, 60, 170)),
    ("cyan", (60, 190, 190)),
]


@STREAMS.register("toy")
class ToyStream(S.Stream):
    def __init__(
        self,
        n_sessions: int = 2,
        blocks_per_session: int = 6,
        block_len: int = 50,
        fps: float = 5.0,
        img_size: int = 64,
        noise: int = 12,
        seed: int = 0,
    ):
        self.n_sessions = n_sessions
        self.blocks = blocks_per_session
        self.block_len = block_len
        self.fps = fps
        self.img_size = img_size
        self.noise = noise
        self.seed = seed

    # ---- ground truth helpers (for tests) ----------------------------------
    def true_boundaries(self) -> list[int]:
        return [b * self.block_len for b in range(1, self.blocks)]

    def _color_of(self, session_idx: int, frame_idx: int):
        block = min(frame_idx // self.block_len, self.blocks - 1)
        name, rgb = _PALETTE[(session_idx + block) % len(_PALETTE)]
        return block, name, rgb

    # ---- Stream interface ---------------------------------------------------
    def sessions(self):
        n_frames = self.blocks * self.block_len
        for s in range(self.n_sessions):
            sid = f"toy_{s:02d}"
            times = [i / self.fps for i in range(n_frames)]
            video = S.DenseFrameChannel("video", sid, times, [(sid, i) for i in range(n_frames)])
            utts = []
            for b in range(self.blocks):
                _, name, _ = self._color_of(s, b * self.block_len)
                t0 = (b * self.block_len + 2) / self.fps
                t1 = (b * self.block_len + 6) / self.fps
                utts.append(
                    {
                        "t0": t0,
                        "t1": t1,
                        "payload": {
                            K.TEXT_UTTERANCE: f"look a {name} thing",
                            K.UTTERANCE_NUM: s * self.blocks + b,
                        },
                    }
                )
            yield S.Session(
                sid,
                {
                    "video": video,
                    "transcript": S.SparseIntervalChannel("transcript", sid, utts),
                },
            )

    def total_observations(self):
        return self.n_sessions * (self.blocks * self.block_len + self.blocks)

    # ---- reading pixels (refs become images only here) ----------------------
    def _span(self, ref: ItemRef):
        loc = ref.locator
        if len(loc) == 2:
            return loc[0], loc[1], loc[1] + 1
        return loc[0], loc[1], loc[2]

    def num_frames(self, ref: ItemRef) -> int:
        _, i0, i1 = self._span(ref)
        return i1 - i0

    def open_frame(self, ref: ItemRef, index: int) -> Image.Image:
        sid, i0, _ = self._span(ref)
        s = int(sid.split("_")[1])
        fidx = i0 + index
        _, _, rgb = self._color_of(s, fidx)
        rng = np.random.RandomState((self.seed * 1_000_003 + s * 100_003 + fidx) % (2**31))
        img = np.clip(
            np.array(rgb, dtype=np.int16)[None, None, :]
            + rng.randint(-self.noise, self.noise + 1, (self.img_size, self.img_size, 3)),
            0,
            255,
        ).astype(np.uint8)
        return Image.fromarray(img, "RGB")
