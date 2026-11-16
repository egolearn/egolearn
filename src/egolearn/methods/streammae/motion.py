"""Causal, patch-level motion metadata computed once when each frame arrives."""

from __future__ import annotations

import numpy as np
from PIL import Image

from egolearn.config.registry import ADMISSION
from egolearn.core import keys as K
from egolearn.core.events import make_event
from egolearn.core.refs import ItemRef
from egolearn.data.sources.base import Observation
from egolearn.memory.admission.base import EventBuilder

CONTEXT_KIND = "streammae_motion"


def prepare_image(image: Image.Image, frame_size: tuple[int, int] | None) -> Image.Image:
    """Use the identical RGB geometry during admission and augmentation (H, W)."""
    image = image.convert("RGB")
    if frame_size is not None:
        image = image.resize((frame_size[1], frame_size[0]), Image.Resampling.BICUBIC)
    return image


def frame_geometry(frame_size) -> tuple[int, int] | None:
    if frame_size is None:
        return None
    if len(frame_size) != 2 or any(int(n) != n or n < 1 for n in frame_size):
        raise ValueError("frame_size must be positive [height, width] or null")
    return int(frame_size[0]), int(frame_size[1])


def patch_motion(current: np.ndarray, previous: np.ndarray | None, patch_size: int) -> np.ndarray:
    """RGB L1 change averaged within complete patches; zero at discontinuities.

    Right/bottom partial patches are excluded from candidate first-stage crops.
    Differences use signed arithmetic so uint8 subtraction cannot wrap around.
    """
    if patch_size < 1 or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("motion needs RGB pixels and a positive patch size")
    hh, ww = current.shape[0] // patch_size, current.shape[1] // patch_size
    if min(hh, ww) < 1:
        raise ValueError("frame is smaller than one motion patch")
    if previous is None or previous.shape != current.shape:
        return np.zeros((hh, ww), dtype=np.float32)
    difference = np.abs(current.astype(np.int16) - previous.astype(np.int16)).sum(axis=2)
    return (
        difference[: hh * patch_size, : ww * patch_size]
        .reshape(hh, patch_size, ww, patch_size)
        .mean(axis=(1, 3), dtype=np.float32)
    )


@ADMISSION.register("streammae_frames")
class MotionFrames(EventBuilder):
    """Emit frames with cached motion maps BEFORE example selection/DataDrop.

    Keeps only the preceding decoded frame. The FIFO owns maps for its window.
    Session starts and geometry changes have zero motion (uniform crop choice).
    On resume, replay reconstructs the preceding ref without decoding history;
    the first live frame reloads that predecessor. Restored FIFO entries already
    contain their maps. Use with tier sampling, not replay-built stream_head.
    """

    subscribes = ("video",)
    emits = ("RawFrame",)
    replay_requires_memory = True
    emits_each_frame = True
    view_context_kind = CONTEXT_KIND

    def __init__(self, patch_size: int = 16, frame_size=None):
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        self.frame_size = frame_geometry(frame_size)
        self.previous: np.ndarray | None = None
        self.previous_ref: ItemRef | None = None
        self.replaying = False
        self._out: list[dict] = []

    def start_session(self, sid: str) -> None:
        self.previous = None
        self.previous_ref = None

    def set_replaying(self, replaying: bool) -> None:
        self.replaying = replaying

    def _read(self, ref: ItemRef) -> np.ndarray:
        assert self.loader is not None
        return np.array(prepare_image(self.loader.stream.open_frame(ref, 0), self.frame_size))

    def ingest(self, obs: Observation) -> None:
        if obs.channel != "video":
            return
        if not obs.ref.is_point:
            raise ValueError("StreamMAE admission requires point frames")
        context = None
        if not self.replaying:
            current = self._read(obs.ref)
            assert self.loader is not None
            for name in self.loader.views.names():
                view = self.loader.views.get(name)
                if getattr(view, "context_kind", None) == CONTEXT_KIND:
                    view.validate_frame_size(current.shape[:2])
            if self.previous is None and self.previous_ref is not None:
                self.previous = self._read(self.previous_ref)
            context = {
                "motion": patch_motion(current, self.previous, self.patch_size),
                "patch_size": self.patch_size,
                "frame_size": current.shape[:2],
            }
            self.previous = current
        self.previous_ref = obs.ref
        self._out.append(
            make_event(
                "RawFrame",
                **{
                    K.TIME_SESSION: obs.session,
                    K.TIME_T0: obs.t,
                    K.TIME_T1: obs.t,
                    K.VIDEO_REF: obs.ref,
                    K.VIEW_CONTEXT: context,
                },
            )
        )

    def poll(self) -> list[dict]:
        result, self._out = self._out, []
        return result
