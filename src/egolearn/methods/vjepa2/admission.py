"""Causal, nonoverlapping dense video clips for the V-JEPA experiments."""

from __future__ import annotations

import math

from egolearn.config.registry import ADMISSION
from egolearn.core import keys as K
from egolearn.core.events import make_event
from egolearn.data.sources.base import DenseFrameChannel, Observation, Session
from egolearn.memory.admission.base import EventBuilder


@ADMISSION.register("dense_clip")
class DenseClipEmitter(EventBuilder):
    """Emit an exact interval only once its last frame has been observed.

    No pixel decoding, padding, cross-session windows or unfinished tail events.
    The channel owns locator construction, so admission does not interpret paths.
    """

    emits = ("VisualSegment",)

    def __init__(self, clip_len: int = 16, clip_stride: int = 1, max_frame_gap: float = 0.3):
        if clip_len < 2 or clip_stride < 1:
            raise ValueError("dense clips need clip_len >= 2 and clip_stride >= 1")
        if not math.isfinite(max_frame_gap) or max_frame_gap <= 0:
            raise ValueError("max_frame_gap must be positive and finite")
        self.clip_len, self.clip_stride = int(clip_len), int(clip_stride)
        self.span = (self.clip_len - 1) * self.clip_stride + 1
        self.max_frame_gap = float(max_frame_gap)
        self._out: list[dict] = []

    def begin_session(self, session: Session) -> None:
        channel = session.channels.get("video")
        if not isinstance(channel, DenseFrameChannel):
            raise ValueError("dense_clip requires a dense video channel")
        self.channel = channel
        self.index = self.begin = 0
        self.previous: float | None = None

    def ingest(self, obs: Observation) -> None:
        if obs.channel != "video":
            return
        if obs.ref != self.channel.ref_at(self.index) or obs.t != obs.ref.t0:
            raise ValueError("dense_clip observations must follow channel order")
        if not math.isfinite(obs.t):
            raise ValueError("dense_clip requires finite frame timestamps")
        if self.previous is not None:
            dt = obs.t - self.previous
            if dt <= 0 or dt > self.max_frame_gap:
                self.begin = self.index
        self.previous = obs.t
        self.index += 1
        if self.index - self.begin == self.span:
            ref = self.channel.interval_ref(self.begin, self.index)
            self._out.append(
                make_event(
                    "VisualSegment",
                    **{
                        K.TIME_SESSION: ref.session,
                        K.TIME_T0: ref.t0,
                        K.TIME_T1: ref.t1,
                        K.VIDEO_REF: ref,
                    },
                )
            )
            self.begin = self.index

    def poll(self) -> list[dict]:
        out, self._out = self._out, []
        return out

    def end_session(self) -> None:
        # The remaining frames cannot make a complete clip.
        self.begin = self.index
