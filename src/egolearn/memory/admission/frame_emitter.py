"""FrameEmitter: pass every (subsampled) video frame through as a RawFrame.

Memory Storyboard admits frame-granularity items into both memory tiers; the
storyboard structure lives entirely in the LabelStore maintained by labeler
components.  This emitter is that admission path.
"""

from __future__ import annotations

from egolearn.config.registry import ADMISSION
from egolearn.core import keys as K
from egolearn.core.events import make_event
from egolearn.data.sources.base import Observation

from .base import EventBuilder


@ADMISSION.register("frame_emitter")
class FrameEmitter(EventBuilder):
    subscribes = ("video",)
    emits = ("RawFrame",)

    def __init__(self, subsample: int = 1):
        self.subsample = max(1, int(subsample))
        self._count = 0
        self._out: list[dict] = []

    def ingest(self, obs: Observation) -> None:
        if obs.channel != "video":
            return
        if self._count % self.subsample == 0:
            self._out.append(
                make_event(
                    "RawFrame",
                    **{
                        K.TIME_SESSION: obs.session,
                        K.TIME_T0: obs.t,
                        K.TIME_T1: obs.t,
                        K.VIDEO_REF: obs.ref,
                    },
                )
            )
        self._count += 1

    def poll(self) -> list[dict]:
        out, self._out = self._out, []
        return out
