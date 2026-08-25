"""FixedWindowLabeler: label each frame as global_frame_index // class_length.

This reproduces Memory Storyboard's initial labeling
(StreamingDataset.get_default_label): frames enter memory labeled by which
fixed-length chunk of the stream they fall in, and the recluster labeler
later refines those labels.  Used on its own -- fixed windows with no
refinement -- it is the ablation baseline.

It is a pure labeler: it emits no events and only writes the LabelStore.
"""

from __future__ import annotations

from egolearn.config.registry import ADMISSION
from egolearn.data.sources.base import Observation

from .base import EventBuilder


@ADMISSION.register("fixed_window_labeler")
class FixedWindowLabeler(EventBuilder):
    subscribes = ("video",)
    emits = ()

    def __init__(self, class_length: int):
        if class_length <= 0:
            raise ValueError("class_length must be positive")
        self.class_length = int(class_length)
        self._global_idx = 0  # counter keeps running across sessions (one long stream)

    def ingest(self, obs: Observation) -> None:
        if obs.channel != "video":
            return
        assert self.label_store is not None, "fixed_window_labeler needs a label_store"
        self.label_store.assign([obs.ref.key], [self._global_idx // self.class_length])
        self._global_idx += 1
