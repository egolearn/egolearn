"""FixedWindowLabeler: default labels = global_frame_index // class_length.

This is Memory Storyboard's PRIOR labeling (StreamingDataset.get_default_label):
frames enter memory with fixed-length chunk labels, which the recluster
labeler later refines.  It is also a complete admission component on its own
-- fixed windows with no refinement is the ablation baseline.

Pure labeler: emits nothing; writes the LabelStore.
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
        self._global_idx = 0  # frames are counted across sessions (one stream)

    def ingest(self, obs: Observation) -> None:
        if obs.channel != "video":
            return
        assert self.label_store is not None, "fixed_window_labeler needs a label_store"
        self.label_store.assign([obs.ref.key], [self._global_idx // self.class_length])
        self._global_idx += 1
