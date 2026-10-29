"""Attach fixed temporal-class targets to globally shuffled video frames."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

import torch

from egolearn.core import keys as K
from egolearn.core.refs import ItemRef
from egolearn.data.sources.base import DenseFrameChannel
from egolearn.data.sources.video import VideoStream
from egolearn.training.offline import OfflineDriver


class TemporalFrameDriver(OfflineDriver):
    """Train on IID frames using a complete, nonoverlapping temporal-label manifest."""

    def __init__(self, *, temporal_manifest: str, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.stream, VideoStream) or self.unit != "frame":
            raise ValueError("temporal IID requires VideoStream and unit=frame")
        manifest_bytes = Path(temporal_manifest).read_bytes()
        document = json.loads(manifest_bytes)
        self.snapshot_metadata = {
            "temporal_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()
        }
        if document.get("format_version") != 1:
            raise ValueError("unsupported temporal manifest format_version")
        if document.get("num_classes") != num_classes or num_classes < 1:
            raise ValueError("temporal manifest num_classes disagrees with the configured classes")
        self.temporal_sessions = document["sessions"]
        self._ends: dict[str, list[int]] = {}
        self._labels: dict[str, list[int]] = {}
        if set(self.temporal_sessions) != set(self.stream.session_ids()):
            raise ValueError("temporal manifest sessions must match the video stream exactly")
        for sid, entry in self.temporal_sessions.items():
            count = entry["num_frames"]
            fps = entry["fps"]
            if type(count) is not int or count < 0 or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"invalid temporal metadata for {sid}")
            previous = 0
            ends, labels = [], []
            for start, end, label in entry["segments"]:
                if (
                    any(type(x) is not int for x in (start, end, label))
                    or start != previous
                    or not start < end <= count
                    or not 0 <= label < num_classes
                ):
                    raise ValueError(f"invalid temporal coverage or label for {sid}")
                ends.append(end)
                labels.append(label)
                previous = end
            if previous != count:
                raise ValueError(f"incomplete temporal coverage for {sid}")
            self._ends[sid], self._labels[sid] = ends, labels
        for loss in self.losses:
            if hasattr(loss, "num_classes") and loss.num_classes != num_classes:
                raise ValueError("temporal loss and manifest num_classes disagree")

    def _build_index(self):
        refs: list[ItemRef] = []
        visited = set()
        for session in self.stream.sessions():
            visited.add(session.sid)
            channel = session.channels["video"]
            if not isinstance(channel, DenseFrameChannel):
                raise ValueError("temporal IID requires a dense video channel")
            entry = self.temporal_sessions[session.sid]
            if len(channel) != entry["num_frames"] or any(
                not math.isclose(t, i / entry["fps"], rel_tol=1e-9, abs_tol=1e-9)
                for i, t in enumerate(channel.times)
            ):
                raise ValueError(f"temporal metadata disagrees with video {session.sid}")
            refs.extend(channel.ref_at(i) for i in range(0, len(channel), self.subsample))
        for sid in self.temporal_sessions.keys() - visited:
            if self.temporal_sessions[sid]["num_frames"] != 0:
                raise ValueError(f"temporal metadata disagrees with empty video {sid}")
        return refs

    def validate_snapshot(self, state: dict) -> None:
        """Reject changed or unknown temporal targets before restoring training state."""
        metadata = state.get("driver", {}).get("metadata") or {}
        digest = metadata.get("temporal_manifest_sha256")
        if digest is None:
            # Older CLI snapshots retain this identity in their run provenance.
            inputs = (state.get("run_provenance") or {}).get("input_provenance", {})
            digest = inputs.get("driver.batch.temporal_manifest", {}).get("sha256")
        if digest is None:
            raise ValueError("snapshot lacks temporal manifest identity; start a new run")
        if digest != self.snapshot_metadata["temporal_manifest_sha256"]:
            raise ValueError("temporal manifest differs from the training snapshot")

    def _compose(self, refs, seq: int | None = None) -> dict:
        batch = super()._compose(refs, seq)
        batch[self.slot][K.SLOT_LABELS] = torch.tensor(
            [
                self._labels[ref.session][
                    bisect.bisect_right(self._ends[ref.session], ref.locator[1])
                ]
                for ref in refs
            ],
            dtype=torch.long,
        )
        return batch
