"""Index complete storyboard segments and aligned, deduplicated utterance chunks.

Each epoch shuffles the chunks once and drops the incomplete batch. Visual
segments are drawn with replacement; sampling and augmentation derive from
compose_seq so prefetch and snapshot resume reproduce the consumed batches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from egolearn.core import keys as K
from egolearn.data.batching.composer import Composer
from egolearn.data.batching.recipe import Recipe, SlotSpec
from egolearn.data.sources.base import DenseFrameChannel
from egolearn.data.sources.saycam import SaycamStream
from egolearn.memory.labels import LabelStore
from egolearn.methods.babycl.an_chunks import ANChunkEmitter
from egolearn.training.offline import OfflineDriver


@dataclass
class BabyCLOfflineIndex:
    segments: list[dict]
    pairs: list[dict]
    num_frames: int
    storyboard_sha256: str | None  # None for pairs-only indexes
    segment_labels: list[int]


def build_offline_index(stream: SaycamStream, storyboard_json: str | None) -> BabyCLOfflineIndex:
    """Validate storyboard frame order and build raw timestamp-run pairs.

    Ignore the storyboard's stored root_dir: the configured stream owns paths.
    Use the same ANChunkEmitter as streaming to retain its timestamp matching,
    contiguous-run splitting, missing-frame handling, and lifetime deduplication.
    Without a storyboard, pairs-only batches index utterances and no segments.
    """
    videos: dict[str, dict] | None = None
    segment_ids: dict[str, list[int]] = {}
    storyboard_sha256 = None
    if storyboard_json is not None:
        raw = Path(storyboard_json).read_bytes()
        storyboard_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if float(manifest["fps"]) != stream.fps:
            raise ValueError("offline storyboard fps differs from stream.fps")
        videos = {v["video_dir"]: v for v in manifest["videos"]}
        if len(videos) != len(manifest["videos"]):
            raise ValueError("offline storyboard has duplicate session entries")
        for sid, video in videos.items():
            if not isinstance(video.get("segments"), list):
                raise ValueError(f"offline storyboard segments must be a list in session {sid!r}")
        explicit_ids = any("global_id" in s for v in videos.values() for s in v["segments"])
        next_id = 0
        for sid, video in videos.items():
            segment_ids[sid] = []
            for segment in video["segments"]:
                if explicit_ids and "global_id" not in segment:
                    raise ValueError(
                        "offline storyboard must supply global_id for every segment; "
                        f"missing in session {sid!r}"
                    )
                gid = segment.get("global_id", next_id)
                if type(gid) is not int or not 0 <= gid < 2**63:
                    raise ValueError(
                        f"offline storyboard global_id in session {sid!r} "
                        "must be a nonnegative int64"
                    )
                segment_ids[sid].append(gid)
                next_id += 1
    segments, pairs = [], []
    segment_labels = []
    num_frames = 0
    emitter = ANChunkEmitter(fps=stream.fps)
    for session in stream.sessions():
        sid = session.sid
        dense = session.channels["video"]
        assert isinstance(dense, DenseFrameChannel)
        num_frames += len(dense)
        if videos is not None:
            if sid not in videos:
                raise ValueError(f"offline storyboard is missing selected session {sid!r}")
            video = videos[sid]
            # Storyboard ranges address this adapter's filename order. A count-only
            # check would silently pair the wrong pixels with segments/utterances.
            if video["frames"] != stream.frame_files(sid):
                raise ValueError(f"offline storyboard frame order differs for {sid!r}")
            covered_until = 0
            for segment, gid in zip(video["segments"], segment_ids[sid], strict=True):
                start, end = int(segment["start"]), int(segment["end"])
                if start != covered_until or not start < end <= len(dense):
                    raise ValueError(
                        f"offline storyboard has gaps/overlap or invalid bounds in {sid!r}"
                    )
                ref = dense.interval_ref(start, end)
                segments.append({K.VIDEO_REF: ref})
                segment_labels.append(gid)
                covered_until = end
            if covered_until != len(dense):
                raise ValueError(f"offline storyboard does not cover all frames in {sid!r}")

        emitter.start_session(sid)
        for observation in session.observations():
            emitter.ingest(observation)
            pairs.extend(emitter.poll())
        emitter.end_session()
        pairs.extend(emitter.poll())
    if not pairs or (videos is not None and not segments):
        raise ValueError("offline BabyCL requires nonempty storyboard segments and utterance pairs")
    return BabyCLOfflineIndex(segments, pairs, num_frames, storyboard_sha256, segment_labels)


class BabyCLOfflineDriver(OfflineDriver):
    """BabyCL's mixed visual/verbal batch in the resumable offline training loop."""

    def __init__(
        self,
        *,
        pair_view: str,
        storyboard_json: str | None = None,
        segment_views: list[str] | None = None,
        segments_per_batch: int = 3,
        frames_per_segment: int = 32,
        pairs_per_batch: int = 32,
        **kwargs,
    ):
        if segments_per_batch < 0 or min(frames_per_segment, pairs_per_batch) < 1:
            raise ValueError("BabyCL batch counts must be positive (segments_per_batch may be 0)")
        segment_views = list(segment_views or [])
        if segments_per_batch > 0:
            if storyboard_json is None:
                raise ValueError("BabyCL segments need driver.batch.storyboard_json")
            if len(segment_views) != 2:
                raise ValueError("BabyCL needs two segment views")
        elif storyboard_json is not None or segment_views:
            raise ValueError(
                "segments_per_batch=0 is the pairs-only batch: drop storyboard_json "
                "and segment_views"
            )
        super().__init__(
            slot="segments+pairs" if segments_per_batch > 0 else "pairs",
            n=pairs_per_batch,
            views=[*segment_views, pair_view],
            **kwargs,
        )
        if self.subsample != 1:
            raise ValueError("BabyCL offline indexes the complete selected dataset; subsample=1")
        if self.unit != "frame":
            raise ValueError("BabyCL offline supplies image frames")
        if not isinstance(self.stream, SaycamStream):
            raise ValueError("BabyCL offline storyboard requires a SAYCam/BabyView frame stream")
        self.storyboard_json = storyboard_json
        self.segments_per_batch = int(segments_per_batch)
        self.frames_per_segment = int(frames_per_segment)
        self.pairs_per_batch = int(pairs_per_batch)
        self.composer = Composer(
            self.loader, tokenizer=self.tokenizer, label_stores={"events": LabelStore("events")}
        )
        slots = []
        if self.segments_per_batch > 0:
            slots.append(
                SlotSpec(
                    name="segments",
                    source={"kind": "stream_head", "schema": "VisualSegment"},
                    n=self.segments_per_batch,
                    unit="segment",
                    frames_per_item=self.frames_per_segment,
                    views=segment_views,
                    labels_from="events",
                )
            )
        slots.append(
            SlotSpec(
                name="pairs",
                source={"kind": "stream_head", "schema": "AlignedPair"},
                n=self.pairs_per_batch,
                unit="pair",
                views=[pair_view],
                text=True,
            )
        )
        self.batch_recipe = Recipe(slots=slots)
        self.index: BabyCLOfflineIndex | None = None

    @property
    def slot_names(self) -> set[str]:
        return {"segments", "pairs"} if self.segments_per_batch > 0 else {"pairs"}

    @property
    def text_slots(self) -> set[str]:
        return {"pairs"}

    @property
    def index_units(self) -> str:
        return "utterance chunks"

    @property
    def batch_description(self) -> str:
        visual = (
            f"segments({self.segments_per_batch}x{self.frames_per_segment} frames, 2 views) + "
            if self.segments_per_batch > 0
            else ""
        )
        return f"{visual}pairs({self.pairs_per_batch} frames, text)"

    def _build_index(self):
        if self.index is None:
            assert isinstance(self.stream, SaycamStream)
            self.index = build_offline_index(self.stream, self.storyboard_json)
        index = self.index
        self.composer.label_stores["events"].assign(
            [segment[K.VIDEO_REF].key for segment in index.segments], index.segment_labels
        )
        self.snapshot_metadata = {"storyboard_sha256": index.storyboard_sha256}
        if len(index.pairs) < self.pairs_per_batch:
            raise ValueError("not enough utterance chunks for a complete BabyCL batch")
        print(
            f"[babycl-iid] {index.num_frames} frames, {len(index.segments)} segments, "
            f"{len(index.pairs)} utterance chunks; "
            f"batch={self.segments_per_batch}x{self.frames_per_segment} visual"
            f"+{self.pairs_per_batch} image/text; storyboard_sha256={index.storyboard_sha256}",
            flush=True,
        )
        if not self.segments_per_batch:
            print("[babycl-iid] pairs-only batches (segments_per_batch=0)", flush=True)
        return index.pairs

    def validate_snapshot(self, state: dict) -> None:
        """Reject changed segment labels before restoring offline training state."""
        if self.storyboard_json is None:
            return
        metadata = state.get("driver", {}).get("metadata") or {}
        digest = metadata.get("storyboard_sha256")
        if digest is None:
            inputs = (state.get("run_provenance") or {}).get("input_provenance", {})
            digest = inputs.get("driver.batch.storyboard_json", {}).get("sha256")
        if digest is None:
            raise ValueError("snapshot lacks storyboard identity; start a new run")
        if digest != hashlib.sha256(Path(self.storyboard_json).read_bytes()).hexdigest():
            raise ValueError("storyboard differs from the training snapshot")

    def _compose(self, refs, seq: int | None = None) -> dict:
        if self.index is None:
            raise RuntimeError("build the BabyCL offline index before composing batches")
        if seq is None:
            seq = self.ctx.clock.compose_seq
            self.ctx.clock.compose_seq = seq + 1
        rng = self.ctx.fresh_rng(f"babycl.offline.draw.{seq}")
        segments = rng.choices(self.index.segments, k=self.segments_per_batch)
        batch = self.composer.compose(
            self.batch_recipe,
            {},
            {"VisualSegment": segments, "AlignedPair": refs},
            self.ctx,
            seq=seq,
        )
        if batch is None:
            raise RuntimeError("offline BabyCL could not fill all loss slots")
        return batch
