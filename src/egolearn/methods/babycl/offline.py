"""IID BabyCL batches from a complete offline storyboard and aligned utterances.

The learner starts from scratch with the whole dataset available. Every step draws
three storyboard segments independently with replacement, samples 32 frames from
each, and adds 32 image/utterance pairs. TCL retains BabyCL's batch-local labels.

An ``utterances`` epoch shuffles the complete AN-chunk index without replacement
and drops its incomplete final batch (the original BabyCL IID baseline's epoch).
A ``frame_budget`` epoch instead draws independent batches for
``floor(dataset_frames / images_per_batch)`` steps, matching an image-only IID
epoch's image budget. This is an exposure budget, not a once-per-frame traversal.
Sampling and augmentation derive from compose_seq, so prefetch and resume agree.

``segments_per_batch: 0`` is the pairs-only (CVCL-style) batch: every step draws
image/utterance pairs and nothing else, no storyboard is needed, and the visual
losses have no slot to consume.
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
from egolearn.methods.babycl.an_chunks import ANChunkEmitter
from egolearn.training.offline import OfflineDriver


@dataclass
class BabyCLOfflineIndex:
    segments: list[dict]
    pairs: list[dict]
    num_frames: int
    storyboard_sha256: str | None  # None for pairs-only indexes


def build_offline_index(stream: SaycamStream, storyboard_json: str | None) -> BabyCLOfflineIndex:
    """Validate storyboard frame order and build raw timestamp-run pairs.

    Ignore the storyboard's stored root_dir: the configured stream owns paths.
    Use the same ANChunkEmitter as streaming to retain its timestamp matching,
    contiguous-run splitting, missing-frame handling, and lifetime deduplication.
    Without a storyboard (pairs-only batches) the index holds no segments and
    still counts every frame, so frame-budget epochs keep their meaning.
    """
    videos: dict[str, dict] | None = None
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
    segments, pairs = [], []
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
            if video["frames"] != stream._frames(sid):
                raise ValueError(f"offline storyboard frame order differs for {sid!r}")
            covered_until = 0
            for segment in video["segments"]:
                start, end = int(segment["start"]), int(segment["end"])
                if start != covered_until or not start < end <= len(dense):
                    raise ValueError(
                        f"offline storyboard has gaps/overlap or invalid bounds in {sid!r}"
                    )
                ref = dense.interval_ref(start, end)
                segments.append({K.VIDEO_REF: ref})
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
    return BabyCLOfflineIndex(segments, pairs, num_frames, storyboard_sha256)


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
        epoch_basis: str = "utterances",
        **kwargs,
    ):
        if epoch_basis not in ("utterances", "frame_budget"):
            raise ValueError("epoch_basis must be 'utterances' or 'frame_budget'")
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
            n=pairs_per_batch if epoch_basis == "utterances" else 1,
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
        self.epoch_basis = epoch_basis
        self.segments_per_batch = int(segments_per_batch)
        self.frames_per_segment = int(frames_per_segment)
        self.pairs_per_batch = int(pairs_per_batch)
        self.images_per_batch = (
            self.segments_per_batch * self.frames_per_segment + self.pairs_per_batch
        )
        self.composer = Composer(self.loader, tokenizer=self.tokenizer)
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
                    batch_label=True,
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
        return "utterance chunks" if self.epoch_basis == "utterances" else "frame-budget batches"

    @property
    def batch_description(self) -> str:
        visual = (
            f"segments({self.segments_per_batch}x{self.frames_per_segment} frames, 2 views) + "
            if self.segments_per_batch > 0
            else ""
        )
        return f"{visual}pairs({self.pairs_per_batch} frames, text); epoch_basis={self.epoch_basis}"

    def _build_index(self):
        if self.index is None:
            assert isinstance(self.stream, SaycamStream)
            self.index = build_offline_index(self.stream, self.storyboard_json)
        index = self.index
        if len(index.pairs) < self.pairs_per_batch:
            raise ValueError("not enough utterance chunks for a complete BabyCL batch")
        print(
            f"[babycl-iid] {index.num_frames} frames, {len(index.segments)} segments, "
            f"{len(index.pairs)} utterance chunks; epoch_basis={self.epoch_basis}; "
            f"batch={self.segments_per_batch}x{self.frames_per_segment} visual"
            f"+{self.pairs_per_batch} image/text; storyboard_sha256={index.storyboard_sha256}",
            flush=True,
        )
        if not self.segments_per_batch:
            print("[babycl-iid] pairs-only batches (segments_per_batch=0)", flush=True)
        if self.epoch_basis == "utterances":
            return index.pairs
        return range(index.num_frames // self.images_per_batch)

    def _compose(self, refs, seq: int | None = None) -> dict:
        if self.index is None:
            raise RuntimeError("build the BabyCL offline index before composing batches")
        if seq is None:
            seq = self.ctx.clock.compose_seq
            self.ctx.clock.compose_seq = seq + 1
        rng = self.ctx.fresh_rng(f"babycl.offline.draw.{seq}")
        segments = rng.choices(self.index.segments, k=self.segments_per_batch)
        pairs = (
            refs
            if self.epoch_basis == "utterances"
            else rng.sample(self.index.pairs, self.pairs_per_batch)
        )
        batch = self.composer.compose(
            self.batch_recipe,
            {},
            {"VisualSegment": segments, "AlignedPair": pairs},
            self.ctx,
            seq=seq,
        )
        if batch is None:
            raise RuntimeError("offline BabyCL could not fill all loss slots")
        return batch
