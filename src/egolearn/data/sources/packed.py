"""PackedFrameStream: the SAYCam/BabyView adapter over a packed frame store
(egolearn.data.packed) instead of per-frame files.

Same sessions, times, channels, refs and pixels as `stream:saycam` on the
frame directories the store was built from -- only the read path changes:
a frame is one `pread` into a large shard, a contiguous clip is one read, and
the store's memory-mapped index replaces the per-session directory listings
(so the stream pickles to a few kilobytes plus the transcript when it is handed
to Loader workers).  Build a store with `python -m egolearn.data.packed pack`.
"""

from __future__ import annotations

import fnmatch

from PIL import Image

from egolearn.config.registry import STREAMS
from egolearn.core.refs import ItemRef
from egolearn.data.packed import PackedFrameStore, decode_frame

from . import base as S
from .saycam import SaycamStream


@STREAMS.register("packed_frames")
class PackedFrameStream(SaycamStream):
    def __init__(
        self,
        root_dir: str,
        transcript_json: str | None = None,
        fps: float | None = None,
        naming: str | None = None,
        sessions: list[str] | None = None,
        include: str | None = None,
        transcript_first: bool = False,
    ):
        """`root_dir` is a packed store directory; `fps` and the frame naming
        scheme come from its manifest, and a config that still carries them
        (a frame-dir config with the backend swapped) is checked against it.
        `sessions` (explicit order) and `include` (fnmatch on session ids)
        select among the packed sessions like the frame-dir adapter does
        among directories."""
        self.store = PackedFrameStore(root_dir)
        if fps is not None and float(fps) != self.store.fps:
            raise ValueError(
                f"stream.fps={fps} but the packed store was built at fps={self.store.fps}"
            )
        if naming is not None and naming != self.store.naming:
            raise ValueError(
                f"stream.naming={naming!r} but the packed store uses {self.store.naming!r}"
            )
        super().__init__(
            root_dir,
            transcript_json=transcript_json,
            fps=self.store.fps,
            naming=self.store.naming,
            sessions=sessions,
            include=include,
            transcript_first=transcript_first,
        )

    # ---- session enumeration (manifest order, not a directory scan) -------
    def session_ids(self) -> list[str]:
        packed = [s.id for s in self.store.sessions()]
        if self._session_ids is not None:
            known = set(packed)
            empty = set(self.store.empty_sessions())  # selected at build time, no frames
            for sid in self._session_ids:
                if sid not in known and sid not in empty:
                    raise KeyError(
                        f"session {sid!r} is not in the packed store {self.store.root!r}"
                    )
            return [sid for sid in self._session_ids if sid in known]
        if self._include is not None:
            return [sid for sid in packed if fnmatch.fnmatch(sid, self._include)]
        return packed

    def _frames(self, video_dir: str) -> list[str]:
        # the source file names, kept in the store: storyboards and other
        # precomputed indexes validate their frame order against them
        if video_dir not in self._frames_cache:
            self._frames_cache[video_dir] = self.store.session_names(video_dir)
        return self._frames_cache[video_dir]

    def sessions(self):
        for sid in self.session_ids():
            span = self.store.span(sid)
            times = self.store.session_times(sid).tolist()
            video = S.DenseFrameChannel("video", sid, times, [(sid, i) for i in range(span.count)])
            channels: dict[str, S.Channel] = {"video": video}
            utts = self._utts_by_stem.get(self._stem(sid), [])
            if utts:
                channels["transcript"] = S.SparseIntervalChannel("transcript", sid, utts)
            yield S.Session(sid, channels)

    def total_observations(self) -> int | None:
        if self._total is None:
            self._total = sum(
                self.store.span(sid).count + len(self._utts_by_stem.get(self._stem(sid), []))
                for sid in self.session_ids()
            )
        return self._total

    # ---- reading pixels ------------------------------------------------------
    def _global(self, ref: ItemRef, index: int) -> int:
        video_dir, i0, i1 = self._locator_span(ref)
        if not (0 <= index < i1 - i0):
            raise IndexError(f"frame index {index} outside ref span {i1 - i0}")
        span = self.store.span(video_dir)
        if not (0 <= i0 + index < span.count):  # a stale ref must not read a neighbour
            raise IndexError(
                f"frame {i0 + index} outside session {video_dir} ({span.count} frames)"
            )
        return span.first + i0 + index

    def open_frame(self, ref: ItemRef, index: int) -> Image.Image:
        return decode_frame(self.store.read(self._global(ref, index)))

    def open_frames(self, ref: ItemRef, indices: list[int]) -> list[Image.Image]:
        """Several frames of one ref in one read (clips); repeats decode once."""
        keys = [self._global(ref, i) for i in indices]
        blobs = self.store.read_many(keys)
        decoded: dict[int, Image.Image] = {}
        for key, blob in zip(keys, blobs, strict=True):
            if key not in decoded:
                decoded[key] = decode_frame(blob)
        return [decoded[key] for key in keys]

    def close(self) -> None:
        self.store.close()

    def __getstate__(self) -> dict:
        # workers rebuild the listing cache from the store's names on demand;
        # pickling it would hand every worker the full listing again
        state = self.__dict__.copy()
        state["_frames_cache"] = {}
        return state
