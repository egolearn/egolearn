"""Port of BabyCL's OnlineSegmenter (mcl utils/online_ms_refine.py) as a
VisualSegment emitter.

Algorithm 1:1 with the original: sliding window over embed_stride-subsampled
frame embeddings; every utterance whose end is visible becomes an anchor
chunk, expanded by similarity within neighbor bounds (2-12 s); gap regions
are cut with cut_gap_region_by_similarity; small segments are packed into
~T_frames big segments; only segments left of the watermark
(win_right - seg_extend_max_sec * fps; everything on flush) are finalized;
global ids are monotone with adjacent-merge (cos >= global_merge_cos) and
carry across sessions (last_gid / last_rep_feat handover).

Framework adaptations (everything else verbatim):
  * frames arrive as Observations; embeddings come from the bound
    EmbeddingProvider in batches at recompute time -- identical values to the
    original's per-frame embedding with the same frozen encoder;
  * labels (global ids) are written once per emitted segment through the
    LabelStore instead of living inside the event;
  * the original also computes similarity-expanded utterance chunks; the mcl
    FEEDER discards them (`big, _ = seg.pop_ready_outputs()`) and emits AN
    chunks from raw timestamp runs instead -- so this component emits only
    VisualSegments, and admission:an_chunk_emitter reproduces the feeder path.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from egolearn.config.registry import ADMISSION
from egolearn.core import keys as K
from egolearn.core.events import make_event
from egolearn.core.refs import ItemRef
from egolearn.data.sources.base import Observation
from egolearn.memory.admission.base import EventBuilder

from .refine import (
    build_prefix_from_sim,
    cut_gap_region_by_similarity,
    expand_chunk_by_similarity,
    pick_representative_from_feats,
)


@ADMISSION.register("babycl_online_segmenter")
class BabyCLOnlineSegmenter(EventBuilder):
    subscribes = ("video", "transcript")
    emits = ("VisualSegment",)

    def __init__(
        self,
        fps: float = 5.0,
        avg_minutes: float = 3.0,
        avg_seg_frames: int | None = None,
        min_ratio: float = 0.5,
        embed_stride: int = 1,
        cp_subsample: int = 1,
        allow_last_short: bool = True,
        seg_extend_min_sec: float = 2.0,
        seg_extend_max_sec: float = 12.0,
        global_merge_cos: float = 0.85,
        window_scale: float = 2.0,
        compute_stride_frames: int = 10,
    ):
        self.fps = float(fps)
        # -- derived quantities, verbatim from OnlineSegmenter.__init__ ------
        if avg_seg_frames is not None:
            self.T_frames = int(avg_seg_frames)
        else:
            self.T_frames = round(float(avg_minutes) * 60.0 * float(fps))
        self.T_frames = max(1, self.T_frames)
        self.window_size_frames = max(round(self.T_frames * float(window_scale)), self.T_frames + 1)
        stride = max(1, int(embed_stride))
        self.embed_stride = stride
        self.extend_min_sub = max(1, round((seg_extend_min_sec * fps) / stride))
        self.extend_max_sub = max(self.extend_min_sub, round((seg_extend_max_sec * fps) / stride))
        self.min_seg_frames = max(1, math.floor(self.T_frames * min_ratio))
        self.T_sub = max(1, round(self.T_frames / stride))
        self.min_seg_sub = max(1, math.floor(self.min_seg_frames / stride))
        self.cp_subsample = max(1, int(cp_subsample))
        self.allow_last_short = bool(allow_last_short)
        self.global_merge_cos = float(global_merge_cos)
        self.seg_extend_max_sec = float(seg_extend_max_sec)
        self.compute_stride_frames = max(1, int(compute_stride_frames))

        # -- cross-session global-id continuity (feeder handover) -----------
        self._last_gid = -1
        self._last_rep: torch.Tensor | None = None

        self._out: list[dict] = []
        self._reset_session("")

    # ------------------------------------------------------------------
    def _reset_session(self, sid: str) -> None:
        self._sid = sid
        self._times: list[float] = []
        self._refs: list[ItemRef] = []
        self._sub_feats: list[torch.Tensor] = []  # [1,D] per embedded frame
        self._idx_map: list[int] = []  # sub idx -> full idx
        self._pending_embed: list[tuple[int, ItemRef]] = []
        self._pending_new_sub = 0
        self._utter_items: list[dict] = []  # visible utterances
        self._emitted_upto = -1  # emitted_upto_full

    def start_session(self, sid: str) -> None:
        self._reset_session(sid)

    def ingest(self, obs: Observation) -> None:
        if obs.session != self._sid:
            return
        if obs.channel == "transcript":
            # surfaced at its end time == "max_t visible" in the original
            self._utter_items.append(dict(obs.payload))
            return
        if obs.channel != "video":
            return
        idx_full = len(self._times)
        self._times.append(obs.t)
        self._refs.append(obs.ref)
        if idx_full % self.embed_stride == 0:
            self._pending_embed.append((idx_full, obs.ref))
            self._pending_new_sub += 1
        self._sync_window()
        if self._pending_new_sub >= max(1, self.compute_stride_frames // self.embed_stride):
            self._recompute_and_finalize()
            self._pending_new_sub = 0

    def end_session(self) -> None:
        self._recompute_and_finalize(force_flush=True)
        # last_gid / last_rep persist on the builder -> handed to next session

    def poll(self) -> list[dict]:
        out, self._out = self._out, []
        return out

    # ------------------------------------------------------------------
    def _sync_window(self) -> None:
        win_left = max(0, len(self._times) - self.window_size_frames)
        while self._idx_map and self._idx_map[0] < win_left:
            self._idx_map.pop(0)
            self._sub_feats.pop(0)

    def _flush_pending_embeds(self) -> None:
        if not self._pending_embed:
            return
        refs = [r for _, r in self._pending_embed]
        assert self.embedder is not None, "babycl_online_segmenter needs an embedder"
        feats = self.embedder.embed_refs(refs, normalize=True)
        for j, (idx, _) in enumerate(self._pending_embed):
            self._idx_map.append(idx)
            self._sub_feats.append(feats[j : j + 1])
        self._pending_embed = []

    def _sub_to_full(self, i_sub: int, right_open: bool = False) -> int:
        if i_sub < len(self._idx_map):
            return int(self._idx_map[i_sub])
        return len(self._times) if right_open else int(self._idx_map[-1])

    # ------------------------------------------------------------------
    def _recompute_and_finalize(self, force_flush: bool = False) -> None:
        self._flush_pending_embeds()
        self._sync_window()
        if len(self._idx_map) <= 1:
            return

        feats_sub = torch.cat(self._sub_feats, dim=0)
        S = feats_sub @ feats_sub.T
        prefix = build_prefix_from_sim(S)

        chunks_sub = self._build_visible_chunks()

        # chunk-centered expansion (bounds = neighbor chunks; only grow)
        small_subsegs: list[tuple[int, int, dict]] = []
        for i, ch in enumerate(chunks_sub):
            s0, e0 = ch["s_sub"], ch["e_sub"]
            left_bound = chunks_sub[i - 1]["e_sub"] if i - 1 >= 0 else 0
            right_bound = (
                chunks_sub[i + 1]["s_sub"] if i + 1 < len(chunks_sub) else len(self._idx_map)
            )
            ss, ee = expand_chunk_by_similarity(
                prefix,
                s0,
                e0,
                left_bound,
                right_bound,
                min_len=self.extend_min_sub,
                max_len=self.extend_max_sub,
            )
            small_subsegs.append((ss, ee, {"type": "chunk"}))

        # gaps -> small segments
        used = sorted((s, e) for (s, e, _) in small_subsegs)
        gaps, cur = [], 0
        for s, e in used:
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < len(self._idx_map):
            gaps.append((cur, len(self._idx_map)))
        for gs, ge in gaps:
            if ge - gs <= self.min_seg_sub:
                if ge - gs > 0:
                    small_subsegs.append((gs, ge, {"type": "gap"}))
            else:
                for s, e in cut_gap_region_by_similarity(
                    prefix,
                    gs,
                    ge,
                    T_sub=self.T_sub,
                    min_seg_sub=self.min_seg_sub,
                    subsample=self.cp_subsample,
                    allow_last_short=self.allow_last_short,
                ):
                    small_subsegs.append((s, e, {"type": "gap"}))

        # merge truly overlapping small segments (adjacent NOT merged)
        small_subsegs.sort(key=lambda x: (x[0], x[1]))
        compact: list[list] = []
        for s, e, meta in small_subsegs:
            if not compact:
                compact.append([s, e, meta])
            else:
                _ps, pe, pm = compact[-1]
                if s < pe:
                    compact[-1][1] = max(pe, e)
                    if pm.get("type") != "chunk" and meta.get("type") == "chunk":
                        compact[-1][2] = meta
                else:
                    compact.append([s, e, meta])
        small_subsegs = [(s, e, m) for (s, e, m) in compact]

        # representatives + back to full-frame domain
        small_segments, small_rep_feats = [], []
        for s_sub, e_sub, meta in small_subsegs:
            rep_sub = pick_representative_from_feats(feats_sub, s_sub, e_sub)
            small_rep_feats.append(feats_sub[rep_sub].clone())
            small_segments.append(
                {
                    "start": self._sub_to_full(s_sub),
                    "end": self._sub_to_full(e_sub, right_open=True),
                    **meta,
                }
            )
        if not small_segments:
            return
        rep_feats = torch.stack(small_rep_feats, dim=0)

        # pack into ~T_frames big segments
        big_segments, groups, acc_len, cur_group = [], [], 0, []
        for i, seg in enumerate(small_segments):
            acc_len += seg["end"] - seg["start"]
            cur_group.append(i)
            if acc_len >= self.T_frames:
                big_segments.append(
                    {
                        "start": small_segments[cur_group[0]]["start"],
                        "end": small_segments[cur_group[-1]]["end"],
                    }
                )
                groups.append(list(cur_group))
                acc_len, cur_group = 0, []
        if cur_group:
            big_segments.append(
                {
                    "start": small_segments[cur_group[0]]["start"],
                    "end": small_segments[cur_group[-1]]["end"],
                }
            )
            groups.append(list(cur_group))

        # big-segment representative features (as original)
        big_reps = []
        for g in groups:
            if len(g) == 1:
                big_reps.append(rep_feats[g[0]])
            else:
                rep = F.normalize(rep_feats[g].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                big_reps.append(rep)

        # finalize left of the watermark
        if force_flush:
            cutoff = len(self._times)
        else:
            safety = round(self.seg_extend_max_sec * self.fps)
            cutoff = max(len(self._times) - self.window_size_frames, len(self._times) - safety)

        for seg, rep in zip(big_segments, big_reps, strict=False):
            s, e = int(seg["start"]), int(seg["end"])
            if e <= cutoff and e > self._emitted_upto:
                gid = self._assign_global_id(rep)
                self._emit(s, e, gid)
                self._emitted_upto = e

    # ------------------------------------------------------------------
    def _build_visible_chunks(self) -> list[dict]:
        """Verbatim _build_visible_chunks: every visible utterance -> one raw
        chunk [min ts, max ts], clipped to the window, mapped to sub domain."""
        if not self._utter_items or not self._times:
            return []
        step2first: dict[int, int] = {}
        for i, t in enumerate(self._times):
            st = round(float(t) * self.fps)
            if st not in step2first:
                step2first[st] = i

        win_left = max(0, len(self._times) - self.window_size_frames)
        win_right = len(self._times)

        chunks_full = []
        for it in self._utter_items:
            ts = sorted(float(x) for x in (it.get(K.TEXT_TIMESTAMPS) or []))
            if not ts:
                continue
            idxs = [
                step2first[round(t * self.fps)] for t in ts if round(t * self.fps) in step2first
            ]
            if not idxs:
                continue
            s_full, e_full = int(min(idxs)), int(max(idxs) + 1)
            s_full, e_full = max(s_full, win_left), min(e_full, win_right)
            if e_full - s_full > 0:
                chunks_full.append((s_full, e_full))

        chunks_full.sort()
        idx_map_arr = np.asarray(self._idx_map)
        chunks_sub = []
        for s_full, e_full in chunks_full:
            s_sub = max(0, int(np.searchsorted(idx_map_arr, s_full, side="right") - 1))
            e_sub = int(np.searchsorted(idx_map_arr, e_full, side="left"))
            e_sub = max(e_sub, s_sub + 1)
            chunks_sub.append({"s_sub": s_sub, "e_sub": e_sub})
        return chunks_sub

    def _assign_global_id(self, rep_feat: torch.Tensor) -> int:
        """Verbatim _assign_global_id: adjacent-merge by cosine."""
        f = F.normalize(rep_feat.view(1, -1), dim=1).squeeze(0)
        if self._last_rep is None:
            gid = max(0, self._last_gid + 1) if self._last_gid < 0 else self._last_gid
        else:
            sim = float(torch.dot(f, self._last_rep).item())
            gid = self._last_gid if sim >= self.global_merge_cos else self._last_gid + 1
        self._last_rep = f
        self._last_gid = gid
        return int(gid)

    def _emit(self, i0: int, i1: int, gid: int) -> None:
        video_dir = self._refs[i0].locator[0]
        ref = ItemRef(self._sid, "video", self._times[i0], self._times[i1 - 1], (video_dir, i0, i1))
        if self.label_store is not None:
            self.label_store.assign([ref.key], [gid])
        self._out.append(
            make_event(
                "VisualSegment",
                **{
                    K.TIME_SESSION: self._sid,
                    K.TIME_T0: self._times[i0],
                    K.TIME_T1: self._times[i1 - 1],
                    K.VIDEO_REF: ref,
                },
            )
        )
