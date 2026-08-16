"""Port of BabyCL's AN-chunk admission path (mcl ms_dataloader OnlineFeeder:
_prepare_chunks_for_video / _emit_completed_chunks).

Code-verified behavior being reproduced -- note this is NOT the segmenter's
similarity-expanded chunks (the feeder discards those):

  * each utterance's raw timestamps are mapped to 5fps steps; steps missing
    a frame are dropped; STRICTLY CONTIGUOUS step runs (diff == 1) become
    separate chunks -- one utterance can yield several chunks;
  * a chunk covers frame indices [step2first[a], step2first[b] + 1);
  * a chunk is emitted ATOMICALLY only once the stream has advanced past its
    end index (complete visibility), deduplicated by
    (utterance_num, start, end, session);
  * clip_importance (weight.sample) and the utterance text flow through
    untouched.
"""

from __future__ import annotations

from egolearn.config.registry import ADMISSION
from egolearn.core import keys as K
from egolearn.core.events import make_event
from egolearn.core.refs import ItemRef
from egolearn.data.sources.base import Observation
from egolearn.memory.admission.base import EventBuilder


@ADMISSION.register("an_chunk_emitter")
class ANChunkEmitter(EventBuilder):
    subscribes = ("video", "transcript")
    emits = ("AlignedPair",)

    def __init__(self, fps: float = 5.0):
        self.fps = float(fps)
        # feeder-lifetime dedup (all sessions): (utt_num, start, end, sid)
        self._seen_keys: set[tuple[int, int, int, str]] = set()
        self._out: list[dict] = []
        self._reset("")

    def _reset(self, sid: str) -> None:
        self._sid = sid
        self._times: list[float] = []
        self._refs: list[ItemRef] = []
        self._step2first: dict[int, int] = {}
        self._pending: list[dict] = []  # chunks awaiting complete visibility

    def start_session(self, sid: str) -> None:
        self._reset(sid)

    def ingest(self, obs: Observation) -> None:
        if obs.session != self._sid:
            return
        if obs.channel == "video":
            idx = len(self._times)
            st = round(float(obs.t) * self.fps)
            if st not in self._step2first:
                self._step2first[st] = idx
            self._times.append(obs.t)
            self._refs.append(obs.ref)
            self._drain()
        elif obs.channel == "transcript":
            self._prepare_chunks(obs.payload)
            self._drain()

    def end_session(self) -> None:
        # all frames ingested -> every well-formed chunk is complete
        self._drain()

    def poll(self) -> list[dict]:
        out, self._out = self._out, []
        return out

    # ------------------------------------------------------------------
    def _prepare_chunks(self, payload: dict) -> None:
        """Verbatim run-splitting from _prepare_chunks_for_video."""
        raw_ts = payload.get(K.TEXT_TIMESTAMPS) or []
        steps = sorted({round(float(t) * self.fps) for t in raw_ts})
        steps = [st for st in steps if st in self._step2first]
        if not steps:
            return

        runs = []
        run_start = prev = steps[0]
        for st in steps[1:]:
            if st == prev + 1:
                prev = st
            else:
                runs.append((run_start, prev))
                run_start = prev = st
        runs.append((run_start, prev))

        for a, b in runs:
            start_idx = self._step2first[a]
            end_idx = self._step2first[b] + 1
            if end_idx <= start_idx:
                continue
            self._pending.append(
                {
                    "start": start_idx,
                    "end": end_idx,
                    "payload": {k: v for k, v in payload.items() if k != K.TEXT_TIMESTAMPS},
                }
            )

    def _drain(self) -> None:
        """Emit chunks whose end index the stream has fully passed."""
        if not self._pending:
            return
        n = len(self._times)
        still = []
        for ch in self._pending:
            if n >= ch["end"]:
                self._emit(ch)
            else:
                still.append(ch)
        self._pending = still

    def _emit(self, ch: dict) -> None:
        utt_num = ch["payload"].get(K.UTTERANCE_NUM)
        key = (int(utt_num) if utt_num is not None else -1, ch["start"], ch["end"], self._sid)
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)
        i0, i1 = ch["start"], ch["end"]
        video_dir = self._refs[i0].locator[0]
        ref = ItemRef(self._sid, "video", self._times[i0], self._times[i1 - 1], (video_dir, i0, i1))
        fields = {
            K.TIME_SESSION: self._sid,
            K.TIME_T0: self._times[i0],
            K.TIME_T1: self._times[i1 - 1],
            K.VIDEO_REF: ref,
        }
        fields.update(ch["payload"])
        self._out.append(make_event("AlignedPair", **fields))
