"""Base interfaces for data sources.  A Stream (a dataset adapter) yields
Sessions, and each Session holds time-aligned channels of observations.

A Session is one contiguous recording (one SAYCam video, one Ego4D clip).
It exposes named channels (time-sorted sequences of observations) of two
kinds:

  * dense channels  -- regularly sampled observations (video frames);
  * sparse channels -- events that cover a time interval and carry a payload
    (utterances, narrations, action annotations).  Both kinds share the same
    structure, which is why BabyCL's utterance machinery transfers to Ego4D
    narrations without code changes.

Alignment is ALWAYS by media time (seconds into the recording), never by
frame index.

The Stream is also the place where images are actually read from storage:
`open_frame(ref, index)` is the only spot where a ref (a lightweight pointer
to a frame or interval on disk) is turned into pixels.  Everything before
the Loader passes around refs only; images are decoded only when a batch is
built.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from egolearn.core.refs import ItemRef


@dataclass(frozen=True)
class Observation:
    t: float  # media time (seconds) of this observation
    session: str
    channel: str
    ref: ItemRef  # point ref for dense, interval ref for sparse
    payload: dict = field(default_factory=dict)  # only key names from egolearn.core.keys


class Channel:
    """Time-sorted source of observations for one modality."""

    name: str
    kind: str  # "dense" | "sparse"

    def __iter__(self) -> Iterator[Observation]:  # pragma: no cover - interface
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class DenseFrameChannel(Channel):
    """Frames of one session, sorted by time.

    A locator is whatever tuple the owning adapter needs to find the frame
    on disk later; this class just stores and passes them through.
    """

    kind = "dense"

    def __init__(self, name: str, session: str, times: list[float], locators: list[tuple]):
        assert len(times) == len(locators)
        self.name = name
        self.session = session
        self.times = times
        self.locators = locators

    def __len__(self):
        return len(self.times)

    def ref_at(self, index: int) -> ItemRef:
        t = self.times[index]
        return ItemRef(self.session, self.name, t, t, self.locators[index])

    def interval_ref(self, i0: int, i1: int) -> ItemRef:
        """Ref covering frame indices [i0, i1)."""
        return ItemRef(
            self.session,
            self.name,
            self.times[i0],
            self.times[max(i0, i1 - 1)],
            (self.locators[i0][0], i0, i1),
        )

    def __iter__(self):
        for i in range(len(self.times)):
            yield Observation(self.times[i], self.session, self.name, self.ref_at(i))


class SparseIntervalChannel(Channel):
    """Interval events (utterances etc.), sorted by end time.

    An interval observation is surfaced at its END time: an utterance is only
    made visible once it has finished, so a streaming learner never acts on
    speech it has not fully heard.  (This rule is inherited from BabyCL.)
    """

    kind = "sparse"

    def __init__(self, name: str, session: str, items: list[dict]):
        # each item: {"t0","t1","payload": {registered keys}}
        self.name = name
        self.session = session
        self.items = sorted(items, key=lambda it: (it["t1"], it["t0"]))

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i, it in enumerate(self.items):
            ref = ItemRef(self.session, self.name, it["t0"], it["t1"], ("interval", i))
            yield Observation(it["t1"], self.session, self.name, ref, dict(it["payload"]))


class Session:
    def __init__(self, sid: str, channels: dict[str, Channel]):
        self.sid = sid
        self.channels = channels

    def observations(self) -> Iterator[Observation]:
        """Merge all channels into one time-ordered iterator (heap merge).
        Observations at the same media time surface in channel declaration
        order (a frame before the utterance ending at that instant).  The
        helper binds `order` eagerly -- a generator expression would read
        the loop variable lazily and give every channel the same rank."""

        def keyed(order: int, ch: Channel):
            for i, obs in enumerate(iter(ch)):
                yield (obs.t, order, i), obs

        iters = [keyed(order, ch) for order, ch in enumerate(self.channels.values())]
        return (obs for _, obs in heapq.merge(*iters, key=lambda pair: pair[0]))

    def __len__(self):
        return sum(len(c) for c in self.channels.values())


class Stream:
    """Interface every dataset adapter implements.

    Subclasses: SaycamStream, BabyViewStream, ToyStream.
    """

    def sessions(self) -> Iterable[Session]:  # pragma: no cover - interface
        raise NotImplementedError

    def total_observations(self) -> int | None:
        """Optional total count of observations across all sessions.  When an
        adapter provides it, the progress clock and progress-based schedules
        become available; returning None just disables those."""
        return None

    # ---- reading pixels (the only place refs become images) ----------------
    def open_frame(self, ref: ItemRef, index: int):
        """Return a PIL image for frame `index` within `ref`'s interval.

        `index` counts frames inside the ref's locator range; for a
        single-frame (point) ref, index must be 0.
        """
        raise NotImplementedError  # pragma: no cover - interface

    def num_frames(self, ref: ItemRef) -> int:
        raise NotImplementedError  # pragma: no cover - interface
