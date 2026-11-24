"""Admission: the components that decide what enters memory.

An EventBuilder watches the raw observation stream and decides which finished
units ("events": single frames, segments, image-text pairs) are offered to
memory.  In an offline pipeline this job is done by dataset preprocessing;
here it happens online, while the stream plays.

Rules (code-verified against both published systems):

  * Events are emitted exactly once and never change afterwards.  A builder
    may only emit an event once it can never change.  BabyCL enforces this
    with a watermark: a segment is only emitted once the stream is far enough
    past its end that nothing can change it.
  * Labels attached to events MAY be revised later -- but only through the
    LabelStore (egolearn.memory.labels): items store a key instead of a
    label, and the label is looked up when a batch is built, so it can still
    be corrected.  Stored items themselves are never mutated.

EventBuilders come in two roles (one class may play both):
  * emitters -- produce events offered to memory, returned from poll();
  * labelers -- produce no events; they only write labels to a LabelStore.
"""

from __future__ import annotations

import torch

from egolearn.core.context import Context
from egolearn.core.refs import ItemRef
from egolearn.data.loading import Loader
from egolearn.data.sources.base import Observation, Session
from egolearn.memory.labels import LabelStore


class EmbeddingProvider:
    """Turns a list of frame refs into embeddings [N, D].  Builders that
    cluster or segment frames use this to compare them.

    It is pluggable because the two published systems differ here (both
    code-verified):
      * Memory Storyboard clusters with the LEARNER'S OWN backbone
        (`embedder: learner`);
      * BabyCL's online segmenter uses a FROZEN ImageNet encoder
        (`embedder: frozen_resnext50`).
    """

    def bind(self, ctx: Context, loader: Loader, model=None) -> None:
        self.ctx = ctx
        self.loader = loader
        self.model = model

    @torch.no_grad()
    def embed_refs(
        self, refs: list[ItemRef], normalize: bool = True
    ) -> torch.Tensor:  # pragma: no cover
        """Return [N, D] embeddings.  `normalize=False` returns raw features:
        Memory Storyboard's prototype merge averages RAW embeddings and only
        normalizes the mean, while its recluster normalizes each frame."""
        raise NotImplementedError


class EventBuilder:
    """Stateful component that watches the stream and builds events and/or
    labels from it.  See the module docstring for the rules it must follow."""

    subscribes: tuple[str, ...] = ("video",)  # channels this builder listens to
    emits: tuple[str, ...] = ()  # names of the event schemas it produces

    def bind(
        self,
        ctx: Context,
        loader: Loader | None = None,
        embedder: EmbeddingProvider | None = None,
        label_store: LabelStore | None = None,
        memory=None,
    ) -> None:
        """`memory` (a TieredMemory) is only bound when the builder's config
        names one.  Memory Storyboard needs it: its class-merge step samples
        prototype frames from the LONG-TERM buffer, so faithful reproduction
        requires the builder to read that buffer."""
        self.ctx = ctx
        self.loader = loader
        self.embedder = embedder
        self.label_store = label_store
        self.memory = memory

    def start_session(self, sid: str) -> None:
        """Called when a new recording session starts.  State that carries
        over between sessions (e.g. keeping label ids continuous) is handled
        here."""

    def begin_session(self, session: Session) -> None:
        """Receive channel metadata; existing builders keep their sid-only hook."""
        self.start_session(session.sid)

    def ingest(self, obs: Observation) -> None:
        raise NotImplementedError  # pragma: no cover

    def poll(self) -> list[dict]:
        """Return events finalized since the last call (possibly none).  Only
        finished events -- never ones that might still change."""
        return []

    def end_session(self) -> None:
        """Called when the session ends: everything still pending can no
        longer change, so it should be finalized and flushed."""

    def set_replaying(self, replaying: bool) -> None:
        """Optional notification while rebuilding admission state on resume.

        Builders may defer expensive derived metadata only if it is restored
        from memory snapshots and their subsequent live state remains exact.
        """
