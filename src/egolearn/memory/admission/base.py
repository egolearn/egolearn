"""Admission: what enters memory, in what unit, with what labels.

An EventBuilder watches the raw observation stream and decides which finalized
units ("events": frames, segments, aligned pairs) are offered to memory.  It
is the online replacement for offline dataset preprocessing.

Contract (code-verified against both published systems):

  * Events are emitted exactly once and never change afterwards.  A builder
    may only emit an event once it can never change (BabyCL's watermark:
    finalize behind a safety margin).
  * Labels attached to events MAY be revised later -- but only through the
    LabelStore indirection (egolearn.memory.labels), never by mutating
    stored items.

EventBuilders come in two roles (one class may play both):
  * emitters  -- produce events for memory admission (poll());
  * labelers  -- produce no events; they maintain a LabelStore.
"""

from __future__ import annotations

import torch

from egolearn.core.context import Context
from egolearn.core.refs import ItemRef
from egolearn.data.loading import Loader
from egolearn.data.sources.base import Observation
from egolearn.memory.labels import LabelStore


class EmbeddingProvider:
    """Frame refs -> embeddings [N, D] (used by segmentation-style builders).

    Two code-verified published variants (this is why it is pluggable):
      * Memory Storyboard clusters with the LEARNER'S OWN backbone
        (`embedder: learner_tap`);
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
        """[N, D] embeddings. `normalize=False` returns raw features --
        Memory Storyboard's prototype merge averages RAW embeddings before
        normalizing the mean, while its recluster normalizes per frame."""
        raise NotImplementedError


class EventBuilder:
    """Stateful online structure inducer. See module docstring for contract."""

    subscribes: tuple[str, ...] = ("video",)  # channel names it ingests
    emits: tuple[str, ...] = ()  # event schema names it produces

    def bind(
        self,
        ctx: Context,
        loader: Loader | None = None,
        embedder: EmbeddingProvider | None = None,
        label_store: LabelStore | None = None,
        memory=None,
    ) -> None:
        """`memory` (a TieredMemory) is bound when the builder's config names
        one -- required for fidelity to Memory Storyboard, whose class-merge
        step samples prototype frames from the LONG-TERM buffer."""
        self.ctx = ctx
        self.loader = loader
        self.embedder = embedder
        self.label_store = label_store
        self.memory = memory

    def start_session(self, sid: str) -> None:
        """Cross-session state (e.g. label continuity) is handed over here."""

    def ingest(self, obs: Observation) -> None:
        raise NotImplementedError  # pragma: no cover

    def poll(self) -> list[dict]:
        """Return newly FINALIZED events (possibly empty). Never provisional."""
        return []

    def end_session(self) -> None:
        """Flush: the session is over, everything left is final."""
