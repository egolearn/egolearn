"""Memory: what the learner keeps from the stream and how it is reused.

Submodules:
  * table      -- Table / TieredMemory: bounded buffers of event items
  * update     -- buffer update policies (fifo, reservoir, ...) + samplers
  * labels     -- LabelStore: revisable labels behind an indirection layer
  * admission  -- EventBuilders: what enters memory (frames, segments, pairs)
"""

from .admission import EmbeddingProvider, EventBuilder
from .labels import LabelStore
from .table import Table, TieredMemory
from .update import Sampler, UpdatePolicy

__all__ = [
    "EmbeddingProvider",
    "EventBuilder",
    "LabelStore",
    "Sampler",
    "Table",
    "TieredMemory",
    "UpdatePolicy",
]
