"""Data: where observations come from and how they become training batches.

Submodules:
  * sources   -- dataset adapters (saycam, babyview, video, toy): sessions of
                 time-aligned channels
  * decoding  -- video decode backends (frame_dir, poodle stub): file+time ->
                 pixels, sequential and random access
  * loading   -- Loader: refs -> tensors, as late as possible
  * views     -- named augmentation pipelines
  * text      -- tokenizer
  * batching  -- batch recipes (slots) + the composer
"""

from . import decoding  # noqa: F401  (registers decoder backends)
from .batching import Composer, Recipe, SlotSpec
from .loading import Loader
from .sources import DenseFrameChannel, Observation, Session, SparseIntervalChannel, Stream
from .text import VocabTokenizer, normalize_utterance
from .views import ViewBank

__all__ = [
    "Composer",
    "DenseFrameChannel",
    "Loader",
    "Observation",
    "Recipe",
    "Session",
    "SlotSpec",
    "SparseIntervalChannel",
    "Stream",
    "ViewBank",
    "VocabTokenizer",
    "normalize_utterance",
]
