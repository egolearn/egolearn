"""Internal plumbing shared by every module: sample-record keys, time-interval
references, event schemas, and the run context (clocks + seeded RNGs).

This package is NOT a configurable module -- nothing here appears in a config
file.  It defines what flows between the five top-level modules (data, model,
memory, training, evaluation).
"""

from . import events, keys, refs
from .context import Clock, Context
from .refs import ItemRef

__all__ = ["Clock", "Context", "ItemRef", "events", "keys", "refs"]
