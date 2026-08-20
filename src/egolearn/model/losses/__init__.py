"""Standard loss library (losses are a submodule of the model: they are the
model's training signals, addressed through its encoders and heads)."""

from .base import Loss
from .clip_infonce import ClipInfoNCE
from .ntxent import NTXent
from .tcl import TemporalClassification

__all__ = ["ClipInfoNCE", "Loss", "NTXent", "TemporalClassification"]
