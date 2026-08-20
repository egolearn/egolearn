"""Model: encoders, heads, predictors and losses, composed by config.

Submodules:
  * model      -- ComposedModel: named encoders + heads + predictors behind
                  one surface; capabilities; the export contract
  * encoders   -- image (resnext50, resnet18, toy_cnn) and video (toy_video)
                  encoders; base.py documents the encoder contract
  * heads      -- projection / classification heads (linear, mlp)
  * predictors -- representation -> predicted representation (JEPA-style)
  * generative -- reserved interface for generative/autoregressive parts
  * losses     -- the standard loss library (ntxent, tcl, clip_infonce)
"""

from . import encoders, generative, predictors  # noqa: F401  (registers components)
from .encoders.base import EncoderOutput
from .heads import LinearHead, MLPHead
from .losses import Loss
from .model import ComposedModel

__all__ = [
    "ComposedModel",
    "EncoderOutput",
    "LinearHead",
    "Loss",
    "MLPHead",
]
