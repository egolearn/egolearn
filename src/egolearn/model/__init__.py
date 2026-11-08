"""Model: encoders, heads, predictors and losses, composed by config.

Submodules:
  * model       -- ComposedModel: named encoders + heads + predictors behind
                   one surface; capabilities; the export contract
  * encoders    -- image (resnext50, resnet18, toy_cnn) and video (video_vit,
                   toy_video) encoders; base.py documents the encoder contract
  * heads       -- projection / classification heads (linear, mlp)
  * predictors  -- representation -> predicted representation (mlp, vjepa)
  * transformer -- ViT blocks + 3D rotary positions, shared by video_vit and
                   the vjepa predictor
  * masking     -- token-grid mask samplers for masked video prediction
  * generative  -- generative components (openai_chat); base.py documents the
                   contract and the Video input type
  * losses      -- the standard loss library (ntxent, tcl, clip_infonce, vjepa)
"""

from . import encoders, generative, predictors  # noqa: F401  (registers components)
from .build import build_model
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
    "build_model",
]
