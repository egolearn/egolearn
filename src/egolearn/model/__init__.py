"""Model: encoders (trunks with named taps), heads, and losses.

Submodules:
  * model   -- EgoModel: trunk taps + heads; the export contract
  * trunks  -- standard architectures (resnext50, resnet18, toy_cnn)
  * heads   -- projection / classification heads (linear, mlp)
  * losses  -- the standard loss library (ntxent, tcl, clip_infonce)
"""

from .heads import LinearHead, MLPHead
from .losses import Loss
from .model import EgoModel
from .trunks import ResNet18Trunk, ResNeXt50Trunk, ToyCNNTrunk

__all__ = [
    "EgoModel",
    "LinearHead",
    "Loss",
    "MLPHead",
    "ResNeXt50Trunk",
    "ResNet18Trunk",
    "ToyCNNTrunk",
]
