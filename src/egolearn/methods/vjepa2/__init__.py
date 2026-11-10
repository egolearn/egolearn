"""V-JEPA 2 pretraining components; import through Hydra's `_imports_`."""

from egolearn.methods.vjepa2.admission import DenseClipEmitter
from egolearn.methods.vjepa2.loss import VJEPA2
from egolearn.methods.vjepa2.model import VJEPA2Encoder, VJEPA2Predictor

__all__ = ["VJEPA2", "DenseClipEmitter", "VJEPA2Encoder", "VJEPA2Predictor"]
