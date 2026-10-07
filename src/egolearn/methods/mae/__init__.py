"""Standard image MAE components, registered by importing this package.

An independent implementation of He et al., "Masked Autoencoders Are Scalable
Vision Learners" (https://arxiv.org/abs/2111.06377), using PyTorch primitives.
No implementation code is vendored from the upstream MAE repository.
"""

from egolearn.methods.mae.loss import MAE
from egolearn.methods.mae.model import MAEDecoder, MAEEncoder
from egolearn.methods.mae.views import mae

__all__ = ["MAE", "MAEDecoder", "MAEEncoder", "mae"]
