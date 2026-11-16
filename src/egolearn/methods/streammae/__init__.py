"""StreamMAE input pipeline; reuses standard MAE's model, decoder and loss."""

from egolearn.methods import mae as mae
from egolearn.methods.streammae.motion import MotionFrames
from egolearn.methods.streammae.views import StreamMAEView

__all__ = ["MotionFrames", "StreamMAEView"]
