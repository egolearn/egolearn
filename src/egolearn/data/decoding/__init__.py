from . import poodle  # noqa: F401  (registers the stub so the error is helpful)
from .base import VideoDecoder
from .frame_dir import FrameDirDecoder

__all__ = ["FrameDirDecoder", "VideoDecoder"]
