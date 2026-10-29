from . import poodle  # noqa: F401  (registers the stub so the error is helpful)
from .base import VideoDecoder
from .frame_dir import FrameDirDecoder
from .pyav import PyAVDecoder

__all__ = ["FrameDirDecoder", "PyAVDecoder", "VideoDecoder"]
