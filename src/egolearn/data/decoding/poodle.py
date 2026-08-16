"""PoodleDecoder: integration slot for the lab's efficient video pipeline.

STATUS: interface stub -- awaiting access to the Poodle/Midway codebase
(2026-08 working-group decision: reuse its efficient random-access video
loading instead of writing our own decoder).

What we need from the Poodle side, mapped onto the VideoDecoder protocol:

  * probe        -> container metadata without a decode pass (fps, #frames);
  * read_window  -> its batched random access: seek to nearest keyframe,
                    decode forward once per window, return the requested
                    subset.  This serves REPLAY (memory hands the loader
                    refs into old videos);
  * iter_frames  -> its sequential reader with an open handle + prefetch
                    buffer.  This serves the STREAMING pass and Poodle's
                    own shuffled-IID training mode;
  * handle cache -> Poodle keeps per-file decode state; expose a close()
                    to bound open handles.

Known wiring gap the integration must close: VideoStream currently serves
every `open_frame` through `read_frame` (random access) even during the
sequential streaming pass -- harmless for frame_dir, wasteful for real
video (a seek per frame).  When this backend lands, either give it a
sequential read-ahead cache behind `read_frame` (detect monotonically
increasing indices per file and keep the decode handle rolling), or teach
the stream side to prefetch via `iter_frames`.  The first option needs no
framework change.

Config usage (once implemented):

    stream:
      _target_: "stream:video"
      root_dir: /path/to/videos
      decoder: {_target_: "decoder:poodle", cache_handles: 8}

Nothing else in the framework changes: refs stay (session, channel, t0, t1,
locator); only this backend turns them into pixels.
"""

from __future__ import annotations

from egolearn.config.registry import DECODERS

from .base import VideoDecoder


@DECODERS.register("poodle")
class PoodleDecoder(VideoDecoder):
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "decoder:poodle is an interface stub -- integration is blocked on "
            "access to the Poodle/Midway video pipeline. Use decoder:frame_dir "
            "for pre-extracted frames in the meantime. See this module's "
            "docstring for the planned mapping."
        )
