"""BabyCL project components -- the pieces that are specific to the BabyCL method.
Importing this module registers:

admission : babycl_online_segmenter (1:1 port of the original OnlineSegmenter),
            an_chunk_emitter (1:1 port of the feeder's AN-chunk path)
embedder  : frozen_resnext50
encoder   : text_bow
view      : cvcl_aug

Everything else BabyCL uses comes from the shared core toolbox (saycam/babyview
adapters, ntxent/tcl/clip_infonce losses, simclr views, resnext50, Table policies).
"""

from . import an_chunks, data, model, refine, segmenter  # noqa: F401
