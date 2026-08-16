"""BabyCL project components (the method-novel machinery). Importing this
module registers:

admission : babycl_online_segmenter (1:1 OnlineSegmenter port),
            an_chunk_emitter (1:1 feeder AN-chunk port)
embedder  : frozen_resnext50
trunk     : text_bow
view      : cvcl_aug

Everything else BabyCL uses is the core toolbox (saycam/babyview adapters,
ntxent/tcl/clip_infonce, simclr views, resnext50, Table policies).
"""

from . import an_chunks, data, model, refine, segmenter  # noqa: F401
