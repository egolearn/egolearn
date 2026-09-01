"""Novel-object shape-bias benchmark from Tartaglini et al. (2022), Exp. 3.

The selected condition crosses 16 novel masks with 16 Brodatz textures,
renders each 180x180 object at a seeded random position on a white 224x224
canvas, and encodes the resulting 256 images once. For every anchor, all 15
same-shape matches are compared with all 15 same-texture matches by cosine
similarity: 57,600 decisions total. Ties count as texture decisions, matching
the released evaluation.

Prepare the checksum-pinned stimuli::

    uv run --extra train python scripts/prepare_shape_bias.py

Config::

    _imports_: [egolearn.evaluation.benchmarks.shape_bias]
    probes:
      - {_target_: "probe:shape_bias", name: shape_bias,
         manifest: data/shape_bias/80_unaligned/stimulus_manifest.json,
         batch_size: 32, view: resize, size: 224,
         output_json: "runs/shape_bias_step_{step}.json"}
"""

from .probe import ShapeBiasProbe
from .score import ShapeBiasResult, score_shape_bias
from .stimuli import (
    download_shape_bias_assets,
    prepare_shape_bias_stimuli,
    validate_stimulus_manifest,
)

__all__ = [
    "ShapeBiasProbe",
    "ShapeBiasResult",
    "download_shape_bias_assets",
    "prepare_shape_bias_stimuli",
    "score_shape_bias",
    "validate_stimulus_manifest",
]
