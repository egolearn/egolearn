"""Fixed-feature linear-decoding benchmark on the canonical Labeled-S split.

The first 2,886 rows of Memory Storyboard's pinned metadata train the
classifier and the remaining 2,901 rows evaluate it, covering all 26
categories.  ``probe:labeled_s_linear`` fits fixed features with full-batch
L-BFGS.

The adapter verifies the split metadata, resolves its paths in memory against
a caller-provided dataset root, and creates no machine-specific manifest.  Pass
``split_file:`` to use a local copy on an offline cluster.

Config::

    _imports_: [egolearn.evaluation.benchmarks.labeled_s_linear]
    probes:
      - {_target_: "probe:labeled_s_linear", name: labeled_s_linear,
         data_root: <labeled-s-root>, view: center_crop, size: 224}
"""

from .probe import LabeledSLinearProbe

__all__ = ["LabeledSLinearProbe"]
