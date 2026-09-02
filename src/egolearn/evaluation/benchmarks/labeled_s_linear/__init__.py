"""Linear-decoding benchmarks on the canonical Labeled-S split.

The first 2,886 rows of Memory Storyboard's pinned metadata train the
classifier and the remaining 2,901 rows evaluate it, covering all 26
categories.  ``probe:labeled_s_linear`` fits fixed features with full-batch
L-BFGS; ``probe:labeled_s_linear_sgd`` uses the augmented, gradient-trained
decoder.

Both adapters verify the split metadata, resolve its paths in memory against
a caller-provided dataset root, and create no machine-specific manifest.  Pass
``split_file:`` to use a local copy on an offline cluster.

Config::

    _imports_: [egolearn.evaluation.benchmarks.labeled_s_linear]
    probes:
      - {_target_: "probe:labeled_s_linear", name: labeled_s_linear,
         data_root: <labeled-s-root>, view: center_crop, size: 224}
      - {_target_: "probe:labeled_s_linear_sgd", name: labeled_s_linear_sgd,
         data_root: <labeled-s-root>, encoder: vision,
         size: 112, batch_size: 1024, workers: 16, epochs: 10}
"""

from .probe import LabeledSGradientLinearProbe, LabeledSLinearProbe

__all__ = ["LabeledSGradientLinearProbe", "LabeledSLinearProbe"]
