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

``probe:labeled_s_stratified_linear`` instead reproduces the BabyCL paper's
probe: every image, a seeded per-category 80/20 split, unit-length features
and scikit-learn's ``C=0.316`` penalty (see ``stratified.py``)::

      - {_target_: "probe:labeled_s_stratified_linear", name: labeled_s_linear_paper,
         data_root: <labeled-s-root>, head: embed512}
"""

from .probe import LabeledSLinearProbe
from .stratified import LabeledSStratifiedLinearProbe

__all__ = ["LabeledSLinearProbe", "LabeledSStratifiedLinearProbe"]

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "Labeled-S linear",
    "source": (
        "[Labeled-S images](https://nyu.databrary.org/volume/564/), [canonical "
        "split](https://github.com/agentic-learning-ai-lab/memory-storyboard/blob/c084d826fdc85a72e78477bb53c88b7353a89563/metas/labeledS_train_val.txt)"
    ),
    "license": "Images: Databrary terms; split metadata: see source",
    "access": "IRB",
    "prepare": (
        "No preparer; supply category folders as `data_root`; the canonical adapter "
        "verifies split metadata, and the stratified adapter splits in memory."
    ),
}
