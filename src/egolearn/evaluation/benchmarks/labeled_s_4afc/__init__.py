"""Labeled-S 4AFC benchmark: the canonical SAYCam evaluation (Vong et al.).

The dataset side of the 4AFC readout (the scoring lives in
evaluation/four_afc.py, shared with babyview_4afc): 4,150 frozen trials over
the 23 Labeled-S categories, built from BabyCL's eval_object_categories_4150
protocol.  The eval images ship pre-squared at 224x224, so the `view`
choice (center_crop vs resize) is vacuous here -- unlike BabyView.

Data sources, manifest conversion, and model vocabulary:
docs/evaluation/data.md#labeled-s-4afc-data

Config::

    _imports_: [egolearn.evaluation.benchmarks.labeled_s_4afc]
    probes:
      - {_target_: "probe:four_afc", name: labeled_s_4afc,
         manifest: <labeled_s_4afc_manifest.json>, vocab_json: <vocab.json>,
         vision_head: embed512, size: 224}
"""

from egolearn.evaluation.four_afc import FourAFCProbe  # noqa: F401  (registers probe:four_afc)

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "Labeled-S 4AFC",
    "source": "[SAYCam / Labeled-S](https://nyu.databrary.org/volume/564/)",
    "license": "Databrary terms; frozen trial metadata obtained separately",
    "access": "IRB",
    "prepare": (
        '`egl data prepare labeled-s-4afc --data_root "$DATA_ROOT" --trials_json '
        '"$TRIALS_JSON" --vocab_json "$VOCAB_JSON" --output "$MANIFEST"`'
    ),
}
