"""BabyView benchmarks: zero-shot 4AFC and a linear probe on the public
BabyView object crops.

The dataset (babyview-project/ccn-2026-object-variability,
`valid5921_public`): 5,921 rater-validated object crops from infant
egocentric video, 72 CDI noun categories, privacy-filtered.  Two tasks:

* **zero-shot 4AFC** -- the Labeled-S 4150-trial protocol transplanted onto
  BabyView crops: one category word vs four candidate images, cosine argmax,
  chance 25%.  Targets/foils come only from the 61 categories in the SAYCam
  vocabulary (the 11 OOV categories are excluded and listed in the report;
  no synonym renaming).  Scoring reuses probe:four_afc unchanged.
* **linear probe** -- multinomial logistic regression on frozen features
  over ALL 72 categories (a vision-only readout has no vocabulary
  constraint), frozen stratified split.  Scoring reuses probe:linear.

Eval preprocessing (frozen 2026-08-30): shortest-side resize + center crop
(`view: center_crop`).  Unlike Labeled-S -- whose eval images ship
pre-squared, making the choice vacuous -- these crops are arbitrary-aspect
object boxes (elongation median 1.34, p90 2.2).  A 3-models x 3-transforms
comparison in the source repo decided it: letterbox lost on every model,
center-crop beat squashing on the strongest and matches the standard
CLIP/DINO preprocessing behind the release's reference embeddings.

`python -m egolearn.evaluation.benchmarks.babyview_4afc.prepare --help`
downloads the crops and writes both manifests plus a provenance report; the
trial construction is a faithful port of the source generator (same RNG
call order, same seed), so the 4150 trials are identical to the source
repo's frozen json.

Config::

    _imports_: [egolearn.evaluation.benchmarks.babyview_4afc]
    probes:
      - {_target_: "probe:four_afc", name: babyview_4afc,
         manifest: <out>/babyview_4afc_trials.json,
         vocab_json: <saycam vocab.json>,
         vision_head: embed512, view: center_crop, size: 224}
      - {_target_: "probe:linear", name: babyview_linear,
         manifest: <out>/babyview_linear.json,
         view: center_crop, size: 224}
"""

from egolearn.evaluation.four_afc import FourAFCProbe  # noqa: F401  (registers probe:four_afc)
from egolearn.evaluation.linear_probe import LinearProbe  # noqa: F401  (registers probe:linear)
