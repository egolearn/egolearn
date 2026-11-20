"""Konkle objects 4AFC: out-of-corpus word-object matching on studio photographs.

Sixty object categories with 17 photographs each (1,020 images) from the
Konkle et al. (2010) object set, in the test split of Vong et al. (2024).
The protocol follows "Scaling cross-situational word learning on children's
naturalistic experiences" (arXiv:2610.31842) as documented in EVAL.md of
https://github.com/mcfrank/vlm-headcam:

* the word is the category label; the model scores it against four images,
  one of the target category and one from each of three other categories;
* 100 trials per category, drawn with ``numpy.random.default_rng(0)`` while
  visiting categories in sorted order;
* accuracy is the mean over all 60 categories of per-category accuracy;
* a category whose label is outside the model vocabulary scores chance (0.25)
  and is left out of the foil pool, so trial draws match across models only
  when their vocabularies cover the same categories.

The preparer draws the trials for one vocabulary and writes them in
probe:four_afc's manifest format; out-of-vocabulary categories are recorded as
``chance_trials``. Scoring reuses probe:four_afc unchanged.

The reference implementation scores a word against 17 region vectors per image
and takes the maximum. probe:four_afc scores the cosine similarity between the
text representation and one pooled image representation.

Config::

    _imports_: [egolearn.evaluation.benchmarks.konkle_4afc]
    benchmarks:
      - {_target_: "probe:four_afc", name: konkle_4afc,
         manifest: <konkle_4afc_trials.json>, vocab_json: <vocab.json>,
         vision_head: embed512, view: resize, size: 224}
"""

from egolearn.evaluation.four_afc import FourAFCProbe  # noqa: F401  (registers probe:four_afc)

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "Konkle objects 4AFC",
    "source": "[Konkle et al. object categories](https://konklab.fas.harvard.edu/)",
    "license": "Images distributed by the Konkle lab for research use; obtained separately",
    "access": "upstream-licensed",
    "prepare": (
        '`egl data prepare konkle-4afc --data_root "$DATA_ROOT" '
        '--vocab_json "$VOCAB_JSON" --output "$MANIFEST"`'
    ),
}
