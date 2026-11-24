# Dataset access and preparation

This page explains how to get access to the restricted SAYCam and BabyView
recordings, and how to prepare data for the benchmarks that have no page of
their own.

## SAYCam access

SAYCam videos, the SAYCam-S frames and the Labeled-S evaluation images are
distributed through [Databrary volume 564](https://nyu.databrary.org/volume/564/).
Access requires IRB approval at your institution and acceptance of Databrary's
data-use terms. The [CVCL dataset notes](https://github.com/wkvong/multimodal-baby#datasets)
identify the releases that EgoLearn uses.

Public labels for Labeled-S, VTWT and Baby Winoground do not grant access to the
images. Keep restricted images, and any manifest that points at them, out of
version control and off the Hub, as the [access tiers](hub.md#access-tiers)
require.

## BabyView training data

BabyView recordings are also shared through Databrary, and the
[BabyView access guide](https://babyview-project.github.io/access/) explains the
account, institutional approval and data-use terms you need before you download.
Weights trained on BabyView need a separate
[checkpoint access request](hub.md#checkpoint-access), while the BabyView
evaluation uses public [object crops](#babyview-object-crops).

## Training vocabulary

Image–text benchmarks look up each label word in the model's training
vocabulary, so pass the exact file the model was trained with as `vocab_json`.
BabyCL models were trained with the CVCL vocabulary file
[`multimodal/vocab.json`](https://github.com/wkvong/multimodal-baby/blob/d64c1bb164ed4c25f0cad8a1c3b7a4d68efae326/multimodal/vocab.json).
A vocabulary rebuilt from the evaluation labels assigns different word indices,
which the checkpoint cannot undo.

## Shape bias

The shape-bias benchmark uses 256 novel-object images that you render locally
from the masks and textures of
[Tartaglini, Vong and Lake (2022)](https://github.com/alexatartaglini/developmental-shape-bias):

```bash
uv run egl data prepare shape-bias
```

The command downloads the source assets at a pinned commit and writes the
images and `stimulus_manifest.json` to `data/shape-bias`, or under
`$EGOLEARN_DATA` when it is set. It refuses to overwrite an existing collection,
and `--root DIR` before `prepare` chooses another location. Pass the manifest to
`shape_bias_eval` as in the [evaluation example](../evaluation.md#example). A
local copy does not grant redistribution rights, so follow the upstream terms
and cite the paper.

The probe scores 57,600 triplets of an anchor, a shape match and a texture
match. `shape_bias` is the fraction in which the shape match is closer to the
anchor by cosine similarity, with ties counted as texture choices. The result
also reports the tie count and the mean similarity margin.

## Labeled-S 4AFC data

The Labeled-S 4AFC benchmark uses BabyCL's 4,150 frozen trials over 23
categories. You need three inputs:

- the Labeled-S images from [SAYCam access](#saycam-access), in their
  `<category>/<filename>` folders
- the trial file `eval_object_categories_4150.json`, released as supplementary
  material with the [BabyCL paper](https://arxiv.org/html/2606.05115v1) and in
  the gated [`labeled-s-splits`](hub.md#gated-collections) collection
- the model's [training vocabulary](#training-vocabulary)

New trials would make a different benchmark, so use the released file and
convert it to a manifest:

```bash
uv run egl data prepare labeled-s-4afc \
  --data_root /path/to/labeled_s \
  --trials_json /path/to/eval_object_categories_4150.json \
  --vocab_json /path/to/training_vocab.json \
  --output data/labeled_s_4afc_manifest.json
```

The converter checks the counts, the vocabulary coverage and every image path.
The manifest holds absolute paths, so regenerate it if the images move, and run
it with the [custom config](../evaluation.md#custom-configs) from the evaluation
guide.

## Labeled-S linear data

Two probes decode the 26 Labeled-S categories from frozen features.
`probe:labeled_s_linear` uses the canonical split of 2,886 training and 2,901
evaluation images. `probe:labeled_s_stratified_linear` follows the BabyCL paper
and splits all images 80/20 within each category. The two also
[fit different classifiers](../evaluation.md#linear-decoding-protocols), and both
take the category folders from [SAYCam access](#saycam-access) as `data_root`:

```yaml
_imports_:
  - egolearn.evaluation.benchmarks.labeled_s_linear
benchmarks:
  - _target_: "probe:labeled_s_linear"
    name: labeled_s_linear
    data_root: /path/to/labeled-s
```

`probe:labeled_s_linear` downloads its split file from a pinned commit. Without
internet access, set `split_file` to a local copy of
[`labeledS_train_val.txt`](https://github.com/agentic-learning-ai-lab/memory-storyboard/blob/c084d826fdc85a72e78477bb53c88b7353a89563/metas/labeledS_train_val.txt),
which the [`labeled-s-splits`](hub.md#gated-collections) collection also holds.
To match the BabyCL paper, use `probe:labeled_s_stratified_linear` with
`head: embed512`.

## BabyView object crops

The BabyView benchmarks use a public release of 5,921 object crops over 72
categories. The 4AFC trials use only categories in the model's training
vocabulary, 61 with the SAYCam vocabulary, while linear decoding uses all 72.
Download the crops and write both manifests:

```bash
uv run egl data prepare babyview-4afc --download \
  --vocab_json /path/to/training_vocab.json
```

The command writes the crops to `data/babyview-public-crops` and the manifests
to `data/babyview-4afc`. Image paths in the manifests are relative, so move the
two directories together. The manifests behind the published results, built
with the SAYCam vocabulary, are in the gated
[`babyview-4afc`](hub.md#gated-collections) collection; with access, fetch them
and download only the crops:

```bash
uv run egl data fetch babyview-4afc --revision v1
uv run egl data prepare babyview-4afc --download_only
```

Point the configs at the fetched files, not at `hf://` paths: a Hub reference
places the manifests in the Hugging Face cache, where their relative paths do
not reach the crops.
 Save this config as `babyview_eval.yaml` and run it with
`uv run egl eval babyview_eval.yaml checkpoint=run/final.pth`:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.babyview_4afc
benchmarks:
  - _target_: "probe:four_afc"
    name: babyview_4afc
    manifest: data/babyview-4afc/babyview_4afc_trials.json
    vocab_json: /path/to/training_vocab.json
    vision_head: embed512
  - _target_: "probe:linear"
    name: babyview_linear
    manifest: data/babyview-4afc/babyview_linear.json
```

For an image-only model, keep only the `probe:linear` entry. The crops have
varying aspect ratios, so keep the default `view: center_crop` and `size: 224`
when you compare results.

## Konkle 4AFC data

Konkle 4AFC tests word–object matching on studio photographs that appear in no
training corpus. It follows the protocol of
[Scaling cross-situational word learning on children's naturalistic experiences](https://arxiv.org/abs/2610.31842)
and uses 60 categories of the [Konkle et al. (2010)](https://konklab.fas.harvard.edu/)
object set, with 17 photographs each. Each trial shows a category word with one
photograph of that category and three photographs of other categories.

Download the object set from the Konkle lab, extract it, and draw the trials for
the model's training vocabulary:

```bash
uv run egl data prepare konkle-4afc --data_root /path/to/17-objects \
  --vocab_json /path/to/training_vocab.json
```

The command writes the trials to `data/konkle-4afc/trials.json`. A category
whose label is missing from the vocabulary gets no trials and scores chance
(0.25). The preparer prints these categories, and two models see the same trials
only when their vocabularies cover the same categories. Save this config as
`konkle_eval.yaml` and run it with
`uv run egl eval konkle_eval.yaml checkpoint=run/final.pth`:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.konkle_4afc
benchmarks:
  - _target_: "probe:four_afc"
    name: konkle_4afc
    manifest: data/konkle-4afc/trials.json
    vocab_json: /path/to/training_vocab.json
    vision_head: embed512
    view: resize
    size: 224
```

`four_afc_acc` averages over all 60 categories, with out-of-vocabulary categories
at 0.25. When some categories are out of vocabulary, `four_afc_acc_scored` also
reports the accuracy over the rest.

Scores are not directly comparable with the paper's. The paper's models score a
word against 17 region vectors per image and keep the best match, while
`probe:four_afc` scores one pooled image representation. The original download
also names files differently from the paper's copy, so the same seed selects
different photographs.

## BabyVLM benchmark data

The Visual Two-Word Test (VTWT) and Baby Winoground come from
[BabyVLM](https://shawnking98.github.io/BabyVLM/). Their images come from
Databrary and need [SAYCam access](#saycam-access):

1. Download `frames.txt` from the
   [Multimodal SAYCam-S release](https://nyu.databrary.org/volume/564/session/67447),
   rename it to `frames.zip` and extract it. It holds 600,285 images, about 14 GB.
2. For Baby Winoground, also download the `BabyVLM_V1` folder of synthetic
   negative images from [Databrary](https://nyu.databrary.org/volume/564/folder/77120).
3. Download the [VTWT](https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/vtwt)
   and [Baby Winoground](https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/baby_winoground)
   label CSVs from the authors' repository, whose
   [license](https://github.com/ShawnKing98/babylmms-eval/blob/main/LICENSE)
   covers the labels and code but not the images.

Save this config as `babyvlm_eval.yaml` and run it with
`uv run egl eval babyvlm_eval.yaml checkpoint=run/final.pth`:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.vtwt
  - egolearn.evaluation.benchmarks.baby_winoground
benchmarks:
  - _target_: "probe:vtwt"
    name: vtwt
    csv_path: /path/to/vtwt.csv
    image_dir: /path/to/saycam_frames
    vocab_json: /path/to/training_vocab.json
    vision_head: embed512
  - _target_: "probe:baby_winoground"
    name: baby_winoground
    csv_path: /path/to/baby_winoground.csv
    pos_dir: /path/to/saycam_frames
    neg_dir: /path/to/BabyVLM_V1
    vocab_json: /path/to/training_vocab.json
    vision_head: embed512
```

These settings match BabyCL models trained with EgoLearn, and CVCL models also
need `image_norm: clip` in both entries. A trial whose image is missing is
skipped and counted in `vtwt_skipped` or `bw_skipped`. The run still completes,
so before you report a score, check that both counts are zero. Also check that
`vtwt_n` equals the number of VTWT rows and `bw_n` the number of Baby Winoground
instances, since an empty CSV also scores 0.0 with nothing skipped.
