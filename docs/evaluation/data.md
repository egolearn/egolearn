# SAYCam evaluation data

Prepare Labeled-S, VTWT, and Baby Winoground inputs for models with image and
text representations. You need approved image access and the evaluated model's
training vocabulary before preparing these benchmarks.

## SAYCam access

SAYCam videos, SAYCam-S frames, and Labeled-S images are distributed through
[Databrary](https://nyu.databrary.org/volume/564/). Obtain access with your
institution's IRB approval and follow the applicable Databrary data-use terms.
The [CVCL dataset notes][cvcl-data] and
[BabyVLM preparation notes][babyvlm-preparation] identify the source releases.

Public labels do not grant access to the corresponding images, and restricted
images and local manifests must stay outside version control. See
[artifact access tiers](hub.md#access-tiers) before sharing any prepared output.

| Benchmark | Image source | Metadata | Preparation |
| --- | --- | --- | --- |
| Labeled-S 4AFC | Labeled-S release on Databrary | Frozen BabyCL trials and the model's training vocabulary | [Manifest and vocabulary](#labeled-s-4afc-data) |
| VTWT | Multimodal SAYCam-S frames on Databrary | [VTWT labels][vtwt-labels] | [Frames and CSV inputs](#babyvlm-benchmark-data) |
| Baby Winoground | SAYCam-S frames and `BabyVLM_V1` synthetic images on Databrary | [Baby Winoground labels][bw-labels] | [Frames, negatives, and CSV inputs](#babyvlm-benchmark-data) |

## Labeled-S 4AFC data

The Labeled-S configuration of `probe:four_afc` uses BabyCL's frozen 4,150 trials
over 23 categories. Retain the released 224×224 crops in their
`<category>/<filename>` layout, separate from the SAYCam-S training frames
used by VTWT.

Obtain the original `eval_object_categories_4150.json` from the BabyCL evaluation
assets, or request the frozen metadata from the authors if you lack access.
The [BabyCL paper](https://arxiv.org/html/2606.05115v1) identifies its code release
as supplementary material. Randomly generating replacement trials produces a
different evaluation.

### Training vocabulary

Use your model's exact training vocabulary to preserve its word-to-integer
mapping during evaluation. BabyCL's CVCL
vocabulary is available as [`multimodal/vocab.json`][cvcl-vocab], pinned at
revision `d64c1bb164ed4c25f0cad8a1c3b7a4d68efae326`, and must remain unchanged
if your model was trained with it.

For a custom vocabulary, retain the original training file and check that every
target label is present. Building a vocabulary from the 23 evaluation labels
assigns different embedding indices, and checkpoint embedding weights alone
cannot recover the word names.

### Manifest conversion

Set `LABELED_S_ROOT` to the category-folder root, `LABELED_S_TRIALS` to the
original trial JSON, and `VOCAB_JSON` to the training vocabulary file. Then run:

```bash
uv run egl data prepare labeled-s-4afc \
  --data_root "$LABELED_S_ROOT" --trials_json "$LABELED_S_TRIALS" \
  --vocab_json "$VOCAB_JSON" --output data/labeled_s_4afc_manifest.json
```

The converter checks trial and category counts, target vocabulary coverage,
and local image paths. It preserves trial membership, foil order, and the target
at index 0, and refuses to overwrite an existing output file.

The input JSON contains a `data` list whose rows supply `target_category`,
`target_img_filename`, and three `foil_img_filenames`. The output contains a
`trials` list with `text`, four `images`, and `answer: 0` for each trial.

Pass the output as `manifest`, the same training vocabulary as `vocab_json`, and
the model's image-text projection head as `vision_head`, which is `embed512`
for BabyCL.
The [Labeled-S configuration](../../src/egolearn/evaluation/benchmarks/labeled_s_4afc/__init__.py)
shows these settings. The manifest contains absolute local paths, so regenerate
it if the image root moves.

## Labeled-S linear data

`probe:labeled_s_linear` uses the canonical 26-class split with 2,886 training
images and 2,901 evaluation images. Obtain the restricted images through
[SAYCam access](#saycam-access), keeping their category folders.
This protocol differs from BabyCL's reported 24-class linear evaluation.

The adapter verifies the split metadata against a pinned SHA-256. With
`split_file: null`, it downloads the small split file from an immutable upstream
commit, without downloading images. For offline evaluation, supply a local copy
of [`labeledS_train_val.txt`](https://github.com/agentic-learning-ai-lab/memory-storyboard/blob/c084d826fdc85a72e78477bb53c88b7353a89563/metas/labeledS_train_val.txt).

Save this complete config as `labeled_s_linear_eval.yaml`, replacing the image
root and local split path:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.labeled_s_linear
benchmarks:
  - _target_: "probe:labeled_s_linear"
    name: labeled_s_linear
    data_root: /path/to/labeled-s
    split_file: /path/to/labeledS_train_val.txt
    encoder: image
    head: null
    view: center_crop
    size: 224
```

Run `uv run egl eval labeled_s_linear_eval.yaml checkpoint=run/final.pth` with
your checkpoint path. The `top1` metric reports held-out accuracy on the
canonical split. See [linear decoding protocols](../evaluation.md#linear-decoding-protocols)
before comparing it with other linear probes.

## BabyVLM benchmark data

The Visual Two-Word Test and Baby Winoground use image-text comparisons from
[*BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning*](https://shawnking98.github.io/BabyVLM/).
The authors' [evaluation repository](https://github.com/ShawnKing98/babylmms-eval)
provides task labels separately from the restricted images.

After obtaining [SAYCam access](#saycam-access), follow the
[upstream preparation notes][babyvlm-preparation]:

1. Download `frames.txt` from the
   [Multimodal SAYCam-S release](https://nyu.databrary.org/volume/564/session/67447).
2. Rename it to `frames.zip` and extract it into a local image directory.
   The upstream release contains 600,285 images, approximately 14 GB.
3. For Baby Winoground, also obtain the `BabyVLM_V1` folder from
   [Databrary][babyvlm-images] for synthetic negative images.

Consult the [upstream task license][babyvlm-license] for the label and code
terms. Those terms do not replace the image releases' access requirements.

### VTWT inputs

Configure `probe:vtwt` with `csv_path` for the labels and `image_dir` for SAYCam
frames. The CSV requires these columns:

```text
image,positive_phrase,negative_phrase
```

The probe counts a trial as correct only when the positive phrase has strictly
higher cosine similarity with the image than the negative phrase. It returns
accuracy as `vtwt_acc` and the number scored as `vtwt_n`.
`vtwt_skipped` counts trials skipped because an image is missing.

### Baby Winoground inputs

Configure `probe:baby_winoground` with `csv_path`, `pos_dir` for SAYCam frames,
and `neg_dir` for synthetic images. The CSV requires these columns:

```text
instance_id,positive_phrase,negative_phrase,image,image_tag
```

Each instance needs rows tagged `positive` and `negative`. A group is correct
only when each image prefers its corresponding phrase, using strict similarity
comparisons. Instances with missing rows or images are skipped and counted in
`bw_skipped`.

The probe returns `bw_group_acc` for both comparisons being correct,
`bw_pos_ctx_acc` for the positive image, and `bw_neg_ctx_acc` for the negative
image. `bw_n` counts scored instances.

### Input completeness

Require a nonzero `vtwt_n` or `bw_n` and zero skipped items before reporting a
full-benchmark score. Check the scored counts against the distributed labels.
With no usable images, both probes return finite `0.0` accuracies and the
standalone runner still marks the evaluation complete.

A nonzero skipped count means the score covers a subset of the requested
benchmark. Check directory paths and extraction completeness before comparing
that result with other runs.

### Shared model settings

Both probes require the training vocabulary as `vocab_json` and the image-text
projection head as `vision_head`. Set `image_norm=imagenet` for EgoLearn-trained
BabyCL models or `image_norm=clip` for the CVCL family. By default, images use
bicubic resize and center crop at a resolution of 224 pixels.

Tokenization lowercases phrases, splits on whitespace, and keeps at most 32
words. Unknown words map to padding index 0 and still count in the pooling
length, with no start or end markers added. Use the training vocabulary so
known words select the correct embeddings.

### BabyVLM evaluation config

Save this as `babyvlm_eval.yaml` and replace each input path with the corresponding
prepared file or directory:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.vtwt
  - egolearn.evaluation.benchmarks.baby_winoground
vocab_json: /path/to/training_vocab.json
vision_head: embed512
image_norm: imagenet
benchmarks:
  - _target_: "probe:vtwt"
    name: vtwt
    csv_path: /path/to/vtwt.csv
    image_dir: /path/to/saycam_frames
    vocab_json: ${vocab_json}
    vision_head: ${vision_head}
    image_norm: ${image_norm}
  - _target_: "probe:baby_winoground"
    name: baby_winoground
    csv_path: /path/to/baby_winoground.csv
    pos_dir: /path/to/saycam_frames
    neg_dir: /path/to/BabyVLM_V1
    vocab_json: ${vocab_json}
    vision_head: ${vision_head}
    image_norm: ${image_norm}
```

The shared head and normalization values above match EgoLearn-trained BabyCL
models. Set them to match your evaluated model, then compose the config and run:

```bash
uv run egl eval babyvlm_eval.yaml checkpoint=run/final.pth --cfg job
uv run egl eval babyvlm_eval.yaml \
  checkpoint=run/final.pth output_json=babyvlm_results.json
```

Configuration composition does not open the dataset. A full evaluation needs all
CSV, image, vocabulary, and checkpoint inputs to be accessible.

[cvcl-data]: https://github.com/wkvong/multimodal-baby#datasets
[cvcl-vocab]: https://github.com/wkvong/multimodal-baby/blob/d64c1bb164ed4c25f0cad8a1c3b7a4d68efae326/multimodal/vocab.json
[babyvlm-preparation]: https://github.com/ShawnKing98/babylmms-eval#data-preparation
[babyvlm-license]: https://github.com/ShawnKing98/babylmms-eval/blob/main/LICENSE
[vtwt-labels]: https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/vtwt
[bw-labels]: https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/baby_winoground
[babyvlm-images]: https://nyu.databrary.org/volume/564/folder/77120
