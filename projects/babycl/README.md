# BabyCL / BabyCLIP

BabyCL learns visual and verbal representations from SAYCam in a single
chronological pass. This project provides the BabyCLIP training recipe, an offline
variant, and configuration controls for memory and objective ablations.

For silent KrishnaCam videos, use the
[visual BabyCL training and benchmark guide](docs/krishnacam.md).
For the original WalkingTours videos, use the [WalkingTours experiment guide](docs/walkingtours.md).

## Training

Use the [default config](configs/saycam_default.yaml) for streaming training.
Override `stream.root_dir`, `stream.transcript_json`, and `tokenizer.vocab_json`
with your input paths. BabyCL expects upright SAYCam frames, already flipped from
their raw orientation.

The [shared settings](configs/saycam_shared.yaml) also construct Labeled-S 4AFC,
VTWT, and Baby Winoground probes. Override their manifest, CSV, image, and vocabulary
paths, or set `probes=[]` to disable them. Their manifest, CSV, and vocabulary files
must exist even for `dry_run=true`. See the
[evaluation data guide](../../docs/evaluation/data.md) for access and preparation.

```bash
egl train projects/babycl/configs/saycam_default.yaml dry_run=true
egl train projects/babycl/configs/saycam_default.yaml
```

The recipe combines visual instance discrimination, segment supervised contrastive
learning (SupCon), and image–text contrastive learning. Each full batch contains
three visual segments with 32 sampled frames each, plus 32 utterance–frame pairs.
Early batches use an all-visual fallback when too few utterance pairs are available.

SupCon treats frames that share a global segment ID as positives, so segments
that the segmenter merges count as one class.

Visual segments and utterance pairs occupy separate memories, each with a short
FIFO tier whose overflow enters a long reservoir tier. After a segment is
admitted, training performs eight updates with freshly sampled batches.

`saycam_default` retains the paper's run length and precision settings:

- **Run length.** The original fixed 26568 steps before training (3321
  offline-storyboard segments times 8 updates) and indexed the learning-rate
  schedule by that count. It stopped after about 73% of SAYCam-S.
  `driver.max_steps` and `solver.schedule.total_steps` reproduce this. Set both
  to `null` to train on the whole stream with the progress-driven schedule.
- **Precision.** `driver.amp` runs the segment branch in float16 with a loss
  scaler and keeps the cross-modal branch in float32.

The segment views also use the 2-D Gaussian blur (`blur_impl: torchvision`), an
utterance that ends on a frame is visible at that frame
(`stream.transcript_first`), and the pairs long tier holds 16868 items.

## SAYCam results

The table reports the earlier `saycam_default`, which used temporal classification
instead of SupCon, with seeds 1337 and 2002 and final checkpoints.
All scores are percentages. The paper column quotes the updated version of the
[paper](https://arxiv.org/abs/2606.05115), which reaches arXiv later; the
version there now reports an earlier Labeled-S 4AFC figure.

| Benchmark | Seed 1337 | Seed 2002 | Paper | Differs from the paper |
| --- | ---: | ---: | ---: | --- |
| Labeled-S 4AFC | 42.53 | 43.04 | 42.72 ± 1.36 | Paper: end-of-training mean ± approximate 95% CI over five seeds |
| Labeled-S linear | 45.26 | 44.91 | 44.95 | Paper: end-of-training checkpoint of the best seed |
| mini-ImageNet linear | 25.52 | 26.12 | 25.60 | Paper: end-of-training checkpoint of the best seed |
| VTWT | 60.60 | 58.43 | 61.32 | Paper: end-of-training checkpoint of the best seed |
| Baby Winoground | 15.89 | 12.05 | 14.79 | Paper: end-of-training checkpoint of the best seed |
| BabyView 4AFC | 30.99 | 30.29 | 32.53 | Paper: end-of-training checkpoint of the best seed |
| BabyView linear | 49.71 | 52.06 | 49.96 | Paper: end-of-training checkpoint of the best seed |

Baby Winoground reports the group score, which counts an item only when both
the original and the edited scene prefer their own caption. Every row uses the
paper's protocol, so the remaining differences come from training.

### Differences from the paper

- **Randomness.** EgoLearn draws its sampling, augmentation and initialization
  randomness differently, so a seed here is not the paper's seed. The paper's
  code is not run-to-run reproducible either: same-seed reruns of its default
  run reach a final Labeled-S 4AFC between 42.2% and 44.5%.
- **Segmentation.** The frozen ResNeXt-50 that segments the stream runs on a
  GPU. Floating-point differences move a few near-tied cuts, which changes 9 of
  the 3751 segments the run trains on.
- **Pixel rounding.** The original converted segment frames to 8 bits by
  truncation before augmentation, which darkens some intensities by one level.

### Reproducing the table

Check out commit `8f39160` for the temporal-classification recipe these runs
used. Train each seed, then evaluate its final checkpoint. The training command
skips the in-training probes, which do not change training.
`paper_evals.yaml` lists the environment variables that point at the
evaluation data.

```bash
egl train projects/babycl/configs/saycam_default.yaml seed=1337 \
  run_dir=$RUN_DIR stream.root_dir=$SAYCAM_ROOT \
  stream.transcript_json=$SAYCAM_TRANSCRIPTS tokenizer.vocab_json=$BABYCL_VOCAB \
  probes=[]
egl eval projects/babycl/configs/paper_evals.yaml checkpoint=$RUN_DIR/final.pth
```

One run takes about 6 hours on an NVIDIA L40S.

## Offline BabyCL variant

`saycam_iid` trains the full BabyCL objective with the complete selected dataset
available from the start. It inherits the streaming model, augmentations, losses,
and solver from `saycam_shared.yaml`.

This differs from the paper's CVCL-only IID baseline:

| Setting | Paper IID baseline | `saycam_iid` |
| --- | --- | --- |
| Losses | CVCL only | NT-Xent + SupCon + CLIP-InfoNCE |
| Batch | 8 image/text pairs | 3 x 32 visual frames + 32 image/text pairs |
| Epochs | 400 | 1 by default, configurable with `driver.epochs` |
| AdamW learning rate | 1e-4 | 1e-3 vision/embed head, 1e-4 remaining parameters |
| Weight decay | 0.1 | 1e-4, with vision/embed BN and bias exempt |
| Schedule | ReduceLROnPlateau | Cosine with warmup |

An epoch shuffles every aligned utterance chunk once and drops the incomplete
batch, while visual segments are sampled independently with replacement. Supply a
complete storyboard manifest using the [offline input guide](docs/offline.md),
which documents its format and launch command.

## Ablations

The following settings express the paper's ablation choices. For pairs-only runs,
remove the segment slot, its fallback, and the NT-Xent and SupCon losses together.

| Paper ablation | Configuration change |
| --- | --- |
| Buffer policy | Change `memories.*.tiers.long.update`, or use a single reservoir tier |
| Buffer size | Change each tier's `capacity` |
| Multimodal fraction | Use segment/pair counts 3/32, 2/64, or 1/96 for fractions 1/4, 1/2, or 3/4 |
| Clip-importance weighting | Enable `use_sample_weights` on the image–text loss and supply transcript `clip_importance` values |
| Four or eight resampled updates | Set `driver.rate.steps_per_admit` |
| CL-CVCL baseline | Use a pairs-only recipe and remove NT-Xent and SupCon |
| One-pass CVCL | Also set `driver.rate.steps_per_admit: 1` |
| BabyView | Select `stream:babyview`, normalize text, and supply frames pre-cropped to 9:16 portrait |

Without transcript `clip_importance` values, the weighting flag leaves the loss
unweighted. No BabyView training config or portrait pre-crop view ships.

## Segmentation and implementation

A frozen ImageNet ResNeXt supplies visual features for the segmenter, which expands
utterance-anchored chunks by similarity and packs them into segments of about three
minutes. It finalizes each segment only after the
stream has advanced far enough that later observations cannot change it.

Image–text pairs use contiguous runs of the utterance's original frame timestamps
rather than the segmenter's similarity-expanded chunks. Each chunk enters memory
after all its frames have appeared, with duplicates removed.

The [method package](../../src/egolearn/methods/babycl/) contains the ported segmentation
math, admission components, text encoder, and augmentation recipes. Component
docstrings record adaptations from the original implementation. The
[architecture guide](../../docs/design/architecture.md#method-differences) compares
BabyCL's memory and training choices with Memory Storyboard.
