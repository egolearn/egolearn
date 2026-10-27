# BabyCL / BabyCLIP

BabyCL learns visual and verbal representations from SAYCam in a single
chronological pass. This project provides the BabyCLIP training recipe, an offline
variant, and configuration controls for memory and objective ablations.

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

The recipe combines visual instance discrimination, temporal classification, and
image–text contrastive learning. Each full batch contains three visual segments
with 32 sampled frames each, plus 32 utterance–frame pairs. Early batches use an
all-visual fallback when too few utterance pairs are available.

Visual segments and utterance pairs occupy separate memories, each with a short
FIFO tier whose overflow enters a long reservoir tier. After a segment is
admitted, training performs eight updates with freshly sampled batches.

The configuration is a code-level port, without a numerical-parity claim.
Training uses float32 arithmetic in one process. Automatic mixed precision,
used by the original trainer, is not implemented.

## Offline BabyCL variant

`saycam_iid` trains the full BabyCL objective with the complete selected dataset
available from the start. It inherits the streaming model, augmentations, losses,
and solver from `saycam_shared.yaml`.

This differs from the paper's CVCL-only IID baseline:

| Setting | Paper IID baseline | `saycam_iid` |
| --- | --- | --- |
| Losses | CVCL only | NT-Xent + TCL + CLIP-InfoNCE |
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
remove the segment slot, its fallback, and the NT-Xent and TCL losses together.

| Paper ablation | Configuration change |
| --- | --- |
| Buffer policy | Change `memories.*.tiers.long.update`, or use a single reservoir tier |
| Buffer size | Change each tier's `capacity` |
| Multimodal fraction | Use segment/pair counts 3/32, 2/64, or 1/96 for fractions 1/4, 1/2, or 3/4 |
| Clip-importance weighting | Enable `use_sample_weights` on the image–text loss and supply transcript `clip_importance` values |
| Four or eight resampled updates | Set `driver.rate.steps_per_admit` |
| CL-CVCL baseline | Use a pairs-only recipe and remove NT-Xent and TCL |
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
