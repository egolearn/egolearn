# BabyCL / BabyCLIP

BabyCL learns visual and verbal representations from a child's egocentric video
in one chronological pass. This project implements the BabyCLIP recipe from the
[BabyCL paper](https://arxiv.org/abs/2606.05115) on SAYCam, an offline variant,
and the settings for the paper's ablations. Visual-only versions train on
[KrishnaCam](docs/krishnacam.md) and [WalkingTours](docs/walkingtours.md).

A frozen ImageNet ResNeXt-50 cuts the stream into segments of about three
minutes, which enter one replay memory while utterance–frame pairs enter
another. After each new segment, the learner runs eight updates on batches of
three segments and 32 pairs. Each batch trains instance discrimination, a
temporal objective that groups frames from the same segment, and image–text
contrastive learning. The
[architecture page](../../docs/design/architecture.md#method-differences)
compares these choices with Memory Storyboard's.

## Training

You need the SAYCam frames, aligned transcripts and vocabulary described in
[SAYCam inputs](docs/training_data.md). Install the training dependencies with
`uv sync --extra train`, set the input paths, and validate the pipeline:

```bash
export SAYCAM_ROOT=/path/to/saycam_5fps
export SAYCAM_TRANSCRIPTS=/path/to/expand_correct_seg.json
export BABYCL_VOCAB=/path/to/vocab.json
uv run egl train projects/babycl/configs/saycam_default.yaml \
  "stream.root_dir=$SAYCAM_ROOT" "stream.transcript_json=$SAYCAM_TRANSCRIPTS" \
  "tokenizer.vocab_json=$BABYCL_VOCAB" 'probes=[]' dry_run=true
```

The dry run ends with `[dry-run] pipeline valid.`, and without `dry_run=true`
the command trains on a GPU. `probes=[]` turns off the in-training evaluation
probes. To keep them, set their data paths from
[`saycam_shared.yaml`](configs/saycam_shared.yaml) after preparing the files
with the [evaluation data guide](../../docs/evaluation/data.md).

`saycam_default` stops at the paper run's fixed step count, before the end of
SAYCam-S. To train on the whole stream, add
`driver.max_steps=null solver.schedule.total_steps=null`.

## Configuration

Each of the paper's ablations is a config change:

| Paper setting | Change |
| --- | --- |
| Buffer policy or size | `memories.*.tiers.*.update` or `capacity` |
| Multimodal fraction | Segment and pair counts of 3/32, 2/64 or 1/96 for 1/4, 1/2 or 3/4 |
| Updates per segment | `driver.rate.steps_per_admit` |
| Clip-importance weighting | `use_sample_weights: true` on the image–text loss, with `clip_importance` in the transcripts |
| CL-CVCL baseline | Remove the segment slot, its fallback, and the instance and temporal losses |

## SAYCam results

The table compares the final checkpoints of two `saycam_default` runs with the
paper, in percent. The paper reports Labeled-S 4AFC as a five-seed mean with an
approximate 95% interval, which differs between paper versions, and the other
rows from its best seed. Baby Winoground is the group score.

| Benchmark | Seed 1337 | Seed 2002 | Paper |
| --- | ---: | ---: | ---: |
| Labeled-S 4AFC | 42.53 | 43.04 | 42.72 ± 1.36 |
| Labeled-S linear | 45.26 | 44.91 | 44.95 |
| mini-ImageNet linear | 25.52 | 26.12 | 25.60 |
| VTWT | 60.60 | 58.43 | 61.32 |
| Baby Winoground | 15.89 | 12.05 | 14.79 |
| BabyView 4AFC | 30.99 | 30.29 | 32.53 |
| BabyView linear | 49.71 | 52.06 | 49.96 |

To reproduce a column, set the evaluation-data variables listed at the top of
[`paper_evals.yaml`](configs/paper_evals.yaml), then train and evaluate:

```bash
export RUN_DIR=runs/babycl-saycam-seed1337
uv run egl train projects/babycl/configs/saycam_default.yaml seed=1337 \
  "run_dir=$RUN_DIR" "stream.root_dir=$SAYCAM_ROOT" \
  "stream.transcript_json=$SAYCAM_TRANSCRIPTS" "tokenizer.vocab_json=$BABYCL_VOCAB" \
  'probes=[]'
uv run egl eval projects/babycl/configs/paper_evals.yaml "checkpoint=$RUN_DIR/final.pth"
```

### Differences from the paper

- EgoLearn draws its randomness differently, so a seed here is not the paper's
  seed. Same-seed reruns of the original code vary by about two points on
  Labeled-S 4AFC.
- Floating-point differences in the GPU segmenter move a few near-tied segment
  boundaries relative to the original run.
- The original code converts segment frames back to 8 bits by truncation before
  augmentation, which darkens some pixels by one level. EgoLearn skips that
  conversion.

## Offline variant

`saycam_iid.yaml` trains the same model and objectives with the whole dataset
available from the start, reading segments from a
[storyboard file](docs/training_data.md#offline-storyboard) that you provide:

```bash
export BABYCL_STORYBOARD=/path/to/storyboard.json
uv run egl train projects/babycl/configs/saycam_iid.yaml \
  "stream.root_dir=$SAYCAM_ROOT" "stream.transcript_json=$SAYCAM_TRANSCRIPTS" \
  "tokenizer.vocab_json=$BABYCL_VOCAB" "driver.batch.storyboard_json=$BABYCL_STORYBOARD"
```

The run lasts one epoch unless you set `driver.epochs`. Unlike the paper's IID
baseline, it trains all three objectives.
