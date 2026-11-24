# MAE and StreamMAE

MAE and StreamMAE train image encoders from prepared video frames using masked
reconstruction. This guide covers recipe selection, data setup, training, and
checkpoint evaluation. The [method guide](method.md) describes the objective and
sampling choices, without claiming paper reproduction or downstream performance.

## Recipe selection

Use the matched suite when comparing methods with equal optimizer budgets.
The other full-pass recipes derive their budgets from their own window sizes.

| Recipe | Use | Commands and settings |
| --- | --- | --- |
| `toy_mae` | Check the training and export workflow on CPU with synthetic data | [CPU example](#cpu-example) |
| `mae_small`, `mae_base` | Basic ViT-S/16 or ViT-B/16 MAE with stride 1 | [Basic MAE](#basic-mae) |
| `full_*_streaming`, `full_*_iid` | Compare chronological and shuffled MAE on SAYCam or BabyView at stride 32 | [Full-pass training](#full-pass-training) |
| `full_saycam_streammae`, `full_babyview_streammae` | Train StreamMAE at stride 32 with a 2,048-frame window | [Full-pass training](#full-pass-training) |
| `matched_saycam`, `matched_babyview` | Compare MAE, IID, DataDrop, and augmentation variants with equal budgets | [Matched ablations](ablations.md) |
| `full_wt_original_streammae` | Train on the ten original WalkingTours videos | [Original WalkingTours](walkingtours.md) |
| `full_wtpp_london_streammae`, `full_walkingtours_streammae` | Train on a supplied WT++ source or another ordered collection | [WT++ preparation and training](wtpp.md) |

## Prerequisites


Run the commands from the repository root after installing the training dependencies
with `uv sync --extra train`. The synthetic `toy_mae` example runs on CPU without
data or pretrained weights. The project recipes default to CUDA. With their
default CUDA/BF16 settings, the `full_*` and `matched_*` recipes require a GPU
with native BF16 support (CUDA compute capability 8.0 or later).

### Dataset access and layout

Obtain the source data through [SAYCam access](../../docs/evaluation/data.md#saycam-access)
or [BabyView access](https://babyview-project.github.io/access/), then prepare the
frames locally. Set `SAYCAM_ROOT` or `BABYVIEW_ROOT` to your frame directory, or
pass its path as `stream.root_dir` in the training command.

SAYCam recipes select child S from session directories named `S_*`. Filenames
end in a timestamp in seconds, for example:

```text
$SAYCAM_ROOT/
  S_20131130_1313_01/
    S_20131130_1313_01_259.20.jpg
    S_20131130_1313_01_259.40.jpg
```

BabyView recipes read all children from nested recording directories, ordered by
child, date, and numeric part, with the recording ID breaking ties. Number frames
from 1 within each recording:

```text
$BABYVIEW_ROOT/
  <child>/
    <recording>/
      frame_00000001.jpg
      frame_00000002.jpg
```

Apply the intended orientation and dataset crop during preparation. The SAYCam
and BabyView recipes default to 5 fps and do not apply an additional vertical
flip. `stream.fps` describes the prepared frame rate and does not resample images.

For original WalkingTours, follow the [download and frame preparation guide](../../docs/datasets/walkingtours.md).
WT++ uses a separate source supplied by you, described under [WT++ preparation](wtpp.md).

## CPU example


The synthetic CPU example validates the whole training path:

```bash
uv run python -m egolearn.train --config-name toy_mae dry_run=true
uv run python -m egolearn.train --config-name toy_mae
```

The second command performs five optimizer updates and writes run records under
`runs/toy_mae/records/`. It does not save weights by default. To create and export
a toy checkpoint, enable checkpoint saving and choose new output directories:

```bash
uv run python -m egolearn.train --config-name toy_mae \
    run_dir=runs/toy_mae_export +hooks.checkpoint_every=5
uv run egl export runs/toy_mae_export/final.pth exports/toy_mae
```

The export contains `model.safetensors` and `model.json`. This toy encoder expects
32×32 images and is for checking the workflow, not measuring representation quality.

## Training

### Basic MAE

Train ViT-S/16 or ViT-B/16 from scratch on your prepared SAYCam frames:

```bash
uv run python -m egolearn.train --config-dir projects/streammae/configs \
    --config-name mae_small stream.root_dir="$SAYCAM_ROOT"
uv run python -m egolearn.train --config-dir projects/streammae/configs \
    --config-name mae_base stream.root_dir="$SAYCAM_ROOT"
```

Append `dry_run=true` to check model construction and wiring. These recipes still
need an existing frame directory, unlike the synthetic example. If your prepared
frames need flipping, add `+views.reconstruction.vertical_flip=true` for either
of these two configs. See [frame selection](method.md#frame-selection) and
[optimization](method.md#optimization) before changing their batch size or schedule.

### Full-pass training

Use `egolearn.methods.mae.train` for the `full_*` and `matched_*` recipes.
It derives the update budget from your frame inventory and handles resumable state.
For full-pass MAE, choose a dataset and streaming or IID ordering:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_saycam_streaming \
    stream.root_dir="$SAYCAM_ROOT"
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_babyview_iid \
    stream.root_dir="$BABYVIEW_ROOT"
```

The companion configs are `full_saycam_iid` and `full_babyview_streaming`.
For StreamMAE on either dataset:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_saycam_streammae \
    stream.root_dir="$SAYCAM_ROOT"
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_babyview_streammae \
    stream.root_dir="$BABYVIEW_ROOT"
```

Choose a separate `run_dir` for each experiment and seed. Follow
[training and resume](training.md) for frame-count checks, output files, pauses,
and [Slurm submission](training.md#slurm-launcher). The [matched suite](ablations.md)
provides its own commands for comparisons across window sizes and augmentations.

## Checkpoint export and evaluation


The `full_*` and `matched_*` runners write `final.pth` only after training completes.
For a paused run, use `snapshot_latest.pth` or an archived milestone instead.
These files contain resumable training state. Keep the run directory for resume
and export a separate copy of the weights for evaluation.

Set `RUN_DIR` to your training output directory and `EXPORT_DIR` to a new export
directory. For a snapshot you created or otherwise trust, run:

```bash
uv run egl export "$RUN_DIR/final.pth" "$EXPORT_DIR" --trusted-snapshot
```

The export writes `model.safetensors` and `model.json`, including the model
configuration and source checkpoint identity. Existing export directories are
refused. If you evaluate the original snapshot directly, pass
`trusted_snapshot=true` to `egl eval` instead. See [checkpoint trust](../../SECURITY.md#model-export-and-checkpoint-trust).

### ImageNet evaluation

Obtain the ImageNet data and set `IMAGENET_ROOT` to an ImageFolder directory
containing matching `train/<class>/` and `val/<class>/` folders. The commands
below use the 224×224 inputs expected by the project recipes and default to CUDA.
For the 32×32 toy export, you must also set `benchmarks.0.size=32`.

Fit a linear classifier while keeping the pretrained encoder frozen:

```bash
uv run egl eval imagenet_linear_eval \
    checkpoint="$EXPORT_DIR" data_root="$IMAGENET_ROOT" \
    benchmarks.0.encoder=vision output_json=imagenet_linear.json
```

To update the encoder and train a new classification head, use full-network
fine-tuning:

```bash
uv run egl eval imagenet_full_finetune_eval \
    checkpoint="$EXPORT_DIR" data_root="$IMAGENET_ROOT" \
    benchmarks.0.encoder=vision output_json=imagenet_finetune.json
```

Both commands write metrics and protocol details to the named JSON file.
The [evaluation guide](../../docs/evaluation.md) describes frozen probes, and
[full-network fine-tuning](../../docs/evaluation.md#full-network-fine-tuning)
documents adaptation settings, saved checkpoints, and resume. Record these two
protocols separately. The packaged fine-tuning defaults are configurable and do
not establish equivalence to the StreamMAE paper's fine-tuning or attentive-probing results.
