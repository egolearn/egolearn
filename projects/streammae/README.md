# MAE and StreamMAE

This project trains Vision Transformer image encoders on egocentric video frames
with masked autoencoding. [MAE](https://arxiv.org/abs/2111.06377) hides 75% of
each image's patches and trains the model to reconstruct their pixels.
[StreamMAE](https://arxiv.org/abs/2609.40333) trains MAE on a continuous video
stream and adds DataDrop sampling, drop path, color jitter and motion-biased crops.
The recipes train on SAYCam, BabyView or WalkingTours frames, either in recording
order (streaming) or shuffled (IID).

## Quickstart

Install the training dependencies and check the training path on CPU with
synthetic frames:

```bash
uv sync --extra train
uv run egl train toy_mae dry_run=true
uv run egl train toy_mae
```

The last command runs five optimizer updates of a tiny MAE and writes run records
under `runs/toy_mae/records/`.

To train StreamMAE on SAYCam, prepare the frames as described under
[training data](#training-data), then check the recipe with a dry run and start
training:

```bash
export SAYCAM_ROOT=/path/to/saycam_5fps
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_saycam_streammae dry_run=true
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_saycam_streammae
```

The dry run counts the frames, prints the number of optimizer updates on a
`[budget]` line and exits. Training writes to `runs/streammae_saycam` unless you
pass another `run_dir`. Every recipe except `toy_mae` needs a CUDA GPU, and the
full-pass and matched recipes also need native BF16 support (compute capability
8.0 or later).

## Training data

Request access to SAYCam or BabyView through the
[dataset access guide](../../docs/evaluation/data.md), then extract upright JPEG
frames at 5 fps. Set `SAYCAM_ROOT` or `BABYVIEW_ROOT` to the frame directory, or
pass `stream.root_dir=/path/to/frames` on the command line.

SAYCam recipes read child S's recordings from directories named `S_*`, and each
filename ends in its timestamp in seconds:

```text
$SAYCAM_ROOT/
  S_20131130_1313_01/
    S_20131130_1313_01_259.20.jpg
    S_20131130_1313_01_259.40.jpg
```

BabyView recipes read every child's recordings in child, date and part order,
with frames numbered from 1 within each recording:

```text
$BABYVIEW_ROOT/
  <child>/
    <recording>/
      frame_00000001.jpg
      frame_00000002.jpg
```

The `stream.fps` setting states the prepared frame rate and does not resample
images. For WalkingTours and WT++, see [StreamMAE on WalkingTours](walkingtours.md).

## Recipes

`toy_mae` is packaged with egolearn. You pass the other configs, which live in
`projects/streammae/configs`, with `--config-dir` as in the quickstart.

| Config | Entry point | Trains |
| --- | --- | --- |
| `toy_mae` | `egl train` | A tiny MAE on synthetic frames, on CPU |
| `mae_small`, `mae_base` | `egolearn.train` | MAE ViT-S/16 or ViT-B/16 on SAYCam, one update per new frame |
| `full_saycam_streaming`, `full_babyview_streaming` | `egolearn.methods.mae.train` | MAE ViT-S/16 on frames in recording order |
| `full_saycam_iid`, `full_babyview_iid` | `egolearn.methods.mae.train` | MAE ViT-S/16 on shuffled frames, with the streaming update count |
| `full_saycam_streammae`, `full_babyview_streammae` | `egolearn.methods.mae.train` | StreamMAE ViT-S/16 in recording order |
| `matched_saycam`, `matched_babyview` | `egolearn.methods.mae.train` | One variant of the [matched ablations](#matched-ablations) |
| `full_wt_original_streammae`, `full_wtpp_london_streammae`, `full_walkingtours_streammae` | `egolearn.methods.mae.train` | StreamMAE on [WalkingTours or WT++](walkingtours.md) |

`mae_small` and `mae_base` keep the newest 512 frames in a window and train on
the whole window after each new frame. They run with the generic trainer:

```bash
uv run python -m egolearn.train --config-dir projects/streammae/configs \
    --config-name mae_small stream.root_dir="$SAYCAM_ROOT"
```

The other recipes use the full-pass runner, `egolearn.methods.mae.train`, which
makes one pass over every selected frame, sets the number of optimizer updates
from the frame count and saves resumable state. Streaming recipes advance 32
frames per update and train on 512 frames each time. IID recipes shuffle the same
frames and run the same number of updates, which is about 16 epochs. Both use
learning rate 1.6e-4 with a 5% warmup and then a constant rate, so the two
orders differ only in which frames each update sees.

## StreamMAE settings

StreamMAE keeps MAE's encoder, decoder and masked-patch loss but changes which
frames each update trains on and how they are augmented. Each addition has an
override that turns it off:

| Addition | Default | Override to disable |
| --- | --- | --- |
| DataDrop | Train on 512 frames sampled uniformly from a 2,048-frame FIFO window | `run_control.window_size=512 driver.warmup.admitted=480` |
| Drop path | Stochastic depth that rises linearly across blocks to 0.25 | `model.encoders.vision.drop_path_rate=0` |
| Color jitter | Applied with probability 0.5 | `views.reconstruction.jitter_probability=0` |
| Motion-biased crop | With probability 0.5, crop the region with the most motion among four candidates, then apply the usual MAE crop | `views.reconstruction.crop_probability=0` |

Motion is the pixel difference from the previous frame of the same recording, so
the prepared frame rate changes the motion signal. Set
`views.reconstruction.motion_candidates=1` to keep the two-stage crop but choose
its region at random.

To change the stride, set `run_control.stride`, `driver.advance.n` and
`driver.warmup.admitted` together. The warmup must equal the window size minus
the stride, and the runner rejects other values. Use a new run directory after
any of these changes.

## Matched ablations

The matched suite compares data order and each StreamMAE addition at equal cost.
Every variant starts training after the same first 2,048 frames, trains on 512
frames per update, and runs the same number of updates with the same learning
rate at each one. Select a variant with `ablation`:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name matched_saycam \
    ablation=vanilla seed=1337 stream.root_dir="$SAYCAM_ROOT"
```

| `ablation` | Recipe | FIFO window |
| --- | --- | ---: |
| `vanilla` | Streaming MAE | 512 |
| `iid` | Shuffled MAE | None |
| `datadrop` | Streaming MAE with DataDrop | 2,048 |
| `drop_path` | `datadrop` plus drop path | 2,048 |
| `color_jitter` | `drop_path` plus color jitter | 2,048 |
| `random_crops` | `color_jitter` plus two-stage crops at random regions | 2,048 |
| `motion` (default) | `random_crops` with motion-ranked regions, which is full StreamMAE | 2,048 |

`datadrop` is the MAE baseline that the StreamMAE paper compares against. Use
`matched_babyview` with `BABYVIEW_ROOT` for BabyView. Each run writes to
`runs/streammae_matched_<dataset>/<ablation>/seed_<seed>`, for example
`runs/streammae_matched_saycam/vanilla/seed_1337`.

The `full_*` recipes instead start after one window of their own size. Full-pass
MAE therefore runs a few more updates than full-pass StreamMAE on the same frames,
so use the matched suite when you compare the two.

## Long runs and resume

The full-pass runner saves resumable state to `snapshot_latest.pth` every 5,000
updates and whenever it pauses. To resume, rerun the same command with the same
`run_dir`. The runner records the configuration, the frame count, the installed
code and the library versions in `manifest.json`, and it refuses to resume if any
of them changed. You can change loader, logging and checkpoint-frequency settings between
attempts. Use a new `run_dir` for each dataset, seed or setting.

| Exit code | Meaning |
| --- | --- |
| `0` | Training finished and `final.pth` was written |
| `75` | Paused at `run_control.max_seconds` (46 hours by default) or on SIGUSR1, with state saved |
| `76` | Stopped after `run_control.stop_after_steps` updates, with state saved |

For a short check on real data, append `run_control.stop_after_steps=10`. The run
keeps its full learning-rate schedule, and rerunning it with
`run_control.stop_after_steps=0` continues training. Set `run_control.expected_frames`
to the frame count from your own dataset inventory so that an incomplete copy of
the data stops the run.

On a Slurm cluster, submit the same command through the
[cluster launcher](../../docs/cluster.md) with an absolute `run_dir`. Set
`run_control.max_seconds` below the job's time limit. To requeue paused jobs
automatically, export `EGOLEARN_REQUEUE_ON_PAUSE=1` and pass `--requeue` to the
launcher.

The run directory also holds `metrics.jsonl` with the loss, learning rate and
frame counts. SAYCam and BabyView recipes keep a `snapshot_epoch_NNN.pth` at the
update that ends each IID epoch, in both streaming and IID runs.

## Export and evaluation

`final.pth` and the snapshots hold the full training state, which loads with
pickle. Export the encoder weights from a run you created:

```bash
uv run egl export "$RUN_DIR/final.pth" "$EXPORT_DIR" --trusted-snapshot
```

The export directory holds `model.safetensors` and `model.json`. For an unfinished
run, export `snapshot_latest.pth` or an epoch snapshot instead. Read
[checkpoint trust](../../SECURITY.md#model-export-and-checkpoint-trust) before
loading a snapshot that someone else made.

To evaluate on ImageNet, set `IMAGENET_ROOT` to an ImageFolder directory with
`train/` and `val/` subdirectories. Fit a linear classifier on the frozen encoder,
or fine-tune the whole network:

```bash
uv run egl eval imagenet_linear_eval checkpoint="$EXPORT_DIR" \
    data_root="$IMAGENET_ROOT" benchmarks.0.encoder=vision output_json=imagenet_linear.json
uv run egl eval imagenet_full_finetune_eval checkpoint="$EXPORT_DIR" \
    data_root="$IMAGENET_ROOT" benchmarks.0.encoder=vision output_json=imagenet_finetune.json
```

Report the two protocols separately, since they measure different things. The
[checkpoint evaluation guide](../../docs/evaluation.md) covers other benchmarks
and the [fine-tuning settings](../../docs/evaluation.md#full-network-fine-tuning).

## Differences from the paper

- Streaming and IID runs share one learning rate with a constant rate after
  warmup. The paper tunes the IID learning rate separately and uses a cosine
  schedule.
- SAYCam and BabyView recipes advance 32 frames per update and keep the prepared
  frame size, scaling the paper's 336 × 592 first-stage crop region to it. The
  WalkingTours recipes use the paper's stride of 1 and its frame geometry.
- The paper's main protocol is full fine-tuning. The packaged fine-tuning
  settings are not tuned to reproduce its numbers, and frozen linear probes
  measure something different.
- The code is an independent implementation that does not derive from the
  CC BY-NC MAE reference code, and it cannot load reference MAE checkpoints.
