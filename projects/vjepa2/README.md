# V-JEPA 2

[V-JEPA 2](https://arxiv.org/abs/2506.09985) (Assran et al., 2025) learns video
representations by predicting the features of masked parts of a clip from the
visible parts. A predictor maps the context encoder's output to the features of a
target encoder, which is an exponential moving average (EMA) of the context
encoder. This project trains a small ViT-S/16 version from scratch on SAYCam-S
or BabyView clips, in shuffled (IID) or recording (streaming) order. It also
evaluates the encoder on IntPhys 2 and on static-image probes, and releases
four checkpoints.

## Quickstart

Install the training dependencies and train a tiny model on synthetic clips on
CPU:

```bash
uv sync --extra train
uv run python -m egolearn.methods.vjepa2.train \
  --config-dir projects/vjepa2/configs --config-name smoke
```

The smoke recipe runs four updates and writes to `runs/vjepa2_smoke`. Append
`dry_run=true` to build the pipeline without training.

For real data, request access through the
[dataset access guide](../../docs/evaluation/data.md) and extract upright frames
at 5 fps. SAYCam frames go in `$SAYCAM_ROOT/S_<recording>/<name>_<seconds>.jpg`
and BabyView frames in `$BABYVIEW_ROOT/<child>/<recording>/frame_00000001.jpg`.
Then train the short SAYCam recipe on a GPU:

```bash
export SAYCAM_ROOT=/path/to/saycam_5fps
export WANDB_MODE=disabled
uv run python -m egolearn.methods.vjepa2.train \
  --config-dir projects/vjepa2/configs --config-name saycam
```

Use `--config-name babyview` with `BABYVIEW_ROOT` for BabyView. The real-data
recipes log to W&B online unless `WANDB_MODE` says otherwise, so the command above
disables logging. Set `WANDB_MODE=offline` to keep local W&B logs, or set
`WANDB_MODE=online` with `WANDB_ENTITY` and `WANDB_PROJECT` to upload runs to your
own project.

## Recipes

| Config | Entry point | Training |
| --- | --- | --- |
| `smoke` | `egolearn.methods.vjepa2.train` | Tiny model on synthetic clips, on CPU |
| `saycam`, `babyview` | `egolearn.methods.vjepa2.train` | Shuffled clips, batch 8, 1,000 updates |
| `full_saycam_iid`, `full_babyview_iid` | `egolearn.methods.vjepa2.full_train` | Shuffled clips, batch 32, same update count as streaming |
| `full_saycam_streaming`, `full_babyview_streaming` | `egolearn.methods.vjepa2.full_train` | Clips in recording order, batch 32, one pass |

All real-data recipes train a ViT-S/16 encoder and a 6-block, 192-wide predictor
on 16-frame clips at 128 × 128 pixels. A clip never crosses a recording boundary
or a gap in the timestamps. Training uses AdamW with learning rate 1e-4 and
weight decay 0.04, a constant EMA rate of 0.998 and BF16 forward passes. The
learning rate warms up for 100 updates in the short recipes, then follows a
cosine decay. Every run first saves its random initialization as `random.pth`,
which serves as the untrained baseline in evaluation.

The short recipes write `final.pth` and save snapshots every 250 updates, and
`resume=auto` with the same `run_dir` resumes one. The released checkpoints come
from the full recipes below.

## Full IID and streaming runs

The full recipes make one pass over every complete clip in the dataset. In
streaming order, the model trains on a FIFO window of the 32 newest clips, and
each update admits two new clips. IID runs shuffle the same clips each epoch and
run the same number of updates, which is about 16 epochs. The learning rate warms
up over 5% of the run, or at least 100 updates, then decays with a cosine.

```bash
uv run python -m egolearn.methods.vjepa2.full_train \
  --config-dir projects/vjepa2/configs --config-name full_saycam_streaming
```

Append `dry_run=true` to count the clips and print the number of updates on a
`[budget]` line. For a short check on real data, append
`preflight=true run_control.stop_after_steps=8`. The run checks one batch for
static clips and for gradient flow, trains eight updates under the full schedule
and exits with code 76. Rerunning it with `run_control.stop_after_steps=0`
continues training. To catch an incomplete copy of the data, set
`run_control.expected_frames` and `run_control.expected_clips` from your own
dataset inventory.

The runner pauses with exit code 75 at `run_control.max_seconds` (46 hours by
default) or on SIGUSR1, after saving `snapshot_latest.pth`. It also saves that
snapshot every 5,000 updates and keeps a `snapshot_epoch_NNN.pth` at each IID
epoch boundary. To resume, rerun the same command with the same `run_dir`. The
runner records the configuration, the clip inventory and the installed code in
`manifest.json` and refuses to resume if they changed, although loader settings
can change between attempts. `final.pth` is written only when the whole pass
finishes.

On a Slurm cluster, submit through the [cluster launcher](../../docs/cluster.md)
with an absolute `run_dir`, which the runner requires under Slurm. To requeue
paused jobs automatically, export `EGOLEARN_REQUEUE_ON_PAUSE=1` and pass
`--requeue`. For streaming runs, `loader.cache_frames=128 loader.affinity=true`
lets each data worker reuse decoded frames across updates.

## IntPhys 2

Prepare the IntPhys 2 `Main` split with the [IntPhys 2 guide](../../docs/evaluation/intphys2.md)
and set `INTPHYS2_ROOT` to it. Set `RUN_DIR` to your training run's `run_dir`,
here the short SAYCam recipe's default. Evaluate the trained model and its
random initialization with the same configuration:

```bash
export INTPHYS2_ROOT=/path/to/intphys2/Main
export RUN_DIR=runs/vjepa2_saycam
uv run python -m egolearn.evaluate --config-dir projects/vjepa2/configs \
  --config-name intphys2 checkpoint="$RUN_DIR/random.pth" output_json="$RUN_DIR/random-intphys2.json"
uv run python -m egolearn.evaluate --config-dir projects/vjepa2/configs \
  --config-name intphys2 checkpoint="$RUN_DIR/final.pth" output_json="$RUN_DIR/trained-intphys2.json"
```

The configuration scores all 506 video pairs at 128 pixels from 16 uniformly
sampled frames, with context lengths of 2 to 14 frames. Report `voe_acc` with
every `voe_acc_ctx<C>` value. This protocol differs from the IntPhys 2 guide's
64-frame default and from the paper's sliding window, and the random baseline
need not score 50%.

## Static-image transfer

The `image_evals` configuration measures how well the frozen EMA encoder
represents still images. It runs Labeled-S linear decoding, BabyView linear
decoding and novel-object shape bias. Each image is repeated for 16 frames, and
the encoder's output tokens are mean-pooled. Prepare the inputs and set four
environment variables:

| Variable | Input | Preparation |
| --- | --- | --- |
| `LABELED_S_ROOT` | Labeled-S images | [Labeled-S linear data](../../docs/evaluation/data.md#labeled-s-linear-data) |
| `LABELED_S_SPLIT` | Local copy of the canonical split file | [Labeled-S linear data](../../docs/evaluation/data.md#labeled-s-linear-data) |
| `BABYVIEW_LINEAR_MANIFEST` | `babyview_linear.json` | [BabyView object crops](../../docs/evaluation/data.md#babyview-object-crops) |
| `SHAPE_BIAS_MANIFEST` | Shape-bias stimuli manifest | [Shape bias](../../docs/evaluation/data.md#shape-bias) |

Then evaluate `final.pth`, and `random.pth` with a separate output path:

```bash
export RUN_DIR=runs/vjepa2_saycam
uv run python -m egolearn.evaluate --config-dir projects/vjepa2/configs \
  --config-name image_evals checkpoint="$RUN_DIR/final.pth" \
  output_json="$RUN_DIR/trained-image-evals.json"
```

These are static-image transfer scores. They do not test temporal understanding
and are not comparable to the paper's image evaluations.

## Released checkpoints

ViT-S/16 checkpoints from the full recipes, seed 0, are on the Hugging Face Hub.
Request access with the [checkpoint instructions](../../docs/evaluation/hub.md#checkpoint-access),
then set `HF_TOKEN` to a read token for the approved account.

| Dataset | IID | Streaming |
| --- | --- | --- |
| SAYCam-S | [Checkpoint](https://huggingface.co/egolearn/vjepa2-vits16-saycam-iid-seed0/tree/de8ed7eecf2fc31c69fee7ef155bae522dc63374) | [Checkpoint](https://huggingface.co/egolearn/vjepa2-vits16-saycam-streaming-seed0/tree/8b47fcb1149c97f412bb73257d1bc2b9f03ca0bb) |
| BabyView | [Checkpoint](https://huggingface.co/egolearn/vjepa2-vits16-babyview-iid-seed0/tree/d1429d2a790c35be69b5921afd1a44c0a0364ef8) | [Checkpoint](https://huggingface.co/egolearn/vjepa2-vits16-babyview-streaming-seed0/tree/f2ccedd42e06db40ec04b2b722300621f1aa91f5) |

Pass a release to the evaluation commands as a revisioned `checkpoint` reference,
for example:

```text
hf://models/egolearn/vjepa2-vits16-saycam-iid-seed0@de8ed7eecf2fc31c69fee7ef155bae522dc63374/model.safetensors
```

Each model card describes its training run. To export your own completed run in
the same format, run:

```bash
export RUN_DIR=runs/vjepa2_full_saycam_iid
uv run egl export "$RUN_DIR/final.pth" "$RUN_DIR/export"
```

## Differences from the paper

- The encoder is a ViT-S/16 at 128 pixels, trained with batches of 8 or 32 clips.
  This is far below the paper's model size, resolution and batch size, so the
  results are not comparable to its reported accuracy.
- The EMA rate is constant and the schedules are short.
- The code does not convert Meta's checkpoints. It has no multi-GPU training,
  action conditioning, V-JEPA 2.1 or activation checkpointing.
- Mask sampling follows [Meta's reference code](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)
  but draws from a different random stream.
