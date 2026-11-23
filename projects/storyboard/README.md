# Memory Storyboard

Memory Storyboard learns visual representations from a chronological stream by
grouping frames into temporal segments and replaying them from two memory tiers.
This project implements the method from the
[Memory Storyboard paper](https://arxiv.org/abs/2501.12254) with a SAYCam
configuration.

## Training

You need [SAYCam access](../../docs/evaluation/data.md#saycam-access) and raw
5 fps frames in the [BabyCL frame layout](../babycl/docs/training_data.md#frames).
Unlike BabyCL, the recipe expects the raw upside-down frames and flips them
during loading. Install the training dependencies with `uv sync --extra train`,
then validate the config before you train on a GPU:

```bash
export SAYCAM_ROOT=/path/to/saycam_5fps
uv run egl train projects/storyboard/configs/saycam.yaml \
  "stream.root_dir=$SAYCAM_ROOT" run_dir=runs/storyboard-example dry_run=true
uv run egl train projects/storyboard/configs/saycam.yaml \
  "stream.root_dir=$SAYCAM_ROOT" run_dir=runs/storyboard-example
```

## Method and configuration

The learner's own backbone segments the stream. Each time the short memory
turns over, the method reclusters its frames, revises their labels and merges
classes with close prototypes.

| Setting | Default |
| --- | --- |
| Short memory | 5,000-frame FIFO, admitting every eighth frame |
| Long memory | 45,000-frame weighted reservoir, admitting every frame |
| Batch | 64 short-memory and 448 long-memory frames |
| Objectives | SimCLR instance discrimination on the batch, and supervised contrastive learning over long-memory frames that treats frames with the same segment label as positives |
| Updates | One per 64 stream frames, reusing the batch for extra updates |
| Optimizer | SGD, constant learning rate 0.05 |
| Input | 112-pixel crops |

## Evaluation

The released Memory Storyboard ImageNet evaluation uses 112-pixel inputs, so set
the size when you evaluate a checkpoint with the
[standalone evaluator](../../docs/evaluation.md):

```bash
uv run egl eval imagenet_linear_eval \
  checkpoint=runs/storyboard-example/final.pth data_root=/path/to/imagenet \
  benchmarks.0.size=112
```

## Differences from the original code

- The config is a code-level port. Numerical parity with the original code has
  not been established, so match the full evaluation protocol before you
  compare scores with the paper.
- Training runs in float32, while the original used mixed precision. Add
  `+driver.amp.dtype=float16` to train in mixed precision.
- The recluster trigger is checked after every frame rather than every update.
  It fires at the same threshold crossing, up to 64 frames earlier.

The [architecture page](../../docs/design/architecture.md#method-differences)
compares Memory Storyboard with BabyCL.
