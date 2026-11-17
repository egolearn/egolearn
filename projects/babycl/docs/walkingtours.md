# BabyCL on WalkingTours

Train visual BabyCL on all ten original WalkingTours videos in one chronological
pass. Follow the shared [dataset preparation guide](../../../docs/datasets/walkingtours.md)
and set `WALKINGTOURS_ROOT` to the completed frame directory before training.

## Requirements

Run from the repository root with the training dependencies installed through
`uv sync --extra train`. The recipe defaults to CUDA. Its frozen segmenter needs
the torchvision ImageNet ResNeXt-50 weights on first use or in its cache, while
the trainable encoder starts from scratch.

## Training

Start training on a workstation or GPU compute allocation:

```bash
uv run python -m egolearn.train \
    --config-dir projects/babycl/configs --config-name walkingtours \
    stream.root_dir="$WALKINGTOURS_ROOT"
```

The default output directory is `runs/babycl_wt_original/seed_1337`, or
`$EGOLEARN_ARTIFACTS/babycl_wt_original/seed_1337` when that variable is set.
You can choose a different directory with `run_dir`, keeping experiments and seeds separate.

## Recipe

The recipe uses seed 1337 and consumes the complete prepared stream. It trains
ResNeXt-50 with instance contrast and temporal classification. A frozen ImageNet
ResNeXt-50 segmenter produces segments of approximately three minutes, with actual
boundaries determined by visual content.

| Setting | Value |
| --- | --- |
| Batch | Four segments × 32 frames × two views |
| Updates | Eight per admitted segment |
| Short memory | FIFO with 28 segments |
| Long memory | Reservoir with 140 segments |
| Weight checkpoints and rolling snapshots | Every 500 updates |
| Progress snapshots | 5%, 25%, 50%, and completion |

The buffer capacities approximate 10% and 50% of a nominal 14-hour stream
divided into three-minute segments. Segment lengths vary with visual content.
The [StreamMAE experiment](../../streammae/walkingtours.md) uses the same frames
and order, but its architecture, objective, and update budget differ.

## Cluster runs

Commit the intended code and use the [Slurm launcher](../../../docs/cluster.md)
to freeze the source and environment. Run preparation on CPUs before submitting
the GPU job, or use an `afterok` dependency on the preparation job.

BabyCL uses the standard training runner. For snapshot continuation, follow the
[resume instructions](../../../docs/cluster.md#snapshots-and-records).
It does not use the MAE runner's time-limit pause and automatic requeue controls.
