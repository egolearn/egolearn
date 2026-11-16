# StreamMAE on WalkingTours

Train StreamMAE on all ten original WalkingTours videos in the shared dataset
order. Follow the [dataset preparation guide](../../docs/datasets/walkingtours.md)
and set `WALKINGTOURS_ROOT` to its completed frame directory before training.
For the separate WT++ London source, use the [WT++ guide](wtpp.md).

## Requirements

Run from the repository root after installing the training dependencies with
`uv sync --extra train`. The recipe defaults to CUDA and BF16, requiring a GPU
with native BF16 support (compute capability 8.0 or later).

## Training

Read the expected frame count from the completed inventory, then start training
on a workstation or GPU compute allocation:

```bash
WALKINGTOURS_FRAMES=$(uv run python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_frames"])' \
    "$WALKINGTOURS_ROOT/inventory.json")
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_wt_original_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" \
    run_control.expected_frames="$WALKINGTOURS_FRAMES"
```

The default output directory is `runs/streammae_wt_original/seed_1337`, or
`$EGOLEARN_ARTIFACTS/streammae_wt_original/seed_1337` when that variable is set.
Choose a separate `run_dir` whenever you change the data, seed, or scientific settings.

## Recipe and checkpoints

The recipe uses seed 1337 and trains ViT-S/16 from scratch with masked
reconstruction. DataDrop retains 512 frames from a 2,048-frame FIFO, advancing
one frame per update until the full prepared stream is consumed.

It applies the [WT++ crop geometry and regularization](wtpp.md#protocol-and-snapshots)
to the original WT videos. The [BabyCL experiment](../babycl/docs/walkingtours.md)
uses the same frames and order, with a different architecture, objective, and
update budget. These runs do not match training compute or establish paper reproduction.

Training saves resumable state every 5,000 updates and at 20%, 40%, 60%, 80%,
and completion. See [training and resume](training.md) for the output files and
[checkpoint export and evaluation](README.md#checkpoint-export-and-evaluation)
for downstream use.

## Cluster runs

Run frame preparation on CPUs before submitting a GPU job, or use an `afterok`
dependency on the preparation job. The [general Slurm launcher](../../docs/cluster.md)
freezes the source and environment. For allocations shorter than the run, set
`run_control.max_seconds` below the wall limit and export
`EGOLEARN_REQUEUE_ON_PAUSE=1` when submitting through that launcher.

For a preflight, set `run_control.stop_after_steps=16`. The runner saves state
and exits with code 76. Set it to zero to resume with the full training budget.
Time-limit pauses save state and exit with code 75, which the launcher can requeue.

The specialized [MAE launcher](training.md#slurm-launcher) also accepts
`full_wt_original_streammae`; use its `MAE_*` environment and requeue settings.
