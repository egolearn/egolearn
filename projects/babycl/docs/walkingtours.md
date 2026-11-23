# BabyCL on WalkingTours

This page shows how to train visual BabyCL on the ten original WalkingTours
videos in one chronological pass. Prepare the frames first with the
[WalkingTours dataset guide](../../../docs/datasets/walkingtours.md), which also
sets `WALKINGTOURS_ROOT`.

## Training

Install the training dependencies with `uv sync --extra train`. Validate the
config, then train on a GPU:

```bash
uv run egl train projects/babycl/configs/walkingtours.yaml \
    "stream.root_dir=$WALKINGTOURS_ROOT" dry_run=true
uv run egl train projects/babycl/configs/walkingtours.yaml \
    "stream.root_dir=$WALKINGTOURS_ROOT"
```

The run uses seed 1337 and writes to `runs/babycl_wt_original/seed_1337`, or to
the same path under `$EGOLEARN_ARTIFACTS` when you set that variable. Pass
`seed` and `run_dir` to keep other runs apart. The run saves checkpoints and
resumable snapshots every 500 updates, and you continue an interrupted run by
repeating the command with `resume=auto`. Submit long runs with the
[cluster launcher](../../../docs/cluster.md).

## Differences from SAYCam

The videos have no transcripts, so the recipe trains the instance and temporal
objectives without image–text pairs. Each batch holds four segments of 32
frames, and the run consumes the whole stream. The segment memory is a
28-segment FIFO that overflows into a 140-segment reservoir, about 10% and 50%
of a 14-hour stream in three-minute segments.

[StreamMAE on WalkingTours](../../streammae/walkingtours.md) reads the same
frames in the same order with a different model, objective and update budget.
