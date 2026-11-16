# MAE training and resume

Use this guide to monitor, pause, and resume the `full_*` and `matched_*`
recipes launched through `egolearn.methods.mae.train`. Choose a recipe and prepare
your data using the [MAE and StreamMAE overview](README.md) before starting.

## Budget and preflight

The full-pass runner counts the selected frames and derives the optimizer budget
from the recipe's reference window and stride. Check the `[budget]` line printed
at startup and `budget.updates` in `manifest.json` for your run's actual count.
The [method guide](method.md#update-budgets) explains the calculation and the
differences between the full-pass and matched-ablation recipes.

Set `run_control.expected_frames` from an independent dataset inventory to reject
an incomplete mount. The selected inventory must contain at least one reference
window. Use `dry_run=true` to count frames and check pipeline construction without
training. This still requires the dataset and the recipe's configured device.

For a short training check, append `run_control.stop_after_steps=10` to your
full-pass command. It saves resumable state and exits with code 76, keeping the
full learning-rate schedule and budget. Rerun the same command with
`run_control.stop_after_steps=0` to continue training in the same run directory.

## Outputs

Keep the complete run directory together, including its manifest and snapshots.
The default full-pass recipes write the following files:

| File | Contents |
| --- | --- |
| `manifest.json` | Scientific configuration, frame and update budgets, source digest, and runtime versions |
| `metrics.jsonl` | Loss, learning rate, frame consumption, retained examples, and throughput |
| `status.json` | Current state, completed steps, and budget |
| `records/` | Per-attempt configuration, provenance, status, and metrics |
| `attempts/` | Loader and prefetch settings used by each attempt |
| `snapshot_latest.pth` | Rolling resumable state, saved initially, every 5,000 updates, at a controlled pause, and on completion |
| `final.pth` | Resumable state written only when the full budget completes |

SAYCam and BabyView full-pass recipes also archive `snapshot_epoch_NNN.pth`
at each equivalent IID epoch, including a short final epoch. Streaming runs use
the same update thresholds as IID. WalkingTours recipes instead archive
`snapshot_020pct.pth`, `snapshot_040pct.pth`, `snapshot_060pct.pth`,
`snapshot_080pct.pth`, and `snapshot_100pct.pth`.

See the [ablation metrics](ablations.md#records) for the distinction between
consumed frames and retained training examples. To use a saved model downstream,
follow [checkpoint export and evaluation](README.md#checkpoint-export-and-evaluation).

## Resume

Rerun the same recipe with the same `run_dir`. The runner automatically loads
`snapshot_latest.pth` and rejects explicit `resume=<path>` overrides. A directory
containing checkpoints must already have a matching `manifest.json`.

Use a new directory when changing the data, seed, model, or scientific settings.
To move an existing run, copy its complete directory and update `run_dir`.
The recorded source/runtime fingerprint and scientific recipe must still match.
Keep the prepared frames unchanged throughout training and resume.

You can change W&B settings, checkpoint frequency, run-record settings, and loader
settings between attempts. Loader settings include worker count, packed transfers,
pinning, decoded-frame cache capacity, and IID prefetch. Streaming prefetch and
gradient accumulation remain disabled for these recipes. CUDA execution is not
guaranteed to be bitwise deterministic.

The default full-pass loader uses eight workers. Each streaming worker caches up
to 1,024 decoded frames, or 256 for WalkingTours, while IID disables this cache.
Reduce worker count or `loader.cache_frames` if decoded images exceed available RAM.

If an archived milestone is ahead of the restored rolling snapshot, it moves to
`abandoned_snapshots/<attempt>/`. Training recreates it when that threshold is
reached again. Existing milestones at or before the restored update remain
unchanged, and missing past milestones are not reconstructed.

## Time limits and pauses

`run_control.max_seconds` defaults to 46 hours. It counts preparation and replay
as well as training, so leave time before the scheduler's wall limit for saving
state. A SIGUSR1 signal also requests a pause.

Preparation or replay must finish before the runner can save a resumable
snapshot. During training, a pause takes effect after a completed update.
The time limit is therefore a soft deadline.

| Exit code | Meaning |
| --- | --- |
| `0` | Training completed |
| `75` | State saved for a time-limit or signal pause; eligible for scheduler requeue |
| `76` | State saved after `run_control.stop_after_steps`; preflight finished |

Both pauses produce a `paused` run record containing the exit code and completed
step count. Exceptions produce failed records, and nonfinite metrics retain their
`nonfinite` status.

## Slurm launcher

`scripts/slurm/mae.sbatch` requests one GPU, 16 CPUs, 128 GB of RAM, and 48 hours.
It accepts the full-pass and matched-ablation recipe names from the overview.
Supply your account, partition, and Slurm log paths when submitting, and set:

| Variable | Value |
| --- | --- |
| `MAE_SOURCE` | Frozen source directory containing the project configs |
| `MAE_PYTHON` | Absolute Python path with a non-editable EgoLearn installation built from that source |
| `MAE_IMAGE` | Apptainer image |
| `MAE_BINDPATH` | Bind paths for the source, environment, input data, and outputs |
| `MAE_OVERLAY_LIST` | File containing one read-only archive path per line; use an empty file for ordinary frame directories |
| `MAE_RUN_ROOT` | Persistent output root |

Pass the config name followed by Hydra overrides, including the mounted frame root:

```bash
sbatch scripts/slurm/mae.sbatch full_saycam_streaming \
    stream.root_dir="$SAYCAM_ROOT"
```

Full-pass outputs go to `$MAE_RUN_ROOT/<config>`. Matched-ablation outputs include
the ablation and seed, as shown in the [cluster example](ablations.md#cluster-runs).
Pass `run_dir=/path/to/run` to choose another persistent directory, and use
distinct directories for different experiments and seeds.

The launcher requeues only exit code 75. Set `MAE_REQUEUE=0` to disable automatic
requeue, including for preflight jobs. Keep the source snapshot and installed
environment unchanged so later edits cannot alter a queued or resumed run.
