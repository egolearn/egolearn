# Matched MAE ablations

Use this suite to compare data ordering and StreamMAE additions on the same
prepared SAYCam or BabyView inventory. Runs match retained examples and optimizer
updates, with the same learning rate at each update, and no downstream results are
reported for this suite.

## Recipes

Select an ablation and supply your prepared frame root:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name matched_saycam \
    ablation=vanilla stream.root_dir="$SAYCAM_ROOT" seed=1337
```

Use `matched_babyview` with `stream.root_dir="$BABYVIEW_ROOT"` for BabyView.
The default output directory includes the dataset, ablation, and seed, so
different recipes keep separate snapshots and records.

| `ablation` | Training recipe | FIFO capacity | Retained examples |
| --- | --- | ---: | ---: |
| `vanilla` | Streaming MAE, ordinary crop and flip | 512 | 512 |
| `iid` | Globally shuffled MAE, ordinary crop and flip | None | 512 |
| `datadrop` | Streaming MAE with fixed-cardinality DataDrop | 2,048 | 512 |
| `drop_path` | DataDrop plus encoder drop path at 0.25 | 2,048 | 512 |
| `color_jitter` | Drop path plus color jitter with probability 0.5 | 2,048 | 512 |
| `random_crops` | Color jitter plus random two-stage cropping | 2,048 | 512 |
| `motion` | Two-stage cropping with four motion-ranked candidates | 2,048 | 512 |

The additions are cumulative from `datadrop` onward, with `random_crops` and
`motion` differing only in the number of candidate crops used for motion ranking.
The `datadrop` recipe provides the MAE baseline with DataDrop used in the paper's
main comparison, with `vanilla` retained as a separate ablation.

## Budget

For N source frames, reference window R, and stride S, each run performs
`ceil((N - R + S) / S)` updates. Defaults are R = 2,048 and S = 32,
with 512 retained examples per update. The final streaming advance consumes
the remaining frames even when fewer than S remain.

The reference window determines when training begins, independently of FIFO
capacity. All streaming variants start their first update after consuming
2,048 frames. Vanilla MAE keeps only the newest 512 frames during this initial
fill, so some early frames never reach its model. DataDrop samples 512 distinct
frames from the current 2,048-frame window before augmentation or model execution.

IID uses the same frame inventory and update budget. It reshuffles each epoch
and drops incomplete batches, so its frame-consumption count includes repeated
draws and does not measure unique source coverage. Matched updates and retained
examples do not imply identical sampled frames, wall time, or crop-processing cost.

Set `run_control.reference_window_size`, `run_control.stride`, and
`driver.warmup.admitted` together, with warmup equal to reference window minus stride.
The reference window must be at least the FIFO capacity, and stride cannot exceed
that capacity. Set `run_control.expected_frames` from an independent inventory
to reject incomplete dataset mounts.

Full-pass learning-rate schedules use optimizer updates in both modes.
Changing the reference window changes the schedule length and requires a new
run directory. Existing `full_*` configurations retain their own window budgets;
use the `matched_*` suite when comparing different window sizes.

## Records

Each run writes the resolved configuration, budget, and source digest to
`manifest.json`. Automatic records under `records/` include the seed, code
provenance, resolved update count, and final metrics. Local source checkouts record
their commit and dirty status, while jobs launched through `scripts/slurm/submit`
use its archived provenance. The standalone `mae.sbatch` launcher does not supply that
metadata for non-editable installations, so preserve their source revision separately.
Keep these files with the checkpoint when collecting downstream results.

Full-pass `final.pth` and `snapshot_*.pth` files contain training state, so
evaluate snapshots you created with `trusted_snapshot=true`, or export their
weights to safetensors before evaluation.

`metrics.jsonl` separates the following quantities:

| Metric | Meaning |
| --- | --- |
| `train/consumed_frames` | Chronological admissions for streaming, or repeated batch draws for IID |
| `train/window_size` | FIFO capacity, or zero for IID |
| `train/retained_examples` | Examples sent to the model in the current update |
| `train/optimizer_updates` | Completed optimizer updates across attempts |
| `train/examples_seen` | Total retained examples across updates, including repeats |
| `throughput/frames_per_second` | Consumed frames per second in this attempt |
| `throughput/examples_per_second` | Retained examples per second in this attempt |
| `throughput/updates_per_second` | Optimizer updates per second in this attempt |

Throughput starts from the restored counters after resume and includes initial
buffer filling on a fresh run. Loss, learning rate, and these counters also reach
the ordinary training log and W&B when enabled. Resume archives an abandoned
metrics tail before recording replacement updates.

## Evaluation protocol

Use the same seeds, evaluation data, and checkpoint milestones for each variant.
Record frozen probes and full-network fine-tuning separately in any results table.
The existing [evaluation commands](../../docs/evaluation.md) provide frozen probes,
whose scores do not reproduce the paper's primary full-fine-tuning protocol.
These matched recipes control pretraining budgets without claiming numerical
equivalence to the paper's experiments.
