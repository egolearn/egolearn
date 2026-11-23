# Command line

`egl` is the command that egolearn installs, and this page shows the commands
you run most often. Each command lists its options with `--help`, and the
[train and eval reference](reference/cli.md) documents the Hydra interface of
`egl train` and `egl eval`.

Run `uv run egl` from your checkout after `uv sync --extra train`, or call
`egl` directly in an activated environment, where `egolearn` is an alias for
the same command.

| Command | Task | Guide |
| --- | --- | --- |
| `egl train` | Train from a config | [Configuration and overrides](#configuration-and-overrides) |
| `egl eval` | Evaluate a saved model | [Checkpoint evaluation](evaluation.md) |
| `egl export` | Save model weights with the metadata to rebuild the model | [Exporting weights](#exporting-weights) |
| `egl data` | List, prepare, fetch and verify datasets | [Data](#data) |
| `egl doctor` | Check dependencies and the runtime | [Diagnostics](#diagnostics) |
| `egl submit` | Submit a job to Slurm | [Cluster jobs](cluster.md) |
| `egl bench` | Run a benchmark suite across seeds | [Benchmark suites](benchmark_suites.md) |
| `egl publish` | Publish evaluated checkpoints to the results feed | [Publishing results](publishing_results.md) |

## Configuration and overrides

`egl train` and `egl eval` take a config followed by Hydra overrides. The
config is either a packaged name, such as `toy_streaming`, or the path to a
`.yaml` file:

```bash
uv run egl train toy_streaming seed=42 dry_run=true
uv run egl train projects/babycl/configs/saycam_default.yaml --cfg job
uv run egl train toy_streaming --multirun seed=1,2 dry_run=true
```

`key=value` changes a value, `+key=value` adds a key that the config does not
declare, and `~key` removes one. `--cfg job` prints the composed config
without running anything, and `--multirun` runs once for each value in a
comma-separated list. The [packaged configs](reference/configs.md) page lists
every name you can pass. A config file of your own can start from the packaged
defaults with `defaults: [/config, _self_]` and override only what it changes.

Training configs turn Weights & Biases logging off. To log online, export
`WANDB_API_KEY` and name a project. Runs go to your default W&B entity unless you
set `wandb.entity` or `WANDB_ENTITY`:

```bash
uv run egl train toy_streaming wandb.mode=online wandb.project=YOUR_PROJECT
```

## Exporting weights

`egl export CHECKPOINT OUT_DIR` turns a training checkpoint into a new
directory that holds the weights in `model.safetensors` and the config that
rebuilds the model in `model.json`. Evaluation,
[initialization](initialization.md) and the Hugging Face Hub accept this
directory, and [Models and Hub data](evaluation/hub.md) shows how to upload it.
To export a full training snapshot that you created yourself, add
`--trusted-snapshot`, as [checkpoint trust](../SECURITY.md#model-export-and-checkpoint-trust)
explains.

### Toy model export

This example trains the toy model for three steps, exports it and loads it
back, all on the CPU and without any dataset:

```bash
uv run egl train toy_streaming run_dir=runs/toy-export \
  +driver.max_steps=3 +hooks.checkpoint_every=100
uv run egl export runs/toy-export/final.pth exports/toy
uv run python - <<'PYTHON'
import torch
from egolearn.evaluation import load_model

model = load_model("exports/toy", device="cpu")
with torch.no_grad():
    features = model.encode("vision", torch.zeros(1, 3, 64, 64)).pooled
print(features.shape)
PYTHON
```

Setting `hooks.checkpoint_every` turns on checkpointing, so the run writes
`final.pth` when it ends. The last command prints `torch.Size([1, 64])`. To
run the example again, first remove `exports/toy`, since `egl export` refuses
an existing directory.

## Data

`egl data` lists the datasets egolearn knows about and runs their preparation
and download steps:

```bash
uv run egl data list
uv run egl data prepare shape-bias
uv run egl data fetch shape-bias --revision v1
```

`list` shows each dataset's access tier and whether egolearn can prepare it
locally or fetch it from the Hub, and [dataset access](evaluation/data.md)
explains how to obtain each dataset. `prepare NAME --help` lists a preparer's options. Data goes
under `$EGOLEARN_DATA`, or `./data` when it is unset, and neither `prepare`
nor `fetch` writes over an existing collection by default.

## Diagnostics

`egl doctor` checks the interpreter, the virtual environment, the installed
extras, CUDA and the data root:

```bash
uv run egl doctor --extra train --require-cuda
```

It prints one `ok`, `warn` or `fail` line per check and exits with status 1 if
any check fails. Add `--cluster` to check the storage settings of the
[Slurm launcher](cluster.md), and `--require-env NAME` to check that a
variable is set without printing it. On a cluster, run doctor inside a job, as
[cluster diagnostics](cluster.md#diagnostics) shows.
