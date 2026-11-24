# Benchmark suites

A benchmark suite runs one training or evaluation config over a fixed set of
seeds and summarizes the results as a table of means and standard deviations.
Use a suite when you want a comparison you can rerun later. It pins the config,
the seeds and the SHA-256 of each input, and it refuses to build a table from
incomplete runs.

Suites run with `egl bench`, which needs the `train` extra
(`uv sync --extra train`).

## Quickstart

[`benchmarks/toy.yaml`](../benchmarks/toy.yaml) trains the synthetic
`toy_streaming` config for two steps with each of two seeds, on CPU:

```yaml
version: 1
name: toy
entry: train
config: toy_streaming
seeds: [7, 8]
overrides:
  - ++driver.max_steps=2
inputs: {}
metrics:
  train/last_step/loss: min
  train/steps: max
```

Run it twice, the second time with a lower learning rate, and compare the two
runs:

```bash
uv run egl bench run benchmarks/toy.yaml --output runs/toy-first
uv run egl bench run benchmarks/toy.yaml --output runs/toy-second \
  --override solver.groups.0.lr=0.0005
uv run egl bench table runs/toy-second/manifest.json --baseline runs/toy-first/manifest.json
```

Each output directory holds a `manifest.json` and one `seed-<seed>/` directory
per seed with its frozen config, log and `record.json`. The output must be a
new directory, outside the checkout or in an ignored path such as `runs/`, so
retry a failed suite in a new directory. The table prints one row per metric
with its mean, standard deviation, baseline mean and change.

## Suite file

| Field | Meaning |
| --- | --- |
| `version` | Always `1` |
| `name` | Suite name; a baseline must have the same name |
| `entry` | `train` or `eval` |
| `config` | A packaged config name, such as `toy_streaming`, or a `.yaml` path relative to the suite file |
| `seeds` | Seeds to run, one run each |
| `overrides` | Hydra `key=value` overrides applied to every seed |
| `inputs` | Pinned input files, keyed by the config key they set |
| `metrics` | Metric keys to report, each with direction `min` or `max` |

Metric keys depend on the entry. Evaluation metrics are
`<benchmark name>/<metric>`, such as `intphys2/voe_acc`. Training metrics are
`train/last_step/<log key>` for the last training step, `train/steps`, and
`probe/<probe name>/<metric>` for the last result of each training probe.

Overrides cannot change the seed, a pinned key, output paths, resume or dry-run
mode, because every suite run starts fresh from its pinned inputs.

### Pinned inputs

Each entry under `inputs` sets a config key to a path and pins its content.
An evaluation suite must pin `checkpoint`:

```yaml
inputs:
  checkpoint:
    path: /path/to/final.pth
    sha256: <sha256 of final.pth>
  benchmarks.0.data_root:
    path: /path/to/intphys2/Main
    sha256: <sha256 of Main/checksums.sha256>
```

Paths may be absolute, relative to the suite file, or `hf://` references.
Compute the digest that matches the kind of input:

- A file: `sha256sum FILE`.
- A directory: write an inventory with
  `uv run python -m egolearn.hub checksum DIR`, then pin
  `sha256sum DIR/checksums.sha256`. Every listed file is then verified.
- An exported `model.safetensors`: the checkpoint `sha256` that `egl eval`
  records for it, which covers the weights and `model.json` together.

The runner checks every pin before planning and again before and after each
seed, and a changed or missing input fails the run. Pin every data file and
manifest that your comparison depends on, since the runner cannot tell which
other config values point at data.

## Tables

`egl bench table MANIFEST` prints a Markdown table, and `--format csv` or
`--format json` gives other formats. A metric has a mean only when every seed
completed with a finite value, and the seeds that did not are listed under
the table.

With `--baseline OTHER_MANIFEST`, the table adds the baseline mean and the
change, and flags every change in the worse direction. This is not a
significance test. A baseline must have the same suite name, seeds and pinned
inputs, so use it to compare code revisions or overrides on the same data.
The command exits with status 1 when a seed is missing or a metric regresses.

Tables accept only runs from a clean, committed checkout. If you need an
exploratory table from uncommitted code or a wheel install, add
`--allow-unverified`, which marks the table as exploratory.

## Cluster runs

With `--submit`, `egl bench run` plans the suite and submits one Slurm array
task per seed through the cluster launcher. Planning composes every config and
hashes every pinned input, so run it in a CPU allocation, not on a login node,
from a clean checkout set up as in [Cluster jobs](cluster.md). Pass Slurm
options for the array tasks with `--sbatch=OPTION`:

```bash
uv run egl bench run benchmarks/babycl_saycam.yaml \
  --output "$EGOLEARN_ARTIFACTS/babycl-suite" --submit --sbatch=--gres=gpu:1
```

`EGOLEARN_ARTIFACTS` is the output root that the cluster `env.sh` exports. Keep
the output directory and every pinned input on storage that the compute nodes
can read.

## BabyCL SAYCam suite

[`benchmarks/babycl_saycam.yaml`](../benchmarks/babycl_saycam.yaml) trains
BabyCL on SAYCam with three seeds and reports the final Labeled-S 4AFC and
linear decoding accuracies. Its linear probe uses the canonical 26-class split,
not the BabyCL paper's per-category split, so its scores are not the paper's
numbers. Get the data as described in [dataset access](evaluation/data.md),
and set the path and digest variables that the suite file lists, such as
`SAYCAM_ROOT` and `SAYCAM_SHA256`.

## Run records

Training and evaluation write a `record.json` for every attempt, inside or
outside a suite. With the default `record_json=auto`, training writes
`<run_dir>/records/<id>/record.json`, and evaluation writes
`runs/eval/records/<id>/record.json` in the working directory. Set
`record_json=/path/to/new.json` for a specific file, which must not exist yet,
or `record_json=null` to turn records off.

A record holds the run's status, seed, resolved config and metrics, the code
version and Git state, and the SHA-256 of each known input file. Checkpoints
carry the identity of the training run that produced them, and an evaluation
record keeps it next to its own. The status stays `running` if the process is
killed. Records contain local paths, so review them before you share them.
