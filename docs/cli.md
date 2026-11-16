# Command line

Use `uv run egl` from your checkout after `uv sync --extra train`.
In an activated environment, you can run `egl` directly; `egolearn` is an equivalent
installed alias. Each command accepts `--help`, and `uv run egl --version` prints
the installed version.

| Command | Task | Guide |
| --- | --- | --- |
| `train` | Train from a packaged config or YAML file | [Configuration and overrides](#configuration-and-overrides) |
| `eval` | Evaluate a saved model | [First checkpoint evaluation](evaluation/first_evaluation.md) |
| `export` | Save model weights and construction metadata | [Toy model export](#toy-model-export) |
| `data` | Discover, prepare, fetch, and verify datasets | [Data](#data) |
| `doctor` | Check dependencies and runtime settings | [Diagnostics](#diagnostics-and-cluster-submission) |
| `submit` | Submit a clean checkout to Slurm | [Cluster submission](#diagnostics-and-cluster-submission) |
| `bench` | Run comparisons across models and seeds | [Benchmark suites](#benchmark-suites) |
| `publish` | Publish evaluated checkpoints to the website feed | [Results publication](publishing_results.md) |

Run Python commands on a workstation or compute allocation. On cluster login
nodes, use the shell launcher described in the [cluster guide](cluster.md).

## Configuration and overrides

`egl train` and `egl eval` accept a packaged config name or an existing `.yaml`
file as their first argument. File paths are relative to the current directory.
`egl eval` without a config selects the packaged `evaluate` defaults.

```bash
uv run egl train toy_streaming seed=42 dry_run=true
uv run egl train projects/babycl/configs/saycam_default.yaml --cfg job
uv run egl eval path/to/suite.yaml checkpoint=runs/example/final.pth
```

Arguments after the command belong to Hydra: `--help`, `--cfg job`, `--multirun`,
`key=value`, `+key=value`, and `~key` keep their usual meaning. For example,
`egl train toy_streaming --multirun seed=1,2 dry_run=true` launches two dry runs.

Explicit files search their own directory first, then the installed config tree,
then entries supplied through `hydra.searchpath`. They can inherit packaged
defaults such as `defaults: [/config, _self_]` while overriding local components.

The shorthand expands to native Hydra config flags before any remaining arguments.
Later flags take precedence, including Hydra's accepted abbreviations:
`egl train toy_streaming --config-n=config --cfg job` selects `config`.
Native `--config-path`, `--config-name`, and `--config-dir` flags are also accepted;
supplying them later replaces the corresponding shorthand-generated setting.

The existing `python -m egolearn.train` and `python -m egolearn.evaluate` entry
points remain supported.

Shared training configs disable W&B by default. To log online, authenticate with
W&B and set your own destination, for example
`wandb.mode=online wandb.entity=YOUR_ENTITY wandb.project=YOUR_PROJECT` after the
training command. Override both destination fields because the shared entity
default belongs to the lab. Use `wandb.mode=offline` for local logs or
`wandb.mode=disabled` for no W&B logging. V-JEPA 2 uses environment variables as
shown in its [method guide](../projects/vjepa2/README.md#run).

## Toy model export

You can train, export, and reload a model on CPU without downloading any data.
Use a new run and export directory so existing artifacts remain identifiable:

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

The checkpoint hook writes `final.pth` when training ends, even before its periodic
100-step interval. The export contains `model.safetensors` and `model.json`, and
the final command prints `torch.Size([1, 64])`. This checks model loading and
encoding; benchmark scores require the data described in the [evaluation guide](evaluation.md).

## Exporting weights

Install the `train` extra, then export a checkpoint into a new directory:

```bash
uv run egl export runs/example/final.pth exports/example
uv run egl eval imagenet_linear_eval checkpoint=exports/example data_root=/path/to/imagenet
```

`egl export` takes ordinary arguments, not Hydra overrides. `egl export --help`
lists its options. It writes `model.safetensors` and `model.json` with construction
metadata, the export contract, source filename/SHA-256/step, and the export code
revision. A source checkpoint must already contain construction metadata and an
export contract. Existing output directories are refused, including empty ones.
For a full snapshot you created and trust, add `--trusted-snapshot`; see
[checkpoint trust](../SECURITY.md#model-export-and-checkpoint-trust).

Publish the exported directory with:

```bash
uv run python -m egolearn.hub push exports/example --name example \
  --repo YOUR_NAMESPACE/example --repo-type model --license mit
```

Replace `YOUR_NAMESPACE` with an account or organization where you have write
permission. The [Hub guide](evaluation/hub.md#publication) explains overwrite behavior.

Model uploads validate the pair and reject additional payloads, including `.pth`
files. Dataset uploads retain their existing collection format. Snapshots stay
in the local run directory.

`egl publish prepare` and `egl publish push` publish evaluated checkpoints with
per-seed provenance and update the website's benchmark feed. See
[publishing results](publishing_results.md) for the manifest and deployment settings.

## Data

List dataset entries and inspect preparation options before downloading inputs:

```bash
uv run egl data list
uv run egl data list --json
uv run egl data prepare shape-bias --help
```

Catalog access labels describe dataset categories. `FETCH` indicates that the CLI
supports downloading a collection, not that the repository is public or your
account has permission. See [hosted collection access](evaluation/hub.md#packaged-collections)
and [dataset access and preparation](evaluation/data.md) before choosing a source.

### Local preparation

Prepare shape-bias stimuli without hosted-dataset access, or supply the inputs
required by another preparer. In the BabyView example, replace both input paths
with your extracted crops directory and training vocabulary:

```bash
uv run egl data prepare shape-bias
uv run egl data --root /path/to/data prepare babyview-4afc \
  --data_root /path/to/babyview_crops --vocab_json /path/to/vocab.json
```

Fetch and preparation share `$EGOLEARN_DATA`, which defaults to `./data`.
Put `--root DIR` before the data subcommand to override it. Both commands use
`ROOT/<name>` for the collection, while shape-bias source assets live under
`ROOT/.sources/shape-bias`. Explicit source and output options override catalog defaults.

Preparation refuses an existing default output. Choose an explicit output option
only when you intend to use that preparer's overwrite behavior. Help and spatial
`--validate-only` work with existing outputs. `egl data prepare NAME --help` works
without training extras, but preparation itself needs the dependencies and assets
listed in the [benchmark guide](evaluation.md#benchmark-selection).

### Hosted downloads

With access to the hosted shape-bias repository, fetch and verify its `v1` collection:

```bash
uv run egl data fetch shape-bias --revision v1
uv run egl data verify "${EGOLEARN_DATA:-data}/shape-bias"
```

Use a commit ID instead of a tag when you need an immutable revision.
`--repo org/name` selects another published collection for a fetchable entry.
Missing checksums require an explicit `--allow-unverified`, and corrupt checksums
fail. Fetch preserves an existing collection unless you pass `--replace`.
Restricted images and upstream-licensed datasets have no automatic Hub fetch.

### Preparation modules

Each preparer also runs directly as an installed Python module:

| `egl data prepare` name | Module command |
| --- | --- |
| `labeled-s-4afc` | `python -m egolearn.evaluation.benchmarks.labeled_s_4afc.prepare` |
| `babyview-4afc` | `python -m egolearn.evaluation.benchmarks.babyview_4afc.prepare` |
| `shape-bias` | `python -m egolearn.evaluation.benchmarks.shape_bias.prepare` |
| `spatial-relations` | `python -m egolearn.evaluation.benchmarks.spatial_relations.prepare` |
| `spatial-monitoring` | `python -m egolearn.evaluation.benchmarks.spatial_relations.subset` |
| `grasp` | `python -m egolearn.evaluation.benchmarks.grasp.prepare` |

Append `--help` to a module command to see its arguments and options.
Direct module calls do not apply the CLI's shared-root defaults or its
existing-output check. Choose a new output path when preparing data; shape-bias
and BabyView can overwrite files in an existing output directory.

`data verify DIR` checks the collection's `checksums.sha256` without writing files.
It reports missing, altered, or unlisted payloads. Legacy preparation outputs have
their own benchmark validators and may not contain collection checksums; a missing
checksum file fails verification. The existing `python -m egolearn.hub checksum DIR`
command can create collection checksums after preparation.

## Diagnostics and cluster submission

Run these commands on a workstation or in a compute allocation:

```bash
uv run egl doctor
uv run egl doctor --extra train --require-cuda
uv run egl doctor --cluster --require-env WANDB_API_KEY
uv run egl doctor --submit --skip-cuda
uv run egl submit --gres=gpu:1 --time=01:00:00 -- egl train toy_streaming
```

Doctor prints one `ok`, `warn`, or `fail` line per check. Failures return status 1;
warnings alone return 0. It checks the interpreter, virtual environment, installed
extras, CUDA availability, data-root readability and requested environment variable
names.

CUDA is queried in a bounded child process without allocating a tensor;
`--skip-cuda` skips that query. Environment variable values are never printed.

Cluster checks activate with `--cluster` or inside a Slurm job. They share required
and optional storage-variable names with the launcher. Host checks resolve symlinks
before rejecting paths under the host's `$HOME`. Inside an Apptainer container,
`$HOME` is scratch by design: paths must still be absolute, but may be under that
container home.

Use `--submit` on a host that will submit jobs, including a CPU
planning allocation. It implies `--cluster` and checks Bash, Git, `sbatch`,
`iconv`, `tar`, and uv. Without `--submit`, cluster diagnostics check Apptainer
on compute hosts, Bash/Git/`sbatch` outside allocations, and no host tools inside
a job container. Doctor creates no directories, downloads no packages and changes
no settings.

On login nodes use the shell launcher's checks or submit doctor to a CPU/GPU
allocation as shown in [cluster diagnostics](cluster.md#diagnostics).
`--skip-cuda` skips the torch probe, but doctor itself still runs Python.

Submit forwards its arguments to the [cluster launcher](cluster.md), including
`--sweep`, `--commands`, and `--dry-run`. Run it from a clean EgoLearn checkout root
after sourcing `env.sh`: that checkout is the artifact the launcher archives.
The installed CLI can live elsewhere, but a wheel alone cannot supply a checkout
to archive. `--help` works outside a checkout. Use `scripts/slurm/submit` directly
on login nodes, where Python commands are prohibited. No shared lab package is
required; the existing launcher is the sole submission implementation.

## Benchmark suites

```bash
uv run egl bench run benchmarks/toy.yaml --output runs/toy-suite
uv run egl bench table runs/toy-suite/manifest.json
uv run egl bench table runs/toy-suite/manifest.json --format json
```

The [suite guide](benchmark_suites.md) defines pinned inputs, per-seed records,
baseline comparisons and failure handling. Local execution is sequential; `--submit`
creates one Slurm array task per seed. The toy suite uses synthetic data and CPU.
Follow [cluster planning in a CPU job](benchmark_suites.md#cluster-planning-in-a-cpu-job)
before running `egl bench run --submit`: it composes configs and hashes inputs
in the submitting process, so it must run inside a compute allocation.

## Run records

Training and evaluation write a run record for each attempt by default.
`record_json=auto` keeps attempts in separate directories, while
`record_json=/new/record.json` chooses an explicit destination and refuses to
overwrite an existing record. Use these records to inspect the configuration,
status, metrics, and available provenance for a run.
See [run records and scope](benchmark_suites.md#run-records-and-scope) for paths
and Python usage.
