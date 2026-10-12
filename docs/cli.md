# Command line

Use `uv run egl` after `uv sync`, or `egl` in an activated environment.
`egolearn` is an equivalent installed alias. Commands are `train`, `eval`, `data`,
`submit`, `doctor`, and `bench`; each accepts `--help`.

These Python commands are for workstations and compute allocations. On cluster
login nodes, use `scripts/slurm/submit`; follow the [cluster guide](cluster.md).

`egl train` and `egl eval` dispatch to the Hydra training and evaluation modules.
The first argument can be a packaged config name or an existing `.yaml` file;
relative file paths are resolved from the caller's working directory. `egl eval`
without a config selects the packaged `evaluate` defaults. `egl --version`
prints the installed version.

## Configuration and overrides

```bash
egl train toy_streaming seed=42 dry_run=true
egl train projects/babycl/configs/saycam_default.yaml --cfg job
egl eval path/to/suite.yaml checkpoint=runs/example/final.pth
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

## Data

```bash
egl data list
egl data list --json
egl data fetch shape-bias --revision <tag-or-commit>
egl data verify "$EGOLEARN_DATA/shape-bias"
egl data prepare shape-bias
egl data --root /path/to/data prepare babyview-4afc --vocab_json /path/to/vocab.json
```

The catalog records access tiers and source links. Fetch uses the checksummed Hub
mechanism; a listed collection is a publication candidate, not a promise that its
owner has uploaded it. IRB images and upstream-licensed datasets have no automatic
Hub fetch. Public split metadata is listed separately from restricted pixels.
`--repo org/name` selects a published dataset collection for a fetchable entry;
missing checksums require an explicit `--allow-unverified`. Corrupt checksums fail.
Fetch preserves existing collection directories unless `--replace` is explicit.
Preparation refuses an existing default output, preserving fetched collections;
choose an explicit preparer output option to opt into its legacy write behavior.
Help and spatial `--validate-only` remain available for existing outputs.

Fetch and preparation defaults share `$EGOLEARN_DATA` (default `./data`). Put
`--root DIR` before the data subcommand to override that root. Preparation forwards
the benchmark's existing flags unchanged; explicit source/output options override
the catalog defaults. Fetch and prepare use the same `ROOT/<name>` collection
layout; shape-bias source assets live separately under `ROOT/.sources/shape-bias`. Use `egl data prepare NAME --help` for each preparer's
options; this help works in a base installation without training extras.
Preparation still needs the dependencies and source assets documented
in the [evaluation guide](evaluation.md). It does not bypass data access requirements.

`data verify DIR` checks the collection's `checksums.sha256` without writing files.
It reports missing, altered, or unlisted payloads. Legacy preparation outputs have
their own benchmark validators and may not contain collection checksums; a missing
checksum file fails verification. The existing `python -m egolearn.hub checksum DIR`
command can create collection checksums after preparation. Existing module and
script entry points retain their original defaults.

## Diagnostics and cluster submission

Run these commands on a workstation or in a compute allocation:

```bash
egl doctor
egl doctor --extra train --require-cuda
egl doctor --cluster --require-env WANDB_API_KEY
egl submit --gres=gpu:1 --time=01:00:00 -- egl train toy_streaming
```

Doctor prints one `ok`, `warn`, or `fail` line per check. Failures return status 1;
warnings alone return 0. It checks the interpreter, virtual environment, installed
extras, CUDA availability, data-root readability and requested environment variable
names. CUDA is queried in a bounded child process without allocating a tensor;
`--skip-cuda` skips that query. Environment variable values are never printed.
Cluster checks activate with `--cluster` or inside a Slurm job. They share required
and optional storage-variable names with the launcher. Host checks resolve symlinks
before rejecting paths under the host's `$HOME`. Inside an Apptainer container,
`$HOME` is scratch by design: paths must still be absolute, but may be under that
container home. Submission tools are checked outside Slurm allocations and
Apptainer on compute hosts; neither is required inside a job container.
Doctor creates no directories, downloads
no packages and changes no settings.

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
egl bench run benchmarks/toy.yaml --output runs/toy-suite
egl bench table runs/toy-suite/manifest.json
egl bench table runs/toy-suite/manifest.json --format json
```

The [suite guide](benchmark_suites.md) defines pinned inputs, per-seed records,
baseline comparisons and failure handling. Local execution is sequential; `--submit`
creates one Slurm array task per seed. The toy suite uses synthetic data and CPU.
Follow [cluster planning in a CPU job](benchmark_suites.md#cluster-planning-in-a-cpu-job)
before running `egl bench run --submit`: it composes configs and hashes inputs
in the submitting process, so it must run inside a compute allocation.
