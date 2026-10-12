# Benchmark suites

`egl bench` runs a declared set of seeds and generates tables from their local
records. A suite is a versioned YAML file; [the toy suite](../benchmarks/toy.yaml)
is a runnable CPU example. It reports the last training step's loss and step count,
not an evaluation score or an average over training. Records retain skipped-step
counts; a nonfinite training loss stops a recorded run immediately and marks it nonfinite.
Ordinary training without a record retains its existing skip behavior. Final configured probe results are recorded separately.

```yaml
version: 1
name: my-evaluation
entry: eval
config: ../projects/example/configs/evaluate.yaml
seeds: [0, 1, 2]
overrides:
  - device=cpu
inputs:
  checkpoint:
    path: /path/to/model.pth
    sha256: <64-character-sha256>
  benchmarks.0.manifest:
    path: /path/to/trials.json
    sha256: <64-character-sha256>
metrics:
  objects/accuracy: max
```

Replace the example paths and hashes with your own artifacts. Metric keys are
`<benchmark name>/<metric name>` for evaluation and `train/last_step/<log key>` for the last training update and
`probe/<name>/<metric>` for final training probes;
declare each metric's direction as `min` or `max`. A model checkpoint is mandatory
and checksum-pinned for evaluation. `entry: train` accepts a training configuration
and may pin initialization checkpoints or data inputs the same way.

Config files and local input paths resolve relative to the suite file. Packaged
config names, such as `toy_streaming`, also work. Each `inputs` key is a dotted
Hydra config key; the runner sets it to the verified input path. For a directory,
pin the SHA-256 of its `checksums.sha256`; every listed file and the collection's
completeness are then checked. Hub references are resolved before verifying the
pinned content. Input files must remain accessible on the compute node. Pin all
external data/manifests needed for the comparison; the runner cannot infer the
meaning of arbitrary component configuration fields.

## Running

Run these commands on a workstation or compute node. All `egl` commands run
Python; on the cluster, first enter a CPU allocation as described below.

```bash
egl bench run benchmarks/toy.yaml --output runs/toy-first
egl bench run benchmarks/toy.yaml --output runs/toy-second \
  --override solver.groups.0.lr=0.0005
egl bench table runs/toy-second/manifest.json --baseline runs/toy-first/manifest.json
```

Output must be a new directory outside the running source checkout or ignored by
Git (for example, `runs/`). This keeps output creation from changing the recorded
source identity. Before creating it, the runner verifies inputs and
composes a complete configuration for every seed. It writes one
`seed-<seed>/config.yaml`, an expected-run `manifest.json`, and eventually a log and
`record.json` per seed. Configuration defaults are composed at planning time. Interpolations are
preserved and resolve inside each run, after its seed and output paths are set;
worker environment values therefore do not get frozen during planning. Each run
uses its frozen config and isolated working directory; use pinned input paths
rather than relying on relative data paths.

Overrides accept single Hydra `key=value` expressions, including additions with
`+` or `++`. They cannot replace pins or their parent/child keys, seeds, output
locations, Hydra settings, resume, or dry-run mode. Suite execution always starts
fresh. Normal `egl train`/`eval` retain their unrestricted Hydra interface.

The worker rechecks inputs, config and manifest hashes before execution and checks
the code/environment identity against the plan. Changed or missing inputs fail;
completed seeds remain on disk when another fails. Full input verification runs
before and after each task, which adds I/O proportional to the pinned data size. A claimed seed cannot be run
twice into the same directory. Create a new output directory for retries.

### Cluster planning in a CPU job

`egl bench run --submit` composes configurations and hashes every pinned input
before it submits the array. Run that planning in a CPU job, never on the login
node. It requires an editable installation of the clean submitting checkout so
configuration defaults come from the code that will execute. Keep the checkout on
shared scratch. From its root on the login node, source the
[cluster environment](cluster.md#setup-and-submission) and request a compute shell
(replace `CPU_PARTITION` and `ACCOUNT` with your site's settings):

```bash
source env.sh
srun --partition=CPU_PARTITION --account=ACCOUNT --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
```

Once `srun` has started the shell **on the compute node**, run from that same
checkout root:

```bash
uv sync --locked --extra train
uv run --no-sync egl doctor --cluster --skip-cuda
uv run --no-sync egl bench run benchmarks/toy.yaml \
  --output "$EGOLEARN_ARTIFACTS/toy-suite" --submit \
  --sbatch=--partition=CPU_PARTITION --sbatch=--cpus-per-task=2 \
  --sbatch=--mem=8G --sbatch=--time=00:10:00
exit
```

Choose planning time and memory for the input size; hashing a large collection
can take longer than this toy example. The `--sbatch` settings select resources
for the eventual workers independently of the planning job; request GPU resources
there for GPU suites. Submission finishes once the array is queued, so the
planning allocation can exit. Run table generation in an allocation too.

Planning uses the compute host's editable environment and `sbatch`, rather than
the launcher's archived wheel environment, which has no editable Git checkout.
It delegates to the existing clean-checkout cluster launcher with one array
task per seed; repeat `--sbatch=OPTION` for Slurm settings. Commands use the compute
environment's Python. The archived commit and pinned image identify the queued
code. Put output on shared scratch outside `$HOME`, accessible to every task. The
output directory must be outside the checkout or ignored by Git. Inputs and output
must be under configured storage binds or explicit `APPTAINER_BIND` destinations;
unbound and read-only output paths fail before planning creates files. Arrays
disable automatic requeue because each seed is claimed once; retry with a new
output directory. Interrupted tasks remain incomplete, never successful.

## Tables

`egl bench table MANIFEST --format md|csv|json` reports mean and sample standard
deviation. A singleton has an unavailable SD. A metric is averaged only when every
expected seed supplies a finite value from a completed matching run. Missing,
failed, malformed, nonfinite and mismatched records are listed; incomplete metrics
have no mean or SD. A baseline must use the same suite name, entry, metric names
and directions, seed set, and input pins (config keys and SHA-256 checksums).
Seed order and local input paths may differ; identical content at a new location
is still comparable. Changed, added or removed pins or seeds fail before any
table or delta is printed, including with `--allow-unverified`. Code revisions
and unpinned configuration choices may differ for experiments. JSON output includes
the seed set and input hashes without exposing local input paths.
Deltas are current minus baseline, and regressions follow the declared
direction. Any worsening is flagged; this is not a statistical significance test.
Incomplete results and regressions return exit status 1.

Tables reject dirty or unavailable source revisions by default. Use
`--allow-unverified` for explicitly marked exploratory summaries. Tables contain
metrics and seed-level diagnostics, not local config paths. The underlying configs,
logs and records are local run artifacts; do not publish them without reviewing
their contents for private paths and settings.

## Run records and scope

Suites always request an atomic `record.json`. Ordinary `egl train` and `egl eval`
can opt in with `record_json=/path/to/record.json`; the default remains disabled.
Records contain status, seed, resolved config, wall time, metrics, package version,
code revision/dirty state, lockfile hash and container image when available.
The runner adds verified input identities and hashes tying records to the suite
and frozen config. Missing provenance is recorded as `unavailable`, never guessed
from an unrelated checkout in the caller's working directory.

These commands implement suite execution and tables from issue #25's proposal.
They do not yet make run records automatic for every training/evaluation API,
infer dataset revisions from arbitrary components, publish artifacts, or replace
the evaluator's benchmark-specific provenance. Those remain separate work in #25.

`egl bench` measures declared training/evaluation metrics. The separate scripts
under `scripts/bench/` remain throughput and data-loading tools.
