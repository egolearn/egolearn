# Cluster jobs

Use `scripts/slurm/submit [sbatch options] -- <command>` from a clean checkout
on the login node. **Do not run Python on login nodes**, including any `egl`
command, environment installation, diagnostics or benchmark planning. The shell
launcher needs Bash 4.4+, Git, coreutils, findutils, tar, iconv and `sbatch`;
compute nodes need Bash 4.4+, Apptainer and `flock`. Environment builds and image
pulls run in CPU jobs. A compute node used for benchmark planning also needs the
shell launcher's tools listed above and uv, with `sbatch` available on `PATH` to
submit the worker array. Load the site's required modules in the planning shell.

## Setup and submission

Keep the checkout and all storage on scratch, outside `$HOME`. Copy
`scripts/slurm/env.example.sh` to the git-ignored `env.sh`, set `SCRATCH` to your
site's writable storage root, edit account/partition settings, and run:

```bash
chmod 600 env.sh
source env.sh
scripts/slurm/submit --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 -- \
  python -m egolearn.train --config-dir projects/babycl/configs \
  --config-name saycam_default stream.root_dir="$EGOLEARN_DATA/saycam"
```

Add the transcript and vocabulary overrides required by your config. Set
`SBATCH_ACCOUNT` and `SBATCH_PARTITION` in `env.sh` for the training jobs, and
`EGOLEARN_PREP_SBATCH_ARGS` for the CPU partition and any prep-specific account.
No site account, partition, credential, or personal path is built into the repo.
The compatibility command `scripts/slurm/train.sbatch` also submits through this
launcher. It accepts one YAML path or packaged config name followed by Hydra
overrides, for example `scripts/slurm/train.sbatch toy_streaming seed=19`;
do not pass it to `sbatch` directly. Use the generic submit command to
select other resources. Both compatibility wrappers retain one GPU, eight CPUs
and 64 GB memory; training defaults to 48 hours and benchmarking to two hours.
Site time, memory and GPU `SBATCH_*` settings override these defaults. The wrappers
also translate `SBATCH_CPUS_PER_TASK` into `--cpus-per-task`; Slurm itself does not
read that variable. `SBATCH_CPUS_PER_GPU` or a CPU entry in `SBATCH_TRES_PER_TASK`
takes precedence over this wrapper setting. TRES entries replace only the matching
CPU or GPU default; a CPU-only request retains the default GPU. GPU shard and MPS
entries suppress the whole-GPU default as well.
Generic `submit` uses site defaults for CPU and memory unless the caller requests
resources. Runtime OpenMP threads follow CPUs per task, falling back to the batch
step's allocated CPUs, unless `OMP_NUM_THREADS` was explicitly set.

A stamp covers `uv.lock`, `pyproject.toml`, `.python-version`, sync arguments,
the image digest and the submitted commit. Environments are reused under
`$EGOLEARN_ENV_ROOT/<stamp>`. When the image, interpreter or stamp is missing,
submit schedules a short CPU prep job to pull the image and run
`uv sync --frozen --no-editable` **inside** it. Training depends on `afterok` prep;
user AND dependencies are retained. OR dependencies are rejected when prep is
needed. The prep job drops inherited `SBATCH_*` settings except the account; CPU
placement and other prep settings come from `EGOLEARN_PREP_SBATCH_ARGS`. Main jobs
explicitly export the required environment, and inherited arrays are refused.

The package is installed as a wheel from the archived code. Changing the commit
creates a new environment; uv's shared package cache avoids repeated downloads.
Concurrent preparations use an exclusive environment lock, and running jobs hold
shared locks. A redundant queued prep exits as soon as the same environment is
ready, even while training holds a shared lock. No running job's dependencies are mutated by another submission.
`EGOLEARN_SYNC_ARGS` defaults to `--extra train --extra video`; it and prep options
are whitespace-separated arguments without shell evaluation.

## Image

`containers/IMAGE` pins the lab's public `jepa-agent-base` CUDA 13.0 image by digest.
It supplies FFmpeg 7.1.3, jemalloc and uv 0.12.21. Before submitting a long run,
use the [diagnostic job](#diagnostics) to check the image and CUDA on your cluster.
Your site supplies its own account, partition, and storage settings.

The first prep pulls a SIF into `$SIF_DIR`; image unpacking uses node-local `/tmp`,
where Apptainer extraction works independently of shared-filesystem semantics.
A site may provision the pinned SIF in that directory ahead of time. Image updates require a committed digest change in `containers/IMAGE`. GPU jobs add
`--nv`, CPU jobs do not. The node driver must support the image/runtime version.

## Snapshots and records

Every submit archives exactly one clean Git commit. Staged, unstaged and
untracked changes are refused; ignored data, `env.sh`, environments and outputs
are excluded. Commit the intended code before submitting. Git submodules and
tracked symlinks are refused because they could escape the archived tree.
`scripts/slurm/submit` uses Bash, Git and standard Unix tools to archive code and
write `run.json`, without Python on the login node. The installed `egl submit`
alias dispatches to that script and is for workstations or compute allocations.
Run `egl bench run --submit` in a CPU allocation: config composition and hashing
every pinned input happen before submission. Follow the
[benchmark planning steps](benchmark_suites.md#cluster-planning-in-a-cpu-job).
Training dependencies and the container are prepared in CPU jobs.
Command arguments must be UTF-8; submission checks them with `iconv` before
creating a snapshot so `run.json` remains valid JSON.

Each `$EGOLEARN_JOB_ROOT/run-*/` contains:

- `code/`: the submitted tree, with write bits removed and a read-only container bind.
- `work/`: a writable job directory with links to the snapshot's top-level files.
  Relative configs/scripts read that snapshot; relative outputs are written here.
- `run.json`, `dirty.diff`, `job-id`: commit, image, environment stamp, submission
  time, command, empty dirty diff (dirty runs are refused), and the Slurm job ID.
- `commands.sh` and `tasks/<index>/work/` for an array, so tasks have independent
  output directories. The shared command manifest is made read-only.

Edits to the original checkout after submission cannot change the archived code.
Use absolute paths for input data and external configs; tracked relative config
paths resolve inside the snapshot. Do not modify archived code or environments.
Retain the run directory and environment until all dependent jobs finish; these
artifacts remain local and may contain private config paths. Nothing is uploaded.

For `resume=auto`, use an absolute persistent `run_dir`, unique to that experiment
and seed, so a resubmission finds the earlier snapshots, for example:
`run_dir=$EGOLEARN_ARTIFACTS/babycl-seed0 resume=auto` on resubmission. Omit
`resume=auto` on the first run, before any snapshot exists.
The launcher warns when `resume=auto` has no explicit absolute `run_dir` override;
it cannot inspect a command's configuration files. Alternatively supply an
absolute `resume=/path/to/snapshot.pth` and choose a new output directory.

Relative `--output`, `--error` and `--input` paths (including Slurm environment
settings) are resolved from the submitting checkout before the working-directory
change. Logs inside the checkout must be git-ignored (`logs/`, `outputs/` and
`slurm-*.out`/`slurm-*.err` are covered); otherwise use scratch outside the checkout.
Input path patterns and `/dev/null` pass through to Slurm. Invalid command
files and paths fail before snapshot creation. Failures before main submission
cancel known prep jobs and remove their snapshots when cancellation succeeds.
Once main submission has been attempted, an error or interruption retains the
snapshot and prep: a controller timeout may hide a successfully accepted job.
A prep submission that fails without returning a job ID also retains its snapshot.
Inspect `squeue`/`sacct` before cancelling or removing retained runs. Even an empty
queue lookup cannot exclude a job that has already finished.

## Arrays and dry runs

```bash
scripts/slurm/submit --sweep --max-parallel 4 --gres=gpu:1 --cpus-per-task=8 --mem=64G -- \
  python -m egolearn.train --config-dir projects/babycl/configs \
  --config-name saycam_default seed=0,1,2 driver.rate.steps_per_admit=1,4
scripts/slurm/submit --commands /path/to/commands.txt --gres=gpu:1
scripts/slurm/submit --dry-run -- python -m egolearn.train --help
```

`--sweep` expands top-level Hydra comma lists into a Cartesian product. Lists,
mappings, quoted commas and escaped commas remain intact. `--commands` copies
nonblank, noncomment shell commands into one array; use trusted command files.
`--max-parallel` caps concurrent tasks. `--dry-run` writes the local snapshot and
record and prints the sbatch calls, but submits no jobs. `--chdir`, `--export`,
`--wrap` and raw `--array` are managed by the launcher and cannot be overridden.

## Diagnostics

On the login node, the shell launcher checks required storage settings and the
checkout before submitting. Its `--dry-run` performs those checks and writes a
snapshot without executing Python or submitting jobs.

Run `egl doctor` in an allocation. To check the prepared CPU environment, replace
`CPU_PARTITION` with your site's CPU partition and submit from the login node:

```bash
scripts/slurm/submit --partition=CPU_PARTITION --cpus-per-task=2 --mem=8G \
  --time=00:10:00 -- egl doctor --cluster --skip-cuda
```

For a CUDA check, request a GPU job and pass `--require-cuda` instead of
`--skip-cuda`. Doctor's CUDA probe imports torch in a child process. Even with
`--skip-cuda`, doctor itself runs Python and belongs in an allocation.

In a CPU planning shell, use `egl doctor --submit --skip-cuda` to check submission
tools and cluster storage before planning an array. This checks Bash, Git,
`sbatch`, `iconv`, `tar`, and uv on that host, including inside a Slurm allocation.
See the [benchmark planning workflow](benchmark_suites.md#cluster-planning-in-a-cpu-job).

## Caches and credentials

The environment example redirects uv, managed Python, Hugging Face, torch.hub,
Triton, torch.compile, CUDA JIT, matplotlib, W&B (runs/cache/settings/staging),
Apptainer (cache/config/container home) and XDG cache paths to scratch. Required
paths are checked after resolving symlinks, so paths under `$HOME` fail before
submission. Configured storage is bound into the container; cache settings are
forwarded explicitly. Keep inherited cache overrides on scratch too.

Host XDG configuration/data directories stay unchanged to preserve Git settings;
the container receives a scratch home through `--home`. Put `HF_TOKEN` and
`WANDB_API_KEY` in environment variables loaded from mode-0600 `env.sh`. Avoid
`hf auth login` and `wandb login`, which can write credentials under `$HOME`.
No credentials belong in command files, command arguments, or run records.
