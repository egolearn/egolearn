# Cluster jobs

egolearn includes a Slurm launcher, `scripts/slurm/submit`, that runs a
training or evaluation command on a compute node inside a pinned container.
Each submission archives the committed code, so later edits to your checkout
do not change queued jobs.

## Requirements

- The login node needs Bash 4.4 or later, Git, tar, iconv and `sbatch`, but
  no Python.
- Compute nodes need Bash 4.4 or later, Apptainer and `flock`, and GPU nodes
  need a driver that supports CUDA 13.0, the version in the container.
- Your checkout and all job storage must be outside `$HOME`.

`egl` commands run Python, so run them in a job or an interactive allocation
rather than on a login node. There, `egl submit` calls the same launcher.

## Setup and submission

Copy the environment template and keep it private:

```bash
cp scripts/slurm/env.example.sh env.sh
chmod 600 env.sh
```

In `env.sh`, set `SCRATCH` to a writable directory outside `$HOME`. The
template puts job directories, environments, datasets, outputs and caches under
it, and exports their paths, such as `EGOLEARN_JOB_ROOT` for job directories and
`EGOLEARN_ARTIFACTS` for outputs. Also set `SBATCH_ACCOUNT` and `SBATCH_PARTITION` for your site, and
`EGOLEARN_PREP_SBATCH_ARGS` for the CPU partition used by setup jobs. Export
`HF_TOKEN` and `WANDB_API_KEY` there too, rather than with `hf auth login` or
`wandb login`, which write credentials under `$HOME`. Git ignores `env.sh`.

Commit your code, source `env.sh`, and submit. Options before `--` go to
`sbatch`, and the command after `--` runs in the container:

```bash
source env.sh
scripts/slurm/submit --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 -- \
  egl train projects/babycl/configs/saycam_default.yaml \
  stream.root_dir=/path/to/saycam_5fps \
  stream.transcript_json=/path/to/expand_correct_seg.json \
  tokenizer.vocab_json=/path/to/vocab.json \
  "run_dir=$EGOLEARN_ARTIFACTS/babycl-seed0" 'probes=[]'
```

This example trains BabyCL on inputs prepared as its
[training data guide](../projects/babycl/docs/training_data.md) describes, and
`probes=[]` turns off evaluation during training. The launcher prints the job
ID and the path of the run's `run.json`. The first submission of a new commit
also queues a short CPU job that pulls the container image and installs the
environment, and the main job waits for it.

## Snapshots and records

Each submission creates a run directory, `$EGOLEARN_JOB_ROOT/run-XXXXXXXX/`:

| Path | Contents |
| --- | --- |
| `code/` | Read-only copy of the submitted commit |
| `work/` | The job's working directory, where relative output paths land |
| `run.json` | Commit, container image, command and submission time |
| `*.out` | The job's Slurm log |
| `tasks/<index>/work/` | One working directory per array task |

The launcher refuses a checkout with uncommitted or untracked changes. Use
absolute paths for data and `run_dir`, so results land outside the per-job
directory, and keep the run directory until the job has finished. Never put
credentials in a command or command file, because the run directory keeps a
copy.

### Resuming a run

Give each experiment and seed its own absolute `run_dir`. To continue after a
time limit or a failure, submit the same command again with `resume=auto`
added. It loads the newest `snapshot_*.pth` in `run_dir` and fails if there is
none, so leave it out of the first submission. The config must save snapshots
through `hooks.snapshot_every` or `hooks.snapshot_milestones`.

## Sweeps, arrays and dry runs

`--sweep` expands comma-separated Hydra overrides into every combination and
submits them as one job array. This example submits six tasks:

```bash
scripts/slurm/submit --sweep --max-parallel 4 --cpus-per-task=2 --mem=8G --time=00:10:00 -- \
  egl train toy_streaming seed=0,1,2 driver.rate.steps_per_admit=1,4
```

To run a list of commands as one array, write one shell command per line in a
file and pass `--commands FILE`. `--max-parallel N` limits how many tasks run
at once. Add `--dry-run` to any submission to check your settings, write the
snapshot and print the `sbatch` calls without submitting anything:

```bash
scripts/slurm/submit --dry-run -- egl train toy_streaming
```

## Diagnostics

Before a long run on a new cluster, submit `egl doctor` as a short GPU job to
check the container, the environment, the storage settings and CUDA:

```bash
scripts/slurm/submit --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00 -- \
  egl doctor --cluster --require-cuda
```

The job log has one `ok`, `warn` or `fail` line per check.
[Benchmark suites](benchmark_suites.md#cluster-runs) shows how to submit a
suite from a CPU allocation.
