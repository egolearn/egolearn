# Cluster jobs

Use `scripts/slurm/submit [sbatch options] -- <command>` from a clean checkout.
The login node needs Bash, Python 3.12+, Git, coreutils and `sbatch`; compute nodes
need Apptainer and `flock`. There is no login-node environment build or image pull.

## Setup and submission

Keep the checkout and all storage on scratch, outside `$HOME`. Copy
`scripts/slurm/env.example.sh` to the git-ignored `env.sh`, set `SCRATCH` to your
site's writable storage root, edit account/partition settings, and run:

```bash
chmod 600 env.sh
source env.sh
scripts/slurm/submit --gres=gpu:1 --time=24:00:00 -- \
  python -m egolearn.train --config-dir projects/babycl/configs \
  --config-name saycam_default stream.root_dir="$EGOLEARN_DATA/saycam"
```

Add the transcript and vocabulary overrides required by your config. Set
`SBATCH_ACCOUNT` and `SBATCH_PARTITION` in `env.sh` for the training jobs, and
`EGOLEARN_PREP_SBATCH_ARGS` for the CPU partition and any prep-specific account.
No site account, partition, credential, or personal path is built into the repo.
The compatibility command `scripts/slurm/train.sbatch` also submits through this
launcher; do not pass it to `sbatch` directly. Use the generic submit command to
select GPU and other resource requests.

A stamp covers `uv.lock`, `pyproject.toml`, `.python-version`, sync arguments,
the image digest and the submitted commit. Environments are reused under
`$EGOLEARN_ENV_ROOT/<stamp>`. When the image, interpreter or stamp is missing,
submit schedules a short CPU prep job to pull the image and run
`uv sync --frozen --no-editable` **inside** it. Training depends on `afterok` prep;
user AND dependencies are retained. OR dependencies are rejected when prep is
needed. The prep job drops inherited GPU partition/resource requests.

The package is installed as a wheel from the archived code. Changing the commit
creates a new environment; uv's shared package cache avoids repeated downloads.
Concurrent preparations use an exclusive environment lock, and running jobs hold
shared locks. No running job's dependencies are mutated by another submission.
`EGOLEARN_SYNC_ARGS` defaults to `--extra train --extra video`; it and prep options
are whitespace-separated arguments without shell evaluation.

## Image

`containers/IMAGE` pins the lab's public `jepa-agent-base` CUDA 13.0 image by digest.
It supplies FFmpeg 7.1.3, jemalloc and uv 0.12.21. This pin comes from
[jepa-agent PR #94](https://github.com/agentic-learning-ai-lab/jepa-agent/pull/94),
which reports an Apptainer pull and uv check on a compute node. The image build
also exercises FFmpeg under Docker and Apptainer. This change's launcher tests
use fake Slurm and Apptainer; a live site smoke run is still required.

The first prep pulls a SIF into `$SIF_DIR`; image unpacking uses node-local `/tmp`,
where Apptainer extraction works independently of shared-filesystem semantics.
A site may provision the pinned SIF in that directory ahead of time. Image updates require a committed digest change in `containers/IMAGE`. GPU jobs add
`--nv`, CPU jobs do not. The node driver must support the image/runtime version.

## Snapshots and records

Every submit archives exactly one clean Git commit. Staged, unstaged and
untracked changes are refused; ignored data, `env.sh`, environments and outputs
are excluded. Commit the intended code before submitting. Git submodules and
tracked symlinks are refused because they could escape the archived tree.

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

## Arrays and dry runs

```bash
scripts/slurm/submit --sweep --max-parallel 4 --gres=gpu:1 -- \
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
