# shellcheck shell=bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
# Copy to env.sh, chmod 600 env.sh, edit and source it before submission.
# SCRATCH is your site's absolute writable storage root, outside $HOME.
: "${SCRATCH:?set SCRATCH to your scratch directory first}"
export CACHE_ROOT=$SCRATCH/egolearn-cache
export SIF_DIR=$SCRATCH/images
export EGOLEARN_JOB_ROOT=$SCRATCH/egolearn-jobs
export EGOLEARN_ENV_ROOT=$SCRATCH/egolearn-envs
export EGOLEARN_DATA=$SCRATCH/datasets
export EGOLEARN_ARTIFACTS=$SCRATCH/egolearn-artifacts

# Slurm account/partitions are supplied by your site. No values are built in.
# export SBATCH_ACCOUNT=ACCOUNT
# export SBATCH_PARTITION=GPU_PARTITION
# export EGOLEARN_PREP_SBATCH_ARGS="--partition=CPU_PARTITION --time=00:30:00"
export EGOLEARN_SYNC_ARGS="--extra train --extra video"

export UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_PYTHON_INSTALL_DIR=$CACHE_ROOT/uv-python
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME=$CACHE_ROOT/huggingface
export TORCH_HOME=$CACHE_ROOT/torch
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export CUDA_CACHE_PATH=$CACHE_ROOT/nv
export XDG_CACHE_HOME=$CACHE_ROOT
export MPLCONFIGDIR=$CACHE_ROOT/matplotlib
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$CACHE_ROOT/wandb
export WANDB_CONFIG_DIR=$CACHE_ROOT/wandb-config
export WANDB_DATA_DIR=$CACHE_ROOT/wandb-data
export APPTAINER_CACHEDIR=$CACHE_ROOT/apptainer
export APPTAINER_CONFIGDIR=$CACHE_ROOT/apptainer-config
export APPTAINER_HOME=$SCRATCH/apptainer-home
# Prep chooses node-local /tmp for image unpacking; never unpack on shared scratch.
# XDG_CONFIG_HOME/XDG_DATA_HOME stay unchanged so host git settings still work.
# Set credentials as env vars, never through login tools that write to $HOME.
# export HF_TOKEN=
# export WANDB_API_KEY=
export APPTAINERENV_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
