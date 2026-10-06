#!/bin/bash
# Generic training launcher. Cluster-specific values (account, partition,
# GPU type) go on the sbatch command line, never in this file:
#
#   sbatch --account=<account> --gres=gpu:<type>:1 scripts/slurm/train.sbatch \
#       projects/babycl/configs/saycam_default.yaml \
#       stream.root_dir=$DATA_ROOT stream.transcript_json=$TRANSCRIPT_JSON \
#       tokenizer.vocab_json=$VOCAB_JSON
#
# Prereq (once, on a login node; the venv lives on the shared filesystem):
#   uv sync --extra train
# See scripts/slurm/README.md for cluster setup, including the apptainer route.

#SBATCH --job-name=egolearn
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail
module purge 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

TRAIN_CONFIG=${1:?usage: train.sbatch <config.yaml|packaged-name> [hydra overrides...]}
shift

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

# No internet on compute nodes? Pre-sync on the login node and set
# WANDB_MODE=offline (sync later with `wandb sync`).
uv run --no-sync egl train "$TRAIN_CONFIG" "$@"
