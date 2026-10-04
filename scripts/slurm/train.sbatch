#!/bin/bash
# Compatibility command: submit through the snapshotting launcher, never raw sbatch.
# scripts/slurm/train.sbatch <config-dir> <config-name> [hydra overrides...]
set -euo pipefail
[[ -z ${SLURM_JOB_ID:-} ]] || {
  echo 'Run scripts/slurm/train.sbatch on the login node without sbatch.' >&2
  exit 1
}
config_dir=${1:?usage: train.sbatch <config-dir> <config-name> [overrides...]}
config_name=${2:?usage: train.sbatch <config-dir> <config-name> [overrides...]}
shift 2
source "$(dirname "$0")/common.sh"
default_resources 48:00:00
exec "$(dirname "$0")/submit" "${resources[@]}" -- python -m egolearn.train \
  --config-dir "$config_dir" --config-name "$config_name" "$@"
