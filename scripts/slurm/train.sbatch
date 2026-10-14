#!/bin/bash
# Compatibility command: submit through the snapshotting launcher, never raw sbatch.
# scripts/slurm/train.sbatch <config.yaml|packaged-name> [hydra overrides...]
set -euo pipefail
[[ -z ${SLURM_JOB_ID:-} ]] || {
  echo 'Run scripts/slurm/train.sbatch on the login node without sbatch.' >&2
  exit 1
}
config=${1:?usage: train.sbatch <config.yaml|packaged-name> [overrides...]}
shift
source "$(dirname "$0")/common.sh"
default_resources 48:00:00
exec "$(dirname "$0")/submit" "${resources[@]}" -- egl train "$config" "$@"
