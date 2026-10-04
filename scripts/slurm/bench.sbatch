#!/bin/bash
# Submit the data-plane benchmark from immutable code through the same prep path.
# scripts/slurm/bench.sbatch <config.yaml> [benchmark args...]
set -euo pipefail
[[ -z ${SLURM_JOB_ID:-} ]] || {
  echo 'Run scripts/slurm/bench.sbatch on the login node without sbatch.' >&2
  exit 1
}
config=${1:?usage: bench.sbatch <config.yaml> [benchmark args...]}
shift
source "$(dirname "$0")/common.sh"
default_resources 02:00:00
exec "$(dirname "$0")/submit" "${resources[@]}" -- python scripts/bench/bench_dataloading.py \
  --config "$config" --output-json bench.json "$@"
