#!/bin/bash
# Data-plane benchmark launcher (scripts/bench/bench_dataloading.py).
#
#   sbatch --account=<account> --gres=gpu:<type>:1 scripts/slurm/bench.sbatch \
#       <config.yaml> [bench args...]
#
# No GPU keepalive here on purpose: it would sit on the CPUs/GPU and skew
# the latency numbers this job exists to measure.

#SBATCH --job-name=egolearn-bench
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
module purge 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

CONFIG=${1:?usage: bench.sbatch <config.yaml> [bench args...]}
shift

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

uv run --no-sync python scripts/bench/bench_dataloading.py \
  --config "$CONFIG" --output-json "bench_${SLURM_JOB_ID:-local}.json" "$@"
