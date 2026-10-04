#!/bin/bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
set -euo pipefail
cd "${EGOLEARN_RUN_DIR:?}/code"
: "${MANIFEST:?}" "${SLURM_ARRAY_TASK_ID:?}"
[[ $SLURM_ARRAY_TASK_ID =~ ^[0-9]+$ ]] || exit 1
[[ -f $MANIFEST && -r $MANIFEST ]] || exit 1
export EGOLEARN_WORK_DIR="$EGOLEARN_RUN_DIR/tasks/$SLURM_ARRAY_TASK_ID/work"
cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[[ -n $cmd ]] || { echo "no command for task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
exec bash scripts/slurm/run.sbatch bash -c "$cmd"
