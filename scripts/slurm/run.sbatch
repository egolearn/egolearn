#!/bin/bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
set -euo pipefail
cd "${EGOLEARN_RUN_DIR:?}/code"
# shellcheck source=scripts/slurm/common.sh
source scripts/slurm/common.sh
job_context
: "${SIF:?}"
outside_home "$SIF"
# Keep prep from mutating this environment while it is in use.
check_stamp
# read-write: a shared lock on NFS needs a readable descriptor (EBADF otherwise)
exec 8<>"$EGOLEARN_VENV/.egolearn-sync.lock"
flock -s 8
check_stamp
container_env
# Explicit read-only binds override the containing writable storage binds.
export APPTAINER_BIND="$APPTAINER_BIND,$PWD:$PWD:ro,$EGOLEARN_VENV:$EGOLEARN_VENV:ro"
unset PYTHONPATH PYTHONHOME APPTAINERENV_PYTHONPATH APPTAINERENV_PYTHONHOME
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
cd "${EGOLEARN_WORK_DIR:-$EGOLEARN_RUN_DIR/work}"
nv=()
case "${CUDA_VISIBLE_DEVICES:-}" in
  ''|-1|NoDevFiles|void) ;;
  *) nv=(--nv);;
esac
if [[ ${SLURM_GPUS_ON_NODE:-0} != 0 ]]; then nv=(--nv); fi
echo "job=${SLURM_JOB_ID} node=$(hostname) cmd: $*"
apptainer exec "${nv[@]}" --home "$APPTAINER_HOME" "$SIF" "$@"
