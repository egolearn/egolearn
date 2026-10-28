#!/bin/bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
EGOLEARN_READONLY=1
container_env
export APPTAINERENV_UV_PROJECT_ENVIRONMENT="$EGOLEARN_VENV"
export APPTAINERENV_UV_NO_SYNC=1
export APPTAINERENV_OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}
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
if [[ ${EGOLEARN_REQUEUE_ON_PAUSE:-0} != 1 ]]; then
  exec apptainer exec "${nv[@]}" --home "$APPTAINER_HOME" "$SIF" "$@"
fi
# Opt-in for runners that save state before returning 75. A new allocation
# gets a fresh wall-clock budget; never restart inside the expiring one.
container_pid= interrupted=0 pause_requested=0
trap 'interrupted=1; pause_requested=1; if [[ -n "$container_pid" ]]; then kill -USR1 "$container_pid" 2>/dev/null || true; fi' USR1
apptainer exec "${nv[@]}" --home "$APPTAINER_HOME" "$SIF" "$@" &
container_pid=$!
if [[ $pause_requested = 1 ]]; then kill -USR1 "$container_pid" 2>/dev/null || true; fi
set +e
while true; do
  interrupted=0
  wait "$container_pid"
  result=$?
  [[ $interrupted = 1 ]] || break
done
set -e
if [[ $result = 75 ]]; then
  scontrol requeue "${SLURM_JOB_ID:?requeue needs a Slurm job}"
  exit 0
fi
exit "$result"
