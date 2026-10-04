#!/bin/bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${EGOLEARN_RUN_DIR:?}/code"
# shellcheck source=scripts/slurm/common.sh
source scripts/slurm/common.sh
job_context
: "${SIF:?}" "${EGOLEARN_IMAGE:?}" "${EGOLEARN_EXPECTED_STAMP:?}"
outside_home "$SIF"
mkdir -p "$EGOLEARN_VENV"
[[ -f $SIF ]] || mkdir -p "$SIF_DIR"
container_env
export APPTAINER_TMPDIR
APPTAINER_TMPDIR=$(mktemp -d "/tmp/egolearn-prep.${SLURM_JOB_ID:?}.XXXXXXXX")
partial=
cleanup() { rm -rf "$APPTAINER_TMPDIR"; [[ -z $partial ]] || rm -f "$partial"; }
trap cleanup EXIT
# Pull only when the image is missing, so a read-only shared SIF_DIR works for
# everyone else; the lock keeps two users from pulling the same image at once.
if [[ ! -f $SIF ]]; then
  (umask 000; touch "$SIF_DIR/$(basename "$SIF").lock")
  (
    umask 022
    flock -x 9
    if [[ ! -f $SIF ]]; then
      partial="$SIF.${SLURM_JOB_ID}.partial"
      trap '[[ -z $partial ]] || rm -f "$partial"' EXIT
      apptainer pull "$partial" "docker://$EGOLEARN_IMAGE"
      mv "$partial" "$SIF"
      partial=
      sha256sum "$SIF" > "$SIF.sha256.${SLURM_JOB_ID}.partial"
      mv "$SIF.sha256.${SLURM_JOB_ID}.partial" "$SIF.sha256"
    fi
  ) 9>>"$SIF_DIR/$(basename "$SIF").lock"
fi
(
  # Another queued prep may have finished, with training already holding a
  # shared lock. Do not block on an exclusive lock when the environment is ready.
  while ! flock -xn 8; do
    environment_ready && exit 0
    sleep 1
  done
  environment_ready && exit 0
  [[ $(stamp) = "$EGOLEARN_EXPECTED_STAMP" ]] || fail "lock/config changed since submission; resubmit"
  apptainer exec --home "$APPTAINER_HOME" "$SIF" /usr/local/bin/uv --version ||
    fail "image has no working uv at /usr/local/bin/uv; select an image with uv (no host fallback)"
  if [[ ! -x "$EGOLEARN_VENV/bin/python" || ! -f "$EGOLEARN_VENV/.egolearn-sync-stamp" || $(cat "$EGOLEARN_VENV/.egolearn-sync-stamp") != "$EGOLEARN_EXPECTED_STAMP" ]]; then
    read -r -a sync_args <<< "$EGOLEARN_SYNC_ARGS"
    rm -f "$EGOLEARN_VENV/.egolearn-sync-stamp"
    export APPTAINERENV_UV_PROJECT_ENVIRONMENT="$EGOLEARN_VENV"
    export APPTAINERENV_UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
    export APPTAINERENV_UV_PYTHON_PREFERENCE=only-managed
    # Keep the lock inode when initializing or changing the managed interpreter.
    apptainer exec --home "$APPTAINER_HOME" "$SIF" /usr/local/bin/uv venv --allow-existing "$EGOLEARN_VENV"
    apptainer exec --home "$APPTAINER_HOME" "$SIF" /usr/local/bin/uv sync --frozen --no-editable "${sync_args[@]}"
    [[ $(stamp) = "$EGOLEARN_EXPECTED_STAMP" ]] || fail "lock/config changed during sync; resubmit"
    printf '%s\n' "$EGOLEARN_EXPECTED_STAMP" > "$EGOLEARN_VENV/.egolearn-sync-stamp.partial"
    mv "$EGOLEARN_VENV/.egolearn-sync-stamp.partial" "$EGOLEARN_VENV/.egolearn-sync-stamp"
  fi
) 8>"$EGOLEARN_VENV/.egolearn-sync.lock"
