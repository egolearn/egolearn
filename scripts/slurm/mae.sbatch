#!/bin/bash
#SBATCH --job-name=mae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue

# All site paths are supplied by the submitter. MAE_SOURCE must be a frozen
# source snapshot and MAE_PYTHON a non-editable installation built from it.
set -euo pipefail
: "${MAE_SOURCE:?frozen source directory}"
: "${MAE_PYTHON:?absolute Python path with egolearn installed}"
: "${MAE_IMAGE:?Apptainer image}"
: "${MAE_BINDPATH:?Apptainer bind paths}"
: "${MAE_OVERLAY_LIST:?file containing one read-only archive path per line}"
: "${MAE_RUN_ROOT:?output root}"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation required}"
config=${1:?full-pass MAE or StreamMAE config name}
shift
case "$config" in
  full_saycam_streaming|full_saycam_iid|full_babyview_streaming|full_babyview_iid|full_saycam_streammae|full_babyview_streammae|full_walkingtours_streammae|full_wtpp_london_streammae|full_wt_original_streammae) ;;
  *) echo "unsupported MAE config: $config" >&2; exit 2 ;;
esac
mkdir -p "$MAE_RUN_ROOT/$config"
overlays=()
while IFS= read -r archive || [[ -n "$archive" ]]; do
  [[ -z "$archive" ]] || overlays+=(--overlay "${archive}:ro")
done < "$MAE_OVERLAY_LIST"
train_pid=
interrupted=0
trap 'interrupted=1; if [[ -n "$train_pid" ]]; then kill -USR1 "$train_pid" 2>/dev/null || true; fi' USR1
set +e
apptainer exec --nv --cleanenv --no-home \
  --bind "$MAE_BINDPATH" "${overlays[@]}" \
  --env OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,TMPDIR=/tmp,PYTHONDONTWRITEBYTECODE=1 \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,SLURM_JOB_ID=${SLURM_JOB_ID:-local},SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}" \
  "$MAE_IMAGE" "$MAE_PYTHON" -u -m egolearn.methods.mae.train \
  --config-dir "$MAE_SOURCE/projects/streammae/configs" --config-name "$config" \
  "run_dir=$MAE_RUN_ROOT/$config" \
  "hydra.run.dir=$MAE_RUN_ROOT/$config/hydra/${SLURM_JOB_ID:-local}_${SLURM_RESTART_COUNT:-0}" \
  "$@" &
train_pid=$!
while true; do
  interrupted=0
  wait "$train_pid"
  result=$?
  # Re-wait even if the child finished during the trap; bash preserves its
  # actual status, whereas the interrupted wait may have returned 128+SIGUSR1.
  [[ "$interrupted" == 1 ]] || break
done
set -e
if [[ "$result" == 75 && "${MAE_REQUEUE:-1}" == 1 ]]; then
  : "${SLURM_JOB_ID:?automatic requeue requires a Slurm allocation}"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$result"
