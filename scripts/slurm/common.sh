#!/bin/bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
# Shared by submit and the batch entry points.
canonical() {
  local path=$1 tail=
  while [[ ! -e $path ]]; do
    tail="/${path##*/}$tail"
    path=${path%/*}
    [[ -n $path ]] || path=/
  done
  path=$(realpath "$path")
  # Resolve dot components in the nonexistent suffix as well. The loop needs a
  # suffix: an empty array is unbound under set -u in bash 3.2.
  if [[ -n $tail ]]; then
    local component
    local -a components
    IFS=/ read -r -a components <<< "$tail"
    for component in "${components[@]}"; do
      case "$component" in
        ''|.) ;;
        ..) path=${path%/*};;
        *) path="${path%/}/$component";;
      esac
    done
  fi
  printf '%s\n' "${path:-/}"
}
fail() { echo "slurm: $*" >&2; exit 1; }
# One image contract: how the pinned reference is read and where its SIF lives.
IMAGE_RE='^[A-Za-z0-9][A-Za-z0-9.:-]*/[A-Za-z0-9_./-]+(:[A-Za-z0-9_.-]+)?@sha256:[a-fA-F0-9]{64}$'
# Print the reference pinned in containers/IMAGE; nonzero when the file is
# missing or holds anything but one valid pinned reference.
pinned_image() {
  local ref=
  if [[ -f containers/IMAGE ]]; then
    ref=$(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' containers/IMAGE)
  fi
  if [[ -n $ref && $ref =~ $IMAGE_RE ]]; then
    printf '%s\n' "$ref"
  else
    return 1
  fi
}
# Print the SIF file for the pinned reference $1: the
# $SIF_DIR/<name>@sha256-<digest>.sif the first submit pulls.
sif_path() {
  local name=${1%@sha256:*}
  name=${name##*/}
  name=${name%%:*}
  printf '%s\n' "${SIF_DIR:?}/$name@sha256-${1##*:}.sif"
}
# Optional storage paths from env.sh: checked against $HOME and bound into the container.
STORAGE_VARS=(CACHE_ROOT UV_CACHE_DIR HF_HOME TORCH_HOME TRITON_CACHE_DIR
  TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH XDG_CACHE_HOME MPLCONFIGDIR MAMBA_ROOT_PREFIX
  HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR
  WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR EGOLEARN_DATA BENCH_DIR
  EGOLEARN_ARTIFACTS EGOLEARN_JOB_ROOT EGOLEARN_ENV_ROOT EGOLEARN_VENV)
# Required in the environment (env.sh) before any launcher runs.
REQUIRED_VARS=(SIF_DIR APPTAINER_HOME APPTAINER_CACHEDIR APPTAINER_CONFIGDIR
  UV_PYTHON_INSTALL_DIR EGOLEARN_JOB_ROOT EGOLEARN_ENV_ROOT CACHE_ROOT UV_CACHE_DIR
  HF_HOME TORCH_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR CUDA_CACHE_PATH
  XDG_CACHE_HOME MPLCONFIGDIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR)
# The venv sync arguments every prep job uses; env.sh may override them (EGOLEARN_SYNC_ARGS).
DEFAULT_SYNC_ARGS='--extra train --extra video'
outside_home() {
  local resolved home
  [[ $1 = /* ]] || fail "use an absolute path: $1"
  resolved=$(canonical "$1")
  home=$(canonical "${HOME:?}")
  case "$resolved" in "$home"|"$home"/*) fail "path under \$HOME: $1";; esac
}
check_env() {
  local name
  outside_home "$PWD"
  for name in "${REQUIRED_VARS[@]}"; do
    [[ -n ${!name:-} ]] || fail "source env.sh: set $name on scratch"
    outside_home "${!name}"
  done
  for name in "${STORAGE_VARS[@]}"; do
    [[ -z ${!name:-} ]] || outside_home "${!name}"
  done
  [[ -f uv.lock && -f pyproject.toml ]] || fail "run from the checkout root"
  export EGOLEARN_SYNC_ARGS=${EGOLEARN_SYNC_ARGS-$DEFAULT_SYNC_ARGS}
}
stamp() {
  local -a args
  read -r -a args <<< "${EGOLEARN_SYNC_ARGS-$DEFAULT_SYNC_ARGS}"
  {
    sha256sum < uv.lock
    sha256sum < pyproject.toml
    if [[ -f .python-version ]]; then sha256sum < .python-version; fi
    printf '%s\0' --frozen --no-editable "${args[@]}"
    pinned_image
    printf '%s\n' "${EGOLEARN_CODE_COMMIT:?}"
  } | sha256sum | cut -d ' ' -f 1
}
environment_ready() {
  [[ -x ${EGOLEARN_VENV:?}/bin/python && -f $EGOLEARN_VENV/.egolearn-sync-stamp &&
     $(cat "$EGOLEARN_VENV/.egolearn-sync-stamp") = "${EGOLEARN_EXPECTED_STAMP:?}" ]]
}
check_stamp() {
  [[ $(stamp) = "${EGOLEARN_EXPECTED_STAMP:?submit this job with scripts/slurm/submit}" ]] &&
    environment_ready || fail "venv stamp changed or missing; resubmit with scripts/slurm/submit"
}
container_env() {
  mkdir -p "$APPTAINER_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_CONFIGDIR"
  local name path source checkout bind dest
  local -a binds entries dests user_dests
  checkout=$(realpath "$PWD")
  binds=() dests=() user_dests=()
  IFS=, read -r -a entries <<< "${APPTAINER_BIND:-}"
  # Destinations already bound by the user; configured paths matching one are skipped.
  for bind in "${entries[@]}"; do
    [[ -n $bind ]] || continue
    dest=${bind#*:}; dest=${dest%%:*}
    [[ $bind = *:* ]] || dest=$bind
    user_dests+=("$dest")
  done
  for name in "${REQUIRED_VARS[@]}" "${STORAGE_VARS[@]}" PWD; do
    path=${!name:-}
    [[ -n $path ]] || continue
    [[ $path != *:* && $path != *,* ]] || fail "$name contains ':' or ',', which APPTAINER_BIND cannot carry: $path"
    # A destination the user already binds is covered by that entry.
    for dest in "${user_dests[@]}"; do
      [[ $dest != "$path" ]] || continue 2
    done
    [[ -d $path ]] || mkdir -p "$path" 2>/dev/null || fail "cannot create $name=$path"
    source=$(realpath "$path")
    [[ $source != *:* && $source != *,* ]] || fail "$name resolves to a path with ':' or ',', which APPTAINER_BIND cannot carry: $source"
    # A path inside the checkout is already visible through the checkout's bind.
    if [[ $name != PWD ]]; then
      case "$path" in "$PWD"|"$PWD"/*|"$checkout"|"$checkout"/*)
        case "$source" in "$PWD"|"$PWD"/*|"$checkout"|"$checkout"/*)
          continue
          ;;
        esac
        # Symlink out of the checkout: bind the resolved target at its own location.
        entries+=("$source")
        continue
        ;;
      esac
    fi
    # The job uses the configured path; bind the resolved directory there.
    if [[ $source = "$path" ]]; then entries+=("$path"); else entries+=("$source:$path"); fi
  done
  # One bind per container destination; the first (the user's APPTAINER_BIND) wins.
  for bind in "${entries[@]}"; do
    [[ -n $bind ]] || continue
    dest=${bind#*:}; dest=${dest%%:*}
    [[ $bind = *:* ]] || dest=$bind
    for existing in "${dests[@]}"; do
      [[ $existing != "$dest" ]] || continue 2
    done
    if [[ ${EGOLEARN_READONLY:-0} = 1 && ( $dest = "$PWD" || $dest = "$EGOLEARN_VENV" ) ]]; then
      bind="${bind%%:*}:$dest:ro"
    fi
    binds+=("$bind") dests+=("$dest")
  done
  APPTAINER_BIND=$(IFS=,; printf '%s' "${binds[*]}")
  export APPTAINER_BIND
  for name in "${REQUIRED_VARS[@]}" "${STORAGE_VARS[@]}"; do
    [[ -z ${!name:-} ]] || export "APPTAINERENV_$name=${!name}"
  done
  export APPTAINERENV_PREPEND_PATH="$EGOLEARN_VENV/bin"
  export APPTAINERENV_VIRTUAL_ENV="$EGOLEARN_VENV"
}

job_context() {
  : "${EGOLEARN_RUN_DIR:?submit this job with scripts/slurm/submit}"
  cd "$EGOLEARN_RUN_DIR/code"
  check_env
  export EGOLEARN_VENV="$EGOLEARN_ENV_ROOT/$EGOLEARN_EXPECTED_STAMP"
  outside_home "$EGOLEARN_VENV"
}

# Preserve #SBATCH-style precedence: explicit site environment settings win.
default_resources() {
  local tres=,${SBATCH_TRES_PER_TASK:-}
  resources=()
  [[ -n ${SBATCH_TIMELIMIT:-} ]] || resources+=("--time=$1")
  # sbatch itself has no SBATCH_CPUS_PER_TASK input; translate it for these wrappers.
  [[ -n ${SBATCH_CPUS_PER_GPU:-} || $tres = *,cpu[=:]* ]] || resources+=("--cpus-per-task=${SBATCH_CPUS_PER_TASK:-8}")
  [[ -n ${SBATCH_MEM_PER_NODE:-}${SBATCH_MEM_PER_CPU:-}${SBATCH_MEM_PER_GPU:-} ]] || resources+=(--mem=64G)
  [[ -n ${SBATCH_GRES:-}${SBATCH_GPUS:-}${SBATCH_GPUS_PER_NODE:-}${SBATCH_GPUS_PER_TASK:-} ||
     $tres = *,gres/gpu[=:]* || $tres = *,gres/shard[=:]* || $tres = *,gres/mps[=:]* ]] ||
    resources+=(--gres=gpu:1)
}
