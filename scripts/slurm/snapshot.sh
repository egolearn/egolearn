#!/bin/bash
# Sourced by submit: snapshot and provenance use only Bash and standard Unix tools.
snapshot_code() {
  local status entry mode
  status=$(git status --porcelain --untracked-files=all)
  [[ -z $status ]] || fail 'dirty checkout; commit the intended code before submission'
  [[ $(git rev-parse HEAD) = "$EGOLEARN_CODE_COMMIT" ]] ||
    fail 'HEAD changed during submission; resubmit'
  # Only regular files may enter the archive. Reject links before extraction;
  # NUL-delimited records preserve whitespace and newlines in tracked names.
  while IFS= read -r -d '' entry; do
    mode=${entry%% *}
    case "$mode" in
      100644|100755) ;;
      160000) fail 'submodules are not supported by the code snapshot';;
      *) fail "unsupported archive entry: $entry";;
    esac
  done < <(git ls-tree -r -z "$EGOLEARN_CODE_COMMIT")
  mkdir -p "$EGOLEARN_JOB_ROOT"
  # Set job_dir immediately so submit's EXIT trap can clean up a partial archive.
  job_dir=$(mktemp -d "$EGOLEARN_JOB_ROOT/run-XXXXXXXX")
  EGOLEARN_RUN_DIR=$job_dir
  mkdir "$job_dir/code" "$job_dir/work"
  git archive --format=tar "$EGOLEARN_CODE_COMMIT" | tar -xf - -C "$job_dir/code"
  while IFS= read -r -d '' entry; do
    ln -s "$entry" "$job_dir/work/${entry##*/}"
  done < <(find "$job_dir/code" -mindepth 1 -maxdepth 1 -print0)
  chmod -R a-w "$job_dir/code"
  : > "$job_dir/dirty.diff"
}

json_string() {
  local value=$1 char ordinal index LC_ALL=C
  printf '"'
  for ((index=0; index<${#value}; index++)); do
    char=${value:index:1}
    case "$char" in
      '"'|'\') printf '\\%s' "$char";;
      *)
        printf -v ordinal '%d' "'$char"
        if ((ordinal < 32)); then printf '\\u%04x' "$ordinal"
        else printf '%s' "$char"
        fi;;
    esac
  done
  printf '"'
}

write_run_record() {
  local arg separator=
  {
    printf '{\n  "schema_version": 1,\n  "commit": '
    json_string "$EGOLEARN_CODE_COMMIT"
    printf ',\n  "dirty": false,\n  "dirty_diff": "dirty.diff",\n  "submitted": '
    json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf ',\n  "code": "code",\n  "work": "work",\n  "image": '
    json_string "$EGOLEARN_IMAGE"
    printf ',\n  "environment_stamp": '
    json_string "$EGOLEARN_EXPECTED_STAMP"
    printf ',\n  "command": ['
    for arg in "$@"; do
      printf '%s' "$separator"
      json_string "$arg"
      separator=', '
    done
    printf '],\n  "commands_file": '
    if [[ -f $job_dir/commands.sh ]]; then printf '"commands.sh"'; else printf 'null'; fi
    printf '\n}\n'
  } > "$job_dir/run.json"
}
