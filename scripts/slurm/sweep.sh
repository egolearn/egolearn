#!/bin/bash
# Adapted from agentic-learning-ai-lab/jepa-agent at a7124e81 (MIT; see LICENSE).
# Split only top-level, unescaped commas; preserve Hydra quotes and containers.
set -euo pipefail
rows=("")
for token in "$@"; do
  values=()
  if [[ $token =~ ^(\+\+?|~)?[A-Za-z_][A-Za-z0-9_@/.-]*= ]]; then
    key=${token%%=*}=
    value=${token#*=}
    part='' quote='' depth=0 escaped=0
    for ((i=0; i<${#value}; i++)); do
      c=${value:i:1}
      if ((escaped)); then part+=$c; escaped=0; continue; fi
      if [[ $c = "\\" ]]; then part+=$c; escaped=1; continue; fi
      if [[ -n $quote ]]; then
        part+=$c
        [[ $c != "$quote" ]] || quote=
        continue
      fi
      case "$c" in
        "'"|'"') quote=$c; part+=$c;;
        '['|'{'|'(') ((depth+=1)); part+=$c;;
        ']'|'}'|')') depth=$((depth-1)); part+=$c;;
        ',')
          if ((depth == 0)); then values+=("$key$part"); part=; else part+=$c; fi;;
        *) part+=$c;;
      esac
    done
    [[ -z $quote && $depth = 0 && $escaped = 0 ]] || { echo "invalid sweep token: $token" >&2; exit 1; }
    values+=("$key$part")
  else
    values=("$token")
  fi
  next=()
  for row in "${rows[@]}"; do
    for value in "${values[@]}"; do
      printf -v quoted '%q' "$value"
      next+=("${row:+$row }$quoted")
    done
  done
  rows=("${next[@]}")
done
printf '%s\n' "${rows[@]}"
