#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  awk 'END{print NR}'
  exit 0
fi

total=0
for f in "$@"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: file '$f' not found" >&2
    exit 1
  fi
  c=$(awk 'END{print NR}' "$f")
  total=$(( total + c ))
done

echo "$total"
