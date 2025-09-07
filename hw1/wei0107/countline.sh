#!/usr/bin/env bash
# countline.sh — reimplement countline.py using bash only (no Python)

set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'missing file name\n'
  exit 0
elif [ "$#" -gt 1 ]; then
  printf 'only one argument is allowed\n'
  exit 0
fi

fname=$1
if [ -e "$fname" ]; then
  # Use awk to count lines; this also counts the last line even without a trailing newline.
  n=$(awk 'END{print NR}' "$fname")
  printf '%s lines in %s\n' "$n" "$fname"
else
  printf '%s not found\n' "$fname"
fi
