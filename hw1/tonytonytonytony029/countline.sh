#! /usr/bin/env bash
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: countline.sh <file>" >&2
  exit 1
fi
if [ ! -f "$1" ]; then
  echo "error: $1 not found" >&2
  exit 2
fi
lines=$(wc -l < "$1")
lines=$(printf "%s" "$lines" | tr -dc '0-9')
echo "$lines"
