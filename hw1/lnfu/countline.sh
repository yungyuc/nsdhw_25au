#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "missing file name"
    exit 1
fi

if [ $# -gt 1 ]; then
    echo "only one argument is allowed"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "$1 not found"
    exit 1
fi

echo "$(wc -l < "$1") lines in $1"
