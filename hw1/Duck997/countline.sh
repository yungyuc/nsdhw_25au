#!/bin/bash

if [ $# -lt 1 ]; then
    echo "missing file name"
    exit 1
elif [ $# -gt 1 ]; then
    echo "only one argument is allowed"
    exit 1
fi

filename="$1"

if [ ! -f "$filename" ]; then
    echo "$filename not found"
    exit 1
fi

line_count=$(wc -l < "$filename")
echo "$line_count lines in $filename"
