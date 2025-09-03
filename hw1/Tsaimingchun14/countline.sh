#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Missing file name" >&2
    exit 1
elif [ $# -gt 1 ]; then
    echo "Only one argument is allowed" >&2
    exit 1
fi

filename="$1"

if [ ! -f "$filename" ]; then
    echo "$filename not found" >&2
    exit 1
fi

line_count=$(wc -l < "$filename")
echo "$line_count lines in $filename"

exit 0