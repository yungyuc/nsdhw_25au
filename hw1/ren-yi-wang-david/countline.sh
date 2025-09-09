#!/bin/bash
if (( $# < 1 )); then
    echo "missing file name"
fi
if (( $# > 1 )); then
    echo "only one argument is allowed"
fi
file_name="$1"
if [[ -f "$file_name" ]]; then
    n=$(wc -l < "$file_name")
    echo "$n lines in $file_name"
else
    echo "$file_name not found"
fi