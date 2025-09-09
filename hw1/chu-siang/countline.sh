#!/bin/bash

if [ $# -lt 1 ]; then
    echo "missing file name"
elif [ $# -gt 1 ]; then 
    echo "only one argument is allowed"
else
    tmp_n="$1"
    if [ -f "$tmp_n" ]; then
        tmp_l=$(wc -l < "$tmp_n")
        echo "$tmp_l lines in $tmp_n"
    else 
        echo "$tmp_n not found"
    fi
fi

