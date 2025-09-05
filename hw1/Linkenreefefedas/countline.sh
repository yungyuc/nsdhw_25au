#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "missing file name"
    exit 1
elif [ $# -gt 1 ]; then
    echo "only one argument is allowed"
    exit 1
fi

fname="$1"
if [ ! -f "$fname" ]; then
    echo "file not exists"
    exit 1
fi

# 用 wc -l 計算行數
wc -l < "$fname"