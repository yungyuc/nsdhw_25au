#!/bin/bash
#
# countline.sh -- Bash version of countline.py
#

# "$#"為參數個數

if [ $# -lt 1 ]; then
    echo "missing file name"
    exit 1
fi

if [ $# -gt 1 ]; then
    echo "only one argument is allowed"
    exit 1
fi
# "$1"為第一個參數
file="$1"
#檢查檔案是否存在
if [ ! -f "$file" ]; then
    echo "$file not found"
    exit 1
fi


# 計算行數
lines=$(wc -l < "$file")
echo "$lines lines in $file"
