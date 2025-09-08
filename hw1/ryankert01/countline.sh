#!/bin/bash

# if num of args < 2, print "missing file name"
if [ $# -lt 1 ]; then
    echo "missing file name"
    exit 1
fi

# if num of args > 2, print "only one argument is allowed"
if [ $# -gt 1 ]; then
    echo "only one argument is allowed"
    exit 1
fi

# if file does not exist, print "file not found"
if [ ! -f $1 ]; then
    echo "file not found"
    exit 1
fi

# print number of lines in file
echo $(wc -l $1)
exit 0