#!/bin/bash

if (( $# < 1 )) ; then
    echo "missing file name"
    exit 1
elif (( $# > 1 )) ; then
    echo "only one argument is allowed"
    exit 1
fi

if [[ -e "$1" ]] ; then
    echo "$( wc -l < "$1" ) lines in "$1""
else
    echo ""$1" not found"
    exit 1
fi

exit 0