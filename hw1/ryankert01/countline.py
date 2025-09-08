#!/usr/bin/env bash

# Note: the bash script should be inside of ''':', ':'''
# : is bash built-in no operation command, and it's inside of ''':''' will be ignored by python
''':'
PYTHON_BIN=${PYTHON_BIN:-python}
exec $PYTHON_BIN $0 $1
exit 0
':'''


import sys
import os.path
import os


if len(sys.argv) < 2:
    sys.stdout.write('missing file name\n')
elif len(sys.argv) > 2:
    sys.stdout.write('only one argument is allowed\n')
else:
    fname = sys.argv[1]
    if os.path.exists(fname):
        with open(fname) as fobj:
            lines = fobj.readlines()
        sys.stdout.write('{} lines in {}\n'.format(len(lines), fname))
    else:
        sys.stdout.write('{} not found\n'.format(fname))
