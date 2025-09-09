#!/bin/sh
""":"
PY_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "Error: Python binary '$PY_BIN' not found"
    exit 1
fi

exec "$PY_BIN" "$0" "$@"
"""

import sys
import os.path

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