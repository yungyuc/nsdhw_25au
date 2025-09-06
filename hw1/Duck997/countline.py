#!/bin/bash

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "exec: $PYTHON_BIN: not found" >&2
    exit 1
fi

exec "$PYTHON_BIN" - "$@" << 'EOF'
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
        
EOF
