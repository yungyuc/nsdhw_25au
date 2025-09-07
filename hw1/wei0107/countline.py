#!/usr/bin/env bash
# Polyglot header: run under bash first to pick interpreter via $PYTHON_BIN,
# then re-exec this same file with that Python. When run by python{2,3} directly,
# the block between """: and ":""" is just a Python triple-quoted string and is ignored.
""":"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$0" "$@"
":"""

# ---- Python code (compatible with both py2 & py3) ----
from __future__ import print_function  # safe on py2; no-op on py3
import sys
import os

def main(argv):
    if len(argv) < 2:
        sys.stdout.write('missing file name\n')
        return 0
    if len(argv) > 2:
        sys.stdout.write('only one argument is allowed\n')
        return 0

    fname = argv[1]
    if os.path.exists(fname):
        # read as binary so py2/py3 behave consistently on newlines
        try:
            if sys.version_info[0] < 3:
                fobj = open(fname, 'rb')
            else:
                fobj = open(fname, 'rb')
            with fobj as f:
                # Count lines by iterating (memory-friendly) but preserves same result as readlines()
                count = sum(1 for _ in f)
            sys.stdout.write('{} lines in {}\n'.format(count, fname))
        except IOError:
            sys.stdout.write('{} not found\n'.format(fname))
    else:
        sys.stdout.write('{} not found\n'.format(fname))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
