#!/usr/bin/env bash
# -*- coding: utf-8 -*-
""":"
PYTHON_BIN=${PYTHON_BIN:-python3}
exec "$PYTHON_BIN" "$0" "$@"
":"""
from __future__ import print_function
import sys, io, os

def main(argv):
    if len(argv) < 2:
        sys.stdout.write('missing file name\n')
        return 1
    if len(argv) > 2:
        sys.stdout.write('only one argument is allowed\n')
        return 1

    fname = argv[1]
    if os.path.exists(fname):
        with io.open(fname, 'r', encoding=None) as f:
            lines = f.readlines()
        sys.stdout.write('{} lines in {}\n'.format(len(lines), fname))
        return 0
    else:
        sys.stdout.write('{} not found\n'.format(fname))
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
