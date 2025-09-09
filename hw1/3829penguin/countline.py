#!/bin/bash
''':'
exec "${PYTHON_BIN:-python3}" "$0" "$@"
':'''
from __future__ import print_function
import sys, os

def main(argv):
    if len(argv) < 2:
        sys.stdout.write('missing file name\n')
        return 1
    if len(argv) > 2:
        sys.stdout.write('only one argument is allowed\n')
        return 1

    fname = argv[1]
    if os.path.exists(fname):
        n = 0
        with open(fname, 'rb') as f:
            for _ in f:
                n += 1
        sys.stdout.write('{} lines in {}\n'.format(n, fname))
        return 0
    else:
        sys.stdout.write('{} not found\n'.format(fname))
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
