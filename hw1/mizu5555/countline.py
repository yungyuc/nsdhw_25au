#!/bin/bash

exec ${PYTHON_BIN:-python3} - "$@" <<'EOF'

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
        fobj = open(fname)
        lines = fobj.readlines()
        fobj.close()
        sys.stdout.write('%d lines in %s\n' % (len(lines), fname))
    else:
        sys.stdout.write('{} not found\n'.format(fname))
        sys.stdout.write('%s not found\n' % (fname,))
EOF
