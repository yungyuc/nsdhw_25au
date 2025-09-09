#!/usr/bin/env bash
""":"
pybin="${PYTHON_BIN:-python3}"
exec "$pybin" "$0" "$@"
":"""

from __future__ import print_function
import sys

def main():
    if len(sys.argv) != 2:
        print("usage: countline.py <file>", file=sys.stderr)
        return 1
    try:
        total = 0
        with open(sys.argv[1], "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                total += chunk.count(b"\n")
        print(total)
        return 0
    except Exception as e:
        print("error: {0}".format(e), file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())