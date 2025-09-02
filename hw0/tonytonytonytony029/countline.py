#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print("usage: countline.py <file>", file=sys.stderr)
        sys.exit(1)
    try:
        with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
            count = sum(1 for _ in f)
        print(count)
        sys.exit(0)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
