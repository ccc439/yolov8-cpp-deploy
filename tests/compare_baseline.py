"""Compare matched boxes, independent of output ordering."""
import sys
from pathlib import Path


def read(path):
    return [list(map(float, row.split())) for row in Path(path).read_text().splitlines() if row.strip()]


if __name__ == "__main__":
    python, cpp = map(read, sys.argv[1:3])
    if len(python) != len(cpp):
        raise SystemExit(f"Count mismatch: {len(python)} vs {len(cpp)}")
    for a in python:
        candidates = [(max(abs(x-y) for x, y in zip(a[1:5], b[1:5])), i)
                      for i, b in enumerate(cpp) if a[0] == b[0] and abs(a[5]-b[5]) <= .001]
        if not candidates:
            raise SystemExit(f"Missing class/score match: {a}")
        distance, i = min(candidates)
        if distance > .5:
            raise SystemExit(f"Coordinate error: {distance}")
        cpp.pop(i)
    print(f"PASS: {len(python)} matched boxes, coordinate error <= 0.5px, score error <= 0.001")
