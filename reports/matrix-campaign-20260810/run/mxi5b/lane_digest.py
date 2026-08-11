#!/usr/bin/env python3
"""Digest each dossier's child and parent lanes as one concatenated text blob.

Wave 5A observed that T2 reproduces T1 to the last digit on the MLP axes.  If
two cells' child lanes digest identically then they are the same sixteen
responses and grading both is grading the same text twice; the equivalence is
recorded in both receipts instead.  The parent lane is digested for the
opposite reason: it must be identical everywhere, because ADDENDUM 6 decodes
the parent in the same process as the child and a parent that moved would
invalidate the band the child is scored against.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/cells")
CHILD = "child_responses"
PARENT = "parent_responses"


def digest(path: Path, lane: str) -> str:
    body = json.loads(path.read_text(encoding="utf-8"))
    joined = "\x00".join(row["response_verbatim"] for row in body[lane])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> int:
    for cell in sys.argv[1:]:
        path = ROOT / cell / "frozen-review-dossier.json"
        if not path.is_file():
            print("%-26s MISSING" % cell)
            continue
        print("%-26s child=%s  parent=%s" % (
            cell, digest(path, CHILD)[:16], digest(path, PARENT)[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
