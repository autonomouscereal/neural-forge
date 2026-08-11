#!/usr/bin/env python3
"""Is every dossier's parent lane the same sixteen responses, byte for byte?

ADDENDUM 6 scores a child against the parent decoded IN THE SAME PROCESS, so the
parent lane is re-generated for every child and is therefore a free, continuous
calibration: if it ever moves, the review surface moved, and every count on
either side of the move is a comparison between two experiments.

Wave 7 checked this once against wave 6 and found the sixteen responses
byte-identical.  Wave 8 checks it on every dossier it seals, against wave 7's
own MXK14 dossier as the anchor.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def lane_digest(path: Path) -> tuple[str, str, list[str]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    lane = body["parent_responses"]
    payload = json.dumps(
        [[row["id"], row["response_verbatim"]] for row in lane],
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    proto = body["protocol"]["protocol_sha256"]
    pack = body["protocol"]["prompt_pack_sha256"]
    return hashlib.sha256(payload).hexdigest(), f"{proto[:12]}/{pack[:12]}", [
        f"{row['id']}:{row['generated_tokens']}" for row in lane
    ]


def main() -> int:
    anchor = None
    for text in sys.argv[1:]:
        path = Path(text)
        digest, surface, shape = lane_digest(path)
        if anchor is None:
            anchor = digest
        flag = "ANCHOR" if digest == anchor else "*** PARENT LANE MOVED ***"
        print(f"{digest[:24]}  {surface}  {flag}  {path.parent.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
