#!/usr/bin/env python3
"""Row-for-row diff of two parity receipts taken with identical arguments.

Two IQ4_XS parity passes over the same GGUF, under greedy decode (temperature 0, top_k 1,
fixed seed), produced different continuations across two SERVER LAUNCHES. This answers the
narrower question the receipts could not: do two passes against ONE running server agree?

Reports what it found. It sets no threshold -- "reproducible enough" is Cyril's line, not
this script's.
"""
from __future__ import annotations

import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, {row["id"]: row for row in payload["rows"]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        print("usage: compare_parity_passes.py <receipt-a.json> <receipt-b.json>",
              file=sys.stderr)
        return 2
    left_payload, left = load(argv[0])
    right_payload, right = load(argv[1])

    shared = [key for key in left if key in right]
    differing = []
    for key in shared:
        a = (left[key].get("gguf") or {}).get("content", "")
        b = (right[key].get("gguf") or {}).get("content", "")
        if a != b:
            prefix = 0
            while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
                prefix += 1
            differing.append((key, prefix, a, b))

    print("pass A: %s (%s)" % (argv[0], left_payload.get("created_at")))
    print("pass B: %s (%s)" % (argv[1], right_payload.get("created_at")))
    print("server props A: %s" % json.dumps(left_payload.get("server_props"), sort_keys=True))
    print("server props B: %s" % json.dumps(right_payload.get("server_props"), sort_keys=True))
    print("rows compared: %d | rows whose GGUF answer differs: %d"
          % (len(shared), len(differing)))
    for key, prefix, a, b in differing:
        print("  %-8s diverges at char %d" % (key, prefix))
        print("     A: %r" % a[prefix:prefix + 70])
        print("     B: %r" % b[prefix:prefix + 70])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
