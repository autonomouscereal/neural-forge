#!/usr/bin/env python3
"""Which LANES survive a block excision, across the complete fourteen-block census.

A free probe, no GPU: it reads the sealed per-case readings the reviewer wrote
for every single-block child and asks a question the campaign could not ask
until wave 17 closed the census -- *are the lanes lost in a fixed order?*

Nothing here scores anything and nothing here is a gate.  It is an
attempt-ordering diagnostic in the same sense every census in this campaign is,
and it is stated as a count of children rather than as a probability of
anything.

    python mx17_lane_fragility.py <repo>/reports/matrix-campaign-20260810
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

#: the single-block children, by the parent block each one removes
BLOCKS = {
    "X00": 0, "X01": 1, "X02": 2, "X03": 3, "X04": 4, "X06": 6, "X07": 7,
    "X08": 8, "X09": 9, "X10": 10, "X11": 11, "X12": 12, "X13A": 13,
}
#: block 5's per-case reading is wave 15's and predates the per-case file, so
#: the census here is thirteen of the fourteen blocks and says so.
MISSING = {5: "ARCH.X05A, wave 15, sealed before per-case readings were written"}


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rows: dict[str, dict[str, str]] = {}
    for wave in ("wave16", "wave17"):
        for path in glob.glob(os.path.join(root, wave, "human-review", "per-case-*.json")):
            tag = os.path.basename(path)[len("per-case-"):-len(".json")]
            rows.setdefault(tag, json.loads(open(path, encoding="utf-8").read()))
    census = {tag: rows[tag] for tag in BLOCKS if tag in rows}
    survive: collections.Counter[str] = collections.Counter()
    for cases in census.values():
        for case, verdict in cases.items():
            if verdict in ("Useful", "Partial"):
                survive[case] += 1
    total = len(census)
    body = {
        "schema": "neural-forge-matrix-wave17-lane-fragility-v1",
        "children": {tag: BLOCKS[tag] for tag in census},
        "blocks_missing": MISSING,
        "lane_survivals_out_of": total,
        "lane_survivals": {
            case: survive.get(case, 0)
            for case in sorted(next(iter(census.values())))
        },
        "reading": (
            "a count of how many single-block excisions leave each lane useful "
            "or partial; it orders lanes by fragility and gates nothing"
        ),
    }
    print(json.dumps(body, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
