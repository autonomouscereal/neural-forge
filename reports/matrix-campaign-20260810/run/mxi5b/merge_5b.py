#!/usr/bin/env python3
"""Merge wave 5B's ledger shards into the campaign ledger, in a defined order.

Each GPU lane writes its own shard because `append_ledger` issues two writes per
row and only the individual write is atomic, so two lanes appending to one file
can interleave a row.  Merging is therefore a separate, ordered step rather than
a side effect of running.

One row is dropped, and it is the only row this wave removes from anything.
C06.T2.S4.L12 was sealed twice: the first seal's stdout was truncated by a
`tail`, the seal was re-run, and `write_receipt` overwrote
``cell-receipt-behavior-reviewed.json`` with the second instance.  The first
row therefore names a ``receipt_sha`` that no file on disk carries -- a dangling
reference into an append-only ledger.  The campaign ledger has not seen either
row yet, so dropping the dangling one from the staging shard is not rewriting
history; it is declining to publish a pointer to nothing.  Both readings were
identical (10/15, alive-parent-band), so nothing about the result changes.
"""

import json
import shutil
import sys
from pathlib import Path

WORK = Path("/home/cereal/nf-mxi")
LEDGER = Path("/mnt/ai_data/neural-forge-matrix/ledger.jsonl")
SHARDS = ("ledger-gpu0.jsonl", "ledger-gpu4.jsonl", "ledger-reviews-5b.jsonl")
DANGLING = ("C06.T2.S4.L12", "5c6f44c4bba62f85")


def main() -> int:
    existing = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {(row.get("cell_id"), row.get("receipt_sha")) for row in existing}
    print("campaign ledger rows before: %d" % len(existing))

    added = []
    for name in SHARDS:
        path = WORK / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("cell_id"), row.get("receipt_sha"))
            if key in seen:
                continue
            if key == DANGLING:
                print("dropping dangling duplicate review row for %s (%s)" % DANGLING)
                continue
            seen.add(key)
            added.append(row)

    added.sort(key=lambda row: (str(row.get("sealed_at")), str(row.get("cell_id"))))
    backup = LEDGER.with_name("ledger.jsonl.pre-mxi5b-%s" % added[-1]["sealed_at"].replace(":", "").replace("-", "")[:15])
    shutil.copy2(LEDGER, backup)
    print("backup -> %s" % backup)

    with LEDGER.open("a", encoding="utf-8") as handle:
        for row in added:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    total = len(existing) + len(added)
    print("appended %d rows, campaign ledger rows after: %d" % (len(added), total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
