#!/usr/bin/env python3
"""Merge the two per-GPU ledger shards into the authoritative campaign ledger.

`append_ledger` writes a row with two `write()` calls under O_APPEND, so the row
and its newline are individually atomic but the pair is not: two concurrent
runners can interleave.  Wave 1's second half ran on two GPUs at once, so each
runner wrote its own shard and this is the single serialized appender that folds
them in.

The shard lines are already exactly what `append_ledger` would have written
(canonical JSON bytes plus one `\\n`), so the merge is a byte concatenation --
never a re-serialization, which could silently normalize a row.  Rows go in
chronological `sealed_at` order, which is the order they would have landed in
had the two runners shared the file safely.

Refuses unless the authoritative file's existing bytes are a prefix of the
result: the ledger is append-only and no merge may rewrite a sealed row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-9acaf38f/src")

from neural_forge.matrix_cell_receipt import LEDGER_SCHEMA  # noqa: E402

authority = Path(sys.argv[1])
shards = [Path(p) for p in sys.argv[2:]]

before = authority.read_bytes() if authority.is_file() else b""
before_rows = [line for line in before.split(b"\n") if line.strip()]

incoming: list[tuple[str, bytes]] = []
for shard in shards:
    for raw in shard.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        row = json.loads(raw.decode("utf-8"))
        if row.get("schema") != LEDGER_SCHEMA:
            raise SystemExit(f"{shard}: row is not a {LEDGER_SCHEMA} row")
        incoming.append((str(row["sealed_at"]), raw))

# sealed_at is an ISO-8601 UTC stamp; ties break on the row bytes so the merge
# is a total order and re-running it produces the same file.
incoming.sort(key=lambda pair: (pair[0], pair[1]))

payload = b"".join(raw + b"\n" for _, raw in incoming)
result = before + payload

if not result.startswith(before):
    raise SystemExit("refused: the merge would rewrite an existing ledger row")

authority.write_bytes(result)

after = authority.read_bytes()
after_rows = [line for line in after.split(b"\n") if line.strip()]
assert after_rows[: len(before_rows)] == before_rows, "prefix changed"
print(f"rows {len(before_rows)} -> {len(after_rows)} (+{len(incoming)})")
print("prefix verified byte-for-byte")
for stamp, raw in incoming:
    row = json.loads(raw.decode("utf-8"))
    print(f"  {stamp}  {row['cell_id']:24} {row['status']:14} {row['receipt_sha'][:12]}")
