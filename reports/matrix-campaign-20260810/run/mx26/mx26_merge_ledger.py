#!/usr/bin/env python3
"""Merge wave 18's ledger shards into the authoritative campaign ledger, once.

Same contract as `run/merge_shards.py`: the shard lines are already exactly what
`append_ledger` wrote (canonical JSON bytes plus one ``\\n``), so the merge is a
byte concatenation and never a re-serialization, rows go in ``sealed_at`` order,
and the merge is refused unless the authority's existing bytes are a prefix of
the result.

Wave 15 added one rule, because wave 15 needed it, and wave 18 keeps it.  A child that was sealed twice
-- a re-seal after a fixed sealer, a review re-run against a re-sealed receipt --
leaves two run rows in its shard for one child.  Appending both would put two
rows in the ledger for one measurement.  So:

* rows are grouped by ``cell_instance_id``;
* within a group, rows are **deduplicated by ``receipt_sha``**;
* if a group still has more than one row, exactly one is kept -- the one whose
  ``receipt_sha`` a sealed **behaviour verdict** names in
  ``addendum6.reviewed_from_receipt_sha`` -- and the others are reported as
  superseded, by digest, in the merge report.  A group with more than one row and
  no verdict to disambiguate it is a **refusal**, not a guess.

The verdict receipts are read from disk (``--verdict-receipt``) rather than from
the ledger rows, because ``ledger_line`` does not carry the reviewed-from digest
and inventing a place for it in the ledger schema to serve a merge would be the
tail wagging the dog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path, required=True)
    ap.add_argument("--shard", type=Path, action="append", required=True)
    ap.add_argument("--verdict-receipt", type=Path, action="append", default=[])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    before = args.authority.read_bytes() if args.authority.is_file() else b""
    before_rows = [line for line in before.split(b"\n") if line.strip()]

    groups: dict[str, list[tuple[str, bytes, dict]]] = {}
    reviewed_from: set[str] = set()
    for path in args.verdict_receipt:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = (body.get("addendum6") or {}).get("reviewed_from_receipt_sha")
        if digest:
            reviewed_from.add(str(digest))
    for shard in args.shard:
        for raw in shard.read_bytes().split(b"\n"):
            if not raw.strip():
                continue
            row = json.loads(raw.decode("utf-8"))
            if row.get("schema") != "neural-forge-lfm2-matrix-cell-ledger-v1":
                raise SystemExit(f"{shard}: row is not a campaign ledger row")
            groups.setdefault(str(row["cell_instance_id"]), []).append(
                (str(row["sealed_at"]), raw, row)
            )

    kept: list[tuple[str, bytes]] = []
    superseded: list[dict] = []
    for instance, rows in sorted(groups.items()):
        by_digest = {row["receipt_sha"]: (stamp, raw, row) for stamp, raw, row in rows}
        if len(by_digest) == 1:
            stamp, raw, _ = next(iter(by_digest.values()))
            kept.append((stamp, raw))
            continue
        pointed = [digest for digest in by_digest if digest in reviewed_from]
        if len(pointed) != 1:
            raise SystemExit(
                f"REFUSED: {instance} has {len(by_digest)} distinct receipts "
                f"{sorted(d[:12] for d in by_digest)} and "
                f"{len(pointed)} of them are named by a sealed verdict; a merge "
                "may not guess which measurement is the real one."
            )
        stamp, raw, _ = by_digest[pointed[0]]
        kept.append((stamp, raw))
        superseded.extend(
            {"cell_instance_id": instance, "receipt_sha": digest}
            for digest in sorted(by_digest)
            if digest != pointed[0]
        )

    kept.sort(key=lambda pair: (pair[0], pair[1]))
    payload = b"".join(raw + b"\n" for _, raw in kept)
    result = before + payload
    if not result.startswith(before):
        raise SystemExit("refused: the merge would rewrite an existing ledger row")

    report = {
        "schema": "neural-forge-matrix-wave26-ledger-merge-v1",
        "authority": str(args.authority),
        "shards": [str(s) for s in args.shard],
        "rows_before": len(before_rows),
        "rows_appended": len(kept),
        "rows_after": len(before_rows) + len(kept),
        "superseded_rows_dropped": superseded,
        "appended_cells": sorted({json.loads(raw)["cell_id"] for _, raw in kept}),
        "applied": bool(args.apply),
    }
    if args.apply:
        args.authority.write_bytes(result)
        after = args.authority.read_bytes()
        after_rows = [line for line in after.split(b"\n") if line.strip()]
        if after_rows[: len(before_rows)] != before_rows:
            raise SystemExit("the prefix changed after the write; the ledger is damaged")
        report["rows_after_verified"] = len(after_rows)
    args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
