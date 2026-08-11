#!/usr/bin/env python3
"""Verify every wave-7 seal, and re-verify the whole ledger against its receipts.

Three counts the wave has to be able to state, computed rather than asserted:

* how many ledger rows resolve to a receipt on disk and how many of those
  re-verify their content address;
* how many receipts satisfy today's mandatory field set (wave-1 receipts predate
  ``code_identity.compute_architecture`` and cannot, which is the documented and
  expected failure);
* the ledger's identity -- distinct ``(cell_instance_id, receipt_sha)`` pairs,
  distinct receipt digests, and the compute architecture every wave-7 row ran on.

New in wave 7: the per-row ``receipt_schema_version`` column, so the count of
rows sealed under each site registry is a fact the ledger reports about itself.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16")
if str(_DEPLOY / "src") not in sys.path:
    sys.path.insert(0, str(_DEPLOY / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    MatrixCellReceiptError,
    read_ledger,
    validate_receipt,
    verify_seal,
)

ROOT = Path("/mnt/ai_data/neural-forge-matrix")
LEDGER = ROOT / "ledger.jsonl"

# Every place a sealed receipt can live, in the order the campaign wrote them.
SEARCH = [ROOT / "cells", ROOT / "wave6" / "cells", ROOT / "wave7" / "cells"]


def _index() -> dict[str, Path]:
    """digest -> path, over every receipt json under every cell root."""

    found: dict[str, Path] = {}
    for root in SEARCH:
        if not root.is_dir():
            continue
        for path in root.rglob("cell-receipt*.json"):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            digest = body.get("receipt_sha")
            if isinstance(digest, str):
                found.setdefault(digest, path)
    return found


def main() -> int:
    rows = read_ledger(LEDGER)
    index = _index()
    resolved = verified = failed = 0
    mandatory_ok = mandatory_bad = 0
    unresolved: list[str] = []
    wave7_rows = [row for row in rows if int(row.get("campaign_wave", 0)) == 7]
    for row in rows:
        digest = str(row["receipt_sha"])
        path = index.get(digest)
        if path is None:
            unresolved.append(f"{row['cell_id']} {row['cell_instance_id']} {digest[:12]}")
            continue
        resolved += 1
        body = json.loads(path.read_text(encoding="utf-8"))
        try:
            recomputed = verify_seal(body)
        except MatrixCellReceiptError:
            failed += 1
            continue
        verified += 1 if recomputed == digest else 0
        failed += 0 if recomputed == digest else 1
        try:
            validate_receipt(body)
            mandatory_ok += 1
        except MatrixCellReceiptError:
            mandatory_bad += 1

    print(f"ledger rows                     {len(rows)}")
    print(f"  resolve to a receipt on disk  {resolved}")
    print(f"  seal verifies                 {verified}")
    print(f"  seal fails                    {failed}")
    print(f"  do not resolve                {len(unresolved)}")
    for text in unresolved:
        print(f"    - {text}")
    print(f"satisfy today's mandatory set   {mandatory_ok}")
    print(f"do NOT (wave-1, no architecture) {mandatory_bad}")
    pairs = {(row["cell_instance_id"], row["receipt_sha"]) for row in rows}
    print(f"distinct (instance, digest)     {len(pairs)}")
    print(f"distinct receipt digests        {len({row['receipt_sha'] for row in rows})}")
    duplicated = [
        instance
        for instance, count in Counter(row["cell_instance_id"] for row in rows).items()
        if count > 1
    ]
    print(f"duplicated cell_instance_ids    {len(duplicated)} {duplicated}")
    print()
    print(f"wave-7 rows                     {len(wave7_rows)}")
    print(
        "  compute architectures         "
        f"{dict(Counter(row['compute_architecture'] for row in wave7_rows))}"
    )
    print(
        "  receipt_schema_version        "
        f"{dict(Counter(row.get('receipt_schema_version') for row in wave7_rows))}"
    )
    print(
        "  all-rows schema version       "
        f"{dict(Counter(row.get('receipt_schema_version') for row in rows))}"
    )
    print(
        "  verdicts                      "
        f"{dict(Counter(row.get('behavior_verdict') for row in wave7_rows if row.get('behavior_verdict')))}"
    )
    worst = max(
        (float(row["stored_parameter_ratio"]) for row in wave7_rows
         if row.get("stored_parameter_ratio") is not None),
        default=None,
    )
    print(f"  worst wave-7 net stored ratio {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
