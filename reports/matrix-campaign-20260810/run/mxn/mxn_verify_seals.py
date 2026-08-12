#!/usr/bin/env python3
"""Verify every wave-9 seal, and re-verify the whole ledger against its receipts.

Wave 8's mxl_verify_seals.py with the wave-8 cell root kept, the wave-9 root added
and the wave filter moved to 9.  The counts are computed rather than asserted, and the four rows that
have never resolved are expected to still not resolve -- they are named in
NOTES-MX-K 9 and this wave did not touch them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-9acaf38f")
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
SEARCH = [ROOT / "cells", ROOT / "wave6" / "cells", ROOT / "wave7" / "cells",
          ROOT / "wave8" / "cells", ROOT / "wave9" / "cells",
          ROOT / "wave10" / "cells"]


def _index() -> dict[str, Path]:
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
    wave9 = [r for r in rows if int(r.get("campaign_wave", 0)) == 10]
    w9_resolved = w9_verified = 0
    for row in rows:
        digest = str(row["receipt_sha"])
        path = index.get(digest)
        if path is None:
            unresolved.append(f"{row['cell_id']} {row['cell_instance_id']} {digest[:12]}")
            continue
        resolved += 1
        is9 = int(row.get("campaign_wave", 0)) == 9
        w9_resolved += 1 if is9 else 0
        body = json.loads(path.read_text(encoding="utf-8"))
        try:
            recomputed = verify_seal(body)
        except MatrixCellReceiptError:
            failed += 1
            continue
        ok = recomputed == digest
        verified += 1 if ok else 0
        failed += 0 if ok else 1
        w9_verified += 1 if (is9 and ok) else 0
        try:
            validate_receipt(body)
            mandatory_ok += 1
        except MatrixCellReceiptError:
            mandatory_bad += 1

    print(f"ledger rows                      {len(rows)}")
    print(f"  resolve to a receipt on disk   {resolved}")
    print(f"  seal verifies                  {verified}")
    print(f"  seal fails                     {failed}")
    print(f"  do not resolve                 {len(unresolved)}")
    for text in unresolved:
        print(f"    - {text}")
    print(f"satisfy today's mandatory set    {mandatory_ok}")
    print(f"do NOT (wave-1, no architecture) {mandatory_bad}")
    pairs = {(r["cell_instance_id"], r["receipt_sha"]) for r in rows}
    print(f"distinct (instance, digest)      {len(pairs)}")
    print(f"distinct receipt digests         {len({r['receipt_sha'] for r in rows})}")
    dupes = [i for i, c in Counter(r["cell_instance_id"] for r in rows).items() if c > 1]
    print(f"duplicated cell_instance_ids     {len(dupes)} {dupes}")
    print()
    print(f"wave-10 rows                      {len(wave9)}")
    print(f"  resolve / verify               {w9_resolved} / {w9_verified}")
    print("  compute architectures          "
          f"{dict(Counter(r['compute_architecture'] for r in wave9))}")
    print("  receipt_schema_version         "
          f"{dict(Counter(r.get('receipt_schema_version') for r in wave9))}")
    print("  verdicts                       "
          f"{dict(Counter(r.get('behavior_verdict') for r in wave9 if r.get('behavior_verdict')))}")
    print("  modes                          "
          f"{dict(Counter(r.get('composition_mode') for r in wave9 if r.get('composition_mode')))}")
    ratios = [float(r['stored_parameter_ratio']) for r in wave9
              if r.get('stored_parameter_ratio') is not None]
    if ratios:
        print(f"  deepest wave-9 net ratio       {min(ratios)}")
    print("  all-rows schema version        "
          f"{dict(Counter(r.get('receipt_schema_version') for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
