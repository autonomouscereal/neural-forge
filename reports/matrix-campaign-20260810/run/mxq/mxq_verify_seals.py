#!/usr/bin/env python3
"""Verify every wave-12 seal, and re-verify the whole ledger against its receipts.

Wave 11's mxo_verify_seals.py with the wave-12 root ADDED and every earlier root
KEPT.  The "kept" is the whole point: wave 11 lost eleven wave-10 rows to a
wave10 -> wave11 search-root rewrite and the seal count read 427/442 until the
root was put back (NOTES-MX-O 8).  Wave 12 does not repeat it, and this comment
is here so the next wave does not either.

The counts are computed rather than asserted, and the five rows that have never
resolved are expected to still not resolve -- they are named in NOTES-MX-O 8 and
this wave did not touch them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8")
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
# ADDENDUM 12 / wave 15: ``wave12/cells`` is restored here.  NOTES-MX-Q 8 says
# wave 13 caught its own derivation dropping that root and fixed it ON THE HOST;
# the fix never reached the repository, so this committed file has been the
# broken one for two waves and is what a later wave would have derived from
# again.  The authoritative check is `run/mx15/mx15_verify_seals.py`, which reads
# the roots off disk and REFUSES on an undeclared one instead of trusting a list.
SEARCH = [ROOT / "cells", ROOT / "wave6" / "cells", ROOT / "wave7" / "cells",
          ROOT / "wave8" / "cells", ROOT / "wave9" / "cells",
          ROOT / "wave10" / "cells", ROOT / "wave11" / "cells",
          ROOT / "wave12" / "cells", ROOT / "wave13" / "cells"]


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
    this_wave = [r for r in rows if int(r.get("campaign_wave", 0)) == 12]
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
    print(f"wave-12 rows                      {len(this_wave)}")
    print(f"  resolve / verify               {w9_resolved} / {w9_verified}")
    print("  compute architectures          "
          f"{dict(Counter(r['compute_architecture'] for r in this_wave))}")
    print("  receipt_schema_version         "
          f"{dict(Counter(r.get('receipt_schema_version') for r in this_wave))}")
    print("  verdicts                       "
          f"{dict(Counter(r.get('behavior_verdict') for r in this_wave if r.get('behavior_verdict')))}")
    print("  modes                          "
          f"{dict(Counter(r.get('composition_mode') for r in this_wave if r.get('composition_mode')))}")
    ratios = [float(r['stored_parameter_ratio']) for r in this_wave
              if r.get('stored_parameter_ratio') is not None]
    if ratios:
        print(f"  deepest wave-12 net ratio       {min(ratios)}")
    print("  all-rows schema version        "
          f"{dict(Counter(r.get('receipt_schema_version') for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
