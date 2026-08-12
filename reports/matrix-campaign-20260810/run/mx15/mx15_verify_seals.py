#!/usr/bin/env python3
"""Verify every seal, with the search roots read against the wave directories.

This is wave 13's `mxq_verify_seals.py` with **every root kept and two added**,
and the reason it is a new file rather than an edit is worth stating: the
*committed* wave-13 verifier is missing `wave12/cells`.

`NOTES-MX-Q` §8 says wave 13 caught that defect ("the wave12 -> wave13 script
derivation dropped `wave12/cells` from SEARCH ... caught here by reading the
roots before running"), and the copy that ran on the host does have it — but the
fix never reached the repository, so the file the next wave would derive from is
the broken one.  That is the fourth wave in a row this defect has appeared and
the first time it has been visible in git.  The roots below are not inherited
from anything: they are the directories that actually exist under
`/mnt/ai_data/neural-forge-matrix`, and the script **fails loudly** if it finds a
`*/cells` directory that is not in the list.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-1c197ed4")
if str(_DEPLOY / "src") not in sys.path:
    sys.path.insert(0, str(_DEPLOY / "src"))

from neural_forge.matrix_cell_receipt import verify_seal  # noqa: E402

ROOT = Path("/mnt/ai_data/neural-forge-matrix")
LEDGER = ROOT / "ledger.jsonl"
SEARCH = [
    ROOT / "cells",
    ROOT / "wave6" / "cells",
    ROOT / "wave7" / "cells",
    ROOT / "wave8" / "cells",
    ROOT / "wave9" / "cells",
    ROOT / "wave10" / "cells",
    ROOT / "wave11" / "cells",
    ROOT / "wave12" / "cells",
    ROOT / "wave13" / "cells",
    ROOT / "wave14" / "cells",
    ROOT / "wave15" / "cells",
]


def audit_roots() -> list[Path]:
    """Every `*/cells` directory on disk must be a declared root."""
    found = {p for p in ROOT.glob("*/cells") if p.is_dir()}
    if (ROOT / "cells").is_dir():
        found.add(ROOT / "cells")
    missing = sorted(str(p) for p in found - set(SEARCH))
    if missing:
        raise SystemExit(
            "REFUSED: these cell roots exist on disk and are not in SEARCH, which is "
            f"the defect that has cost four waves a wrong count: {missing}"
        )
    return sorted(found)


def main() -> int:
    present = audit_roots()
    print(f"roots audited against disk: {len(present)} present, 0 undeclared")
    index: dict[str, Path] = {}
    for root in SEARCH:
        if not root.is_dir():
            continue
        for path in root.rglob("cell-receipt*.json"):
            try:
                receipt = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            key = str(receipt.get("cell_instance_id", ""))
            if key:
                index.setdefault(key, path)

    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    resolved = verified = failed = 0
    unresolved: list[str] = []
    for row in rows:
        key = str(row.get("cell_instance_id", ""))
        path = index.get(key)
        if path is None:
            unresolved.append(f"{row.get('cell_id')} {key} {str(row.get('receipt_sha'))[:12]}")
            continue
        resolved += 1
        receipt = json.loads(path.read_text())
        try:
            verify_seal(receipt)
            verified += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  SEAL FAIL {row.get('cell_id')} {key}: {exc}")

    print(f"ledger rows                    {len(rows)}")
    print(f"  resolve to a receipt on disk {resolved}")
    print(f"  seal verifies                {verified}")
    print(f"  seal fails                   {failed}")
    print(f"  do not resolve               {len(unresolved)}")
    for item in unresolved:
        print(f"    - {item}")
    print(f"waves represented              {Counter(r.get('campaign_wave') for r in rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
