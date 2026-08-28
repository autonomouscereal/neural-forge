#!/usr/bin/env python3
"""Verify every seal, with the search roots read against the wave directories.

Wave 28's verifier with `wave29/cells` APPENDED and every earlier root kept.
The warning below is on its tenth issue, with one violation behind it; wave 29
cut its tree with `cut_wave29.py`, which carries this file in an explicit
hand-edit set and ASSERTS afterwards that the wave-28 text survived the cut.

**Wave 25 DID repeat wave 20's mistake, caught it here, and recorded it rather
than quietly fixing it; wave 26 did not repeat it, because it never ran the
rename over this file at all.**  Wave 20 cut its tree from wave 19's by a blanket
rename, which rewrote `wave19/cells` to `wave20/cells` in the list below and
silently deleted last wave's root; the audit function caught it, because that
root exists on disk and was no longer declared.  Waves 21 and 22 cut their trees
the same way and **excluded this file from the rename by name**, then added one
line each.  Wave 25 ran the blanket rename over the whole tree INCLUDING this
file, which turned `wave24/cells` into `wave25/cells` -- the exact deletion the
paragraph above forbids -- and the line was restored by hand before the file was
run once.  Waves 26 and 27 each cut their tree with a copier carrying an explicit
hand-edit set with this file in it, so the bytes arrived here verbatim and the
single line below is the only edit: 21 declared roots became 22, `wave26/cells`
still declared, `wave27/cells` appended.  The warning is now on its eighth issue,
with one violation behind it.
The correct edit is, as it has been since wave 15, ONE ADDED LINE.  With the one
thing wave 15 still hardcoded made an argument: the deployment whose `src`
supplies `verify_seal`.  A verifier pinned to a previous wave's deployment either
breaks when that deployment is pruned under the current+1 policy or, worse,
verifies this wave's seals with a previous wave's schema code.

The root list is not inherited from anything: it is the directories that
actually exist under `/mnt/ai_data/neural-forge-matrix`, and the script **fails
loudly** if it finds a `*/cells` directory that is not in the list.  That defect
(`wave12/cells` silently absent) cost four waves a wrong count and was only
visible in git at wave 15.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument(
    "--deployment",
    required=True,
    help="the deployment whose src/ supplies the receipt schema code",
)
_ARGS = _ap.parse_args()
_DEPLOY = Path(_ARGS.deployment)
if not (_DEPLOY / "src" / "neural_forge" / "matrix_cell_receipt.py").is_file():
    raise SystemExit(f"REFUSED: {_DEPLOY} has no src/neural_forge/matrix_cell_receipt.py")
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
    ROOT / "wave16" / "cells",
    ROOT / "wave17" / "cells",
    ROOT / "wave18" / "cells",
    ROOT / "wave19" / "cells",
    ROOT / "wave20" / "cells",
    ROOT / "wave21" / "cells",
    ROOT / "wave22" / "cells",
    ROOT / "wave23" / "cells",
    ROOT / "wave24" / "cells",
    ROOT / "wave25" / "cells",
    ROOT / "wave26" / "cells",
    ROOT / "wave27" / "cells",
    ROOT / "wave28" / "cells",
    ROOT / "wave29" / "cells",
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
