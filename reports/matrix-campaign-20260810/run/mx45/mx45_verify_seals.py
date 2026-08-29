#!/usr/bin/env python3
"""Verify every seal, with the search roots read against the wave directories.

WAVE 45.  Same edit, same reason: **40 declared roots become 41, `wave44/cells`
still declared, `wave45/cells` appended.**  `cut_wave45.py --audit` asserts
twelve roots by name -- waves 34 through 45.  Third consecutive wave to write a
paragraph rather than only a line, which is the practice waves 39 through 42
let lapse.  Nothing here is new; the edit is one line and the reason is wave
30's deleted seal root.

--- wave 44's note, kept ---

WAVE 44.  Same edit, same reason: **39 declared roots become 40, `wave43/cells`
still declared, `wave44/cells` appended.**  `cut_wave44.py --audit` asserts
eleven roots by name -- waves 34 through 44.  Wave 43 restarted this file's
paragraph stack after four waves of appending a LINE without a PARAGRAPH; this
is the second consecutive wave to write one, which is the point at which the
practice is a practice again rather than a repair.  Nothing here is new: the
edit is one line, the reason is wave 30's deleted seal root, and the only claim
this paragraph makes is that the edit was made deliberately and read back.

--- wave 43's note, kept ---

WAVE 43.  Same edit, same reason: **38 declared roots become 39, `wave42/cells`
still declared, `wave43/cells` appended.**  `cut_wave43.py --audit` asserts ten
roots by name -- waves 34 through 43.  One thing is worth recording because it
is checkable from this file alone: the header below is a stack that stops at
wave 38, and waves 39, 40, 41 and 42 each appended their LINE without appending
their PARAGRAPH.  Nothing was lost -- `audit_roots()` compares the declared list
against what is on disk, so a missing root is caught by the code and not by the
prose -- but the file's own stated practice is a paragraph per wave, and four
waves in a row not writing one is the same shape of drift as wave 42's stale
calibration block.  This paragraph restarts the stack rather than editing a
count inside somebody else's.

--- wave 38's note, kept ---

WAVE 38.  This file is in the wave-38 cutter's HAND_EDIT set, so the tree cut
copied it byte for byte and ran no substitution over it; the only edit is ONE
APPENDED LINE.  **33 declared roots become 34, `wave37/cells` still declared,
`wave38/cells` appended.**  `cut_wave38.py --audit` asserts six roots by
name -- waves 33, 34, 35, 36, 37 and this wave's -- which is the mechanical form
of the paragraph below.  Waves 36 and 37 each wrote their own paragraph rather
than editing the count inside their predecessor's; this wave does the same, so
the header is a stack of what each wave appended and not a number that has been
overwritten five times.

--- wave 37's note, kept ---

WAVE 37.  This file is in the wave-37 cutter's HAND_EDIT set, so the tree cut
copied it byte for byte and ran no substitution over it; the only edit is ONE
APPENDED LINE.  **32 declared roots become 33, `wave36/cells` still declared,
`wave37/cells` appended.**  `cut_wave37.py --audit` asserts six roots by
name -- waves 32, 33, 34, 35, 36 and this wave's -- which is the mechanical form
of the paragraph below.  Wave 36 corrected the stale-header failure mode that
waves 34 and 35 carried; this wave keeps the correction by writing its own
paragraph rather than editing the count in wave 36's.

--- wave 36's note, kept ---

WAVE 36.  This file is in the wave-36 cutter's HAND_EDIT set, so the tree cut
copied it byte for byte and ran no substitution over it; the only edit is ONE
APPENDED LINE.  **31 declared roots become 32, `wave35/cells` still declared,
`wave36/cells` appended.**  `cut_wave36.py --audit` asserts five roots by
name -- waves 32, 33, 34, 35 and this wave's -- which is the mechanical form of
the paragraph below.  Wave 35 appended its own root and left the header saying
wave 34; the root was right and the paragraph was stale, which is the smaller of
the two failure modes and is corrected here rather than carried a third time.

--- wave 34's note, kept ---

WAVE 34.  This file is in the wave-34 cutter's HAND_EDIT set, so the tree cut
copied it byte for byte and ran no substitution over it; the only edit is ONE
APPENDED LINE.  **29 declared roots become 30, `wave33/cells` still declared,
`wave34/cells` appended.**  `cut_wave34.py --audit` asserts three roots by
name -- wave 32's, wave 33's and this wave's -- which is the mechanical form of
the paragraph below.  The warning is now on its fifteenth issue and has not
gained a violation since wave 30.

--- wave 33's note, kept ---

WAVE 33.  The tree cut copied this file byte for byte; the only edit was ONE
APPENDED LINE.  28 declared roots became 29, `wave32/cells` still declared,
`wave33/cells` appended.

--- wave 31's note, kept ---

Wave 30's verifier with `wave31/cells` APPENDED and every earlier root kept.
The warning below is on its thirteenth issue with TWO violations behind it, and
wave 31 is the first wave since 27 to act on the lesson rather than record it
again: this file is in `cut_wave31.py`'s HAND_EDIT set, so the tree cut copied
it byte for byte and never ran a substitution over it.  26 declared roots become
27, `wave30/cells` still declared.  `cut_wave31.py --audit` asserts BOTH roots
are present, which is the mechanical form of the paragraph below.

--- wave 30's note, kept because it is the third violation's record ---

The warning below is on its twelfth issue, and it now has TWO violations behind
it.  Wave 30 cut its tree with a blanket rename and the rename rewrote
`wave29/cells` into `wave30/cells` -- wave 20's mistake and wave 25's mistake,
made a third time, on the file whose docstring exists to forbid it.  It was
caught by reading this list before the file was ever run, and the correct edit
was made: `wave29/cells` restored, `wave30/cells` appended, 24 declared roots
became 26.  The lesson wave 26 and 27 acted on -- carry this file in an explicit
hand-edit set and never let the rename touch it -- was not applied by wave 30,
and the audit function is the only reason that is a note rather than a wrong
count.

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
    ROOT / "wave30" / "cells",
    ROOT / "wave31" / "cells",
    ROOT / "wave32" / "cells",
    ROOT / "wave33" / "cells",
    ROOT / "wave34" / "cells",
    ROOT / "wave35" / "cells",
    ROOT / "wave36" / "cells",
    ROOT / "wave37" / "cells",
    ROOT / "wave38" / "cells",
    ROOT / "wave39" / "cells",
    ROOT / "wave40" / "cells",
    ROOT / "wave41" / "cells",
    ROOT / "wave42" / "cells",
    ROOT / "wave43" / "cells",
    ROOT / "wave44" / "cells",
    ROOT / "wave45" / "cells",
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
