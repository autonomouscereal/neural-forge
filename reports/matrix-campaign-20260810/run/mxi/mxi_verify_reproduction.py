#!/usr/bin/env python3
"""Prove a wave-5 re-run of a sealed cell is the SAME solve before it is graded.

ADDENDUM 6's step 1 grades the behaviour of MLP folds that waves 1-3 already
sealed and never behaviour-reviewed.  Those folds cannot be graded from their
sealed receipts alone -- the receipts carry the free-run token ids and three
decode-smoke strings, not the folded tensors -- so the cell has to be re-run to
produce a child the frozen review can reconstitute.

That re-run is only allowed to stand in for the sealed cell if it *is* the
sealed cell.  Nothing about the pipeline guarantees that on its own: the
technique kernels moved between waves (wave 5 corrected two storage counts),
the evidence loader gained a manifest authority in wave 2, and a knob file is a
path, not a value.  So the reproduction is checked against the sealed receipt
field by field, and the fields are chosen to be the ones that would move if
anything real had changed:

* the **solve** telemetry -- solver path, conditioning, correction and prior
  norms, contraction rank.  A different basis, a different ridge or a different
  row selection moves at least one of these.
* the **boundary** scores on the held-out rows.  These are a function of the
  solved weight and the sealed evidence split; two different weights do not
  produce the same held-out cosine to 1e-9.
* the **behaviour** the campaign has always recorded -- free-run token match
  against the copy_exact control, and the composed drift.  These are what the
  old gate read, and reproducing them is what makes the sentence "the cell that
  scored 0.586 is the cell that was reviewed" true.

``stored_parameter_count`` is deliberately NOT required to match: wave 5
corrected two kernels that were undercounting a route's index side, so a T5 or
T8 reproduction *should* seal a larger count than its wave-1 instance.  The
difference is reported rather than tolerated silently.

Exit 0 when every required field matches, 1 otherwise.  Prints a table either
way -- a failed reproduction is a finding, not a crash.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

#: Fields that must agree, with the tolerance each is compared under.  A
#: relative tolerance of 0 means bit-identical.
_REQUIRED: tuple[tuple[str, float], ...] = (
    ("solve.solver_path", 0.0),
    ("solve.condition_number", 1e-9),
    ("solve.prior_norm", 1e-9),
    ("solve.correction_norm", 1e-9),
    ("solve.ridge_relative", 0.0),
    ("solve.clamp_engaged", 0.0),
    ("solve.condition_cap_saturated", 0.0),
    ("boundary.write_heldout_relL2", 1e-9),
    ("boundary.write_heldout_cos", 1e-9),
    ("boundary.post_operator_heldout_relL2", 1e-9),
    ("boundary.post_operator_heldout_cos", 1e-9),
    ("composed.terminal_heldout_relL2", 1e-9),
    ("composed.vs_copy_exact_control_delta", 1e-9),
    ("behavior.free_run.token_match_vs_control", 0.0),
    ("behavior.free_run.mean_lcp", 0.0),
    ("calibration.copy_exact_reproduction_relL2", 1e-9),
)

#: Reported, never required.  Each has a reason it is allowed to move.
_ADVISORY: tuple[tuple[str, str], ...] = (
    (
        "solve.stored_parameter_count",
        "wave 5 counts a route's index side (ADDENDUM 6 6.2)",
    ),
    ("solve.contraction_basis_rank", "absent from some wave-1 receipts"),
    ("code_identity.src_tree_sha", "a different deployment by construction"),
    ("code_identity.repo_head", "a different commit by construction"),
)


def _lookup(body: Any, path: str) -> Any:
    node = body
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _agrees(left: Any, right: Any, tolerance: float) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if tolerance == 0.0:
            return float(left) == float(right)
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance * scale)
    return left == right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--reproduced", type=Path, required=True)
    arguments = parser.parse_args()

    sealed = json.loads(arguments.sealed.read_text(encoding="utf-8"))
    fresh = json.loads(arguments.reproduced.read_text(encoding="utf-8"))
    if sealed["cell_id"] != fresh["cell_id"]:
        print(
            f"REFUSED: {arguments.reproduced} is {fresh['cell_id']}, not "
            f"{sealed['cell_id']}."
        )
        return 1

    print(f"reproduction check: {sealed['cell_id']}")
    print(f"  sealed     {arguments.sealed}  (wave {sealed['campaign_wave']}, "
          f"{sealed['receipt_sha'][:16]})")
    print(f"  reproduced {arguments.reproduced}  (wave {fresh['campaign_wave']}, "
          f"{fresh['receipt_sha'][:16]})")
    print()
    failures = 0
    for path, tolerance in _REQUIRED:
        before, after = _lookup(sealed, path), _lookup(fresh, path)
        ok = _agrees(before, after, tolerance)
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'BAD'}] {path:<52} {before!r} -> {after!r}")
    print()
    for path, reason in _ADVISORY:
        before, after = _lookup(sealed, path), _lookup(fresh, path)
        moved = "" if _agrees(before, after, 0.0) else f"   <- moved: {reason}"
        print(f"  [adv] {path:<52} {before!r} -> {after!r}{moved}")
    print()
    if failures:
        print(f"REPRODUCTION FAILED: {failures} required field(s) disagree.")
        return 1
    print("REPRODUCTION VERIFIED: every required field matches the sealed cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
