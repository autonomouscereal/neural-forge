#!/usr/bin/env python3
"""Build one wave-6 (MX-J) greedy-assembly recipe from an explicit cell list.

Wave 4's `derive_matrix_composition_recipe.py` selects winners out of the
ledger under ADDENDUM 3, whose admission floor is a **free-run** of 0.60.  Every
wave-5 S4 survivor free-runs between 0.066 and 0.249, so that deriver would
refuse all five of them -- which is the whole point of ADDENDUM 6: the grade is
the frozen review, and the token metric is a diagnostic column.  So wave 6's
recipes name their components explicitly and carry the *behavioural* selection
reason, and this script's job is to make the resulting file structurally
identical to a derived one so `validate_recipe` is the same gate it always was.

The isolation block is read out of the sealed wave-5 receipt by digest wherever
one exists.  A phase-2 expansion site has no sealed isolation cell -- that is
exactly what the expansion is measuring -- and says so in its own reason rather
than borrowing the numbers of the site that was measured.

Usage::

    python mxj_recipe.py --variant MXJ.S1 \
        --cell C13.T8.S4.L05:639 --cell C07.T5.S4.L12:121 \
        --reason-file reasons.json --out recipe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-eb32153e")
if str(_DEPLOY / "src") not in sys.path:
    sys.path.insert(0, str(_DEPLOY / "src"))

from neural_forge.matrix_cell_receipt import COMPONENTS, CellId  # noqa: E402
from neural_forge.matrix_composition import (  # noqa: E402
    RECIPE_SCHEMA,
    component_weight_keys,
    evidence_cache_name,
    evidence_endpoints,
    install_order_key,
    validate_recipe,
)

CELLS = Path("/mnt/ai_data/neural-forge-matrix/cells")

#: The wave-5 (NOTES-MX-I §5) behavioural ladder, verbatim, so a wave-6 recipe
#: carries the review count that justified the pick rather than a free-run the
#: addendum retired.
WAVE5_REVIEW = {
    "C13.T8.S4.L05": (12, "alive-parent-band"),
    "C07.T5.S4.L12": (12, "alive-parent-band"),
    "C05.T8.S4.L05": (12, "alive-parent-band"),
    "C11.T8.S4.L00": (11, "alive-parent-band"),
    "C06.T1.S4.L12": (10, "alive-parent-band"),
    "C11.T1.S8.L00": (10, "alive-parent-band"),
    "C12.T8.S4.L00": (9, "alive-degraded"),
    "C09.T5.S4.L02": (9, "alive-degraded"),
    "C08.T5.S4.L02": (8, "alive-degraded"),
    "C13.T5.S4.L05": (11, "alive-parent-band"),
    "C05.T3.S4.L05": (12, "alive-parent-band"),
    "C05.T4.S4.L05": (12, "alive-parent-band"),
    "C07.T1.S4.L12": (11, "alive-parent-band"),
    "C07.T2.S4.L12": (11, "alive-parent-band"),
    "C11.T1.S4.L00": (10, "alive-parent-band"),
    "C06.T2.S4.L12": (10, "alive-parent-band"),
    "C06.T5.S4.L12": (10, "alive-parent-band"),
    "C06.T8.S4.L12": (10, "alive-parent-band"),
}


def _sealed_isolation(cell: CellId) -> dict:
    """Read the sealed wave-5 receipt for this exact cell id, or say there is none."""

    for candidate in (
        CELLS / cell.text / "cell-receipt.json",
        CELLS / cell.text / "wave5" / "cell-receipt.json",
    ):
        if not candidate.is_file():
            continue
        body = json.loads(candidate.read_text(encoding="utf-8"))
        solve = body.get("solve") or {}
        review, band = WAVE5_REVIEW.get(cell.text, (None, None))
        return {
            "campaign_wave": int(body.get("campaign_wave", 5)),
            "receipt_path": str(candidate),
            "receipt_sha": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "receipt_sha_sealed": body.get("receipt_sha"),
            "stored_parameter_count": solve.get("stored_parameter_count"),
            "parent_parameter_count": solve.get("parent_parameter_count"),
            "storage_realized_ratio": solve.get("storage_realized_ratio"),
            "review_useful_or_partial": review,
            "review_verdict": band,
            "measured_at_this_site": True,
        }
    return {
        "campaign_wave": 6,
        "measured_at_this_site": False,
        "note": (
            "No sealed isolation cell exists at this component/technique/site; "
            "this is a wave-6 phase-2 expansion site and the composed review "
            "is the only reading it has."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="CELL_ID:WIDTH",
        help="A component to fold, with the storage width its S level resolves to.",
    )
    parser.add_argument("--description", default="")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    parsed: list[tuple[CellId, int]] = []
    for raw in arguments.cell:
        text, _, width = raw.partition(":")
        if not width:
            raise SystemExit(f"--cell {raw!r} needs an explicit :WIDTH")
        parsed.append((CellId.parse(text.strip()), int(width)))
    parsed.sort(key=lambda pair: install_order_key(pair[0]))

    components = []
    for cell, width in parsed:
        review, band = WAVE5_REVIEW.get(cell.text, (None, None))
        if review is None:
            reason = (
                f"wave-6 phase-2 expansion: {cell.component} "
                f"({COMPONENTS[cell.component]['name']}) under {cell.technique} "
                f"at {cell.level}, added at the UNMEASURED site {cell.site} on "
                "top of the strongest phase-1 child.  Its only grade is the "
                "composed frozen review that follows this install."
            )
        else:
            reason = (
                f"wave-6 greedy assembly: {cell.text} is the wave-5 (NOTES-MX-I "
                f"§5) survivor at {COMPONENTS[cell.component]['name']}, "
                f"sealed {review}/15 {band} at a storage-defined {cell.level}, "
                "site-scoped under ADDENDUM 5 rule 2-S."
            )
        components.append(
            {
                "cell_id": cell.text,
                "component_id": cell.component,
                "weight_keys": list(component_weight_keys(cell)),
                "evidence_cache": evidence_cache_name(cell),
                "evidence_endpoints": list(evidence_endpoints(cell)),
                "x4_width": int(width),
                "isolation": _sealed_isolation(cell),
                "selection_reason": reason,
            }
        )

    folded = {entry["component_id"] for entry in components}
    passed_through = []
    for component_id in sorted(COMPONENTS):
        if component_id in folded:
            continue
        passed_through.append(
            {
                "component_id": component_id,
                "component": COMPONENTS[component_id]["name"],
                "reason": (
                    "not selected by the wave-6 greedy path at this step; "
                    "installed as parent weights (copy_exact).  A component is "
                    "added only after the previous addition has been reviewed."
                ),
            }
        )

    recipe = {
        "schema": RECIPE_SCHEMA,
        "variant": arguments.variant,
        "ranking": "behaviour-first-greedy",
        "description": arguments.description
        or (
            "Wave 6 (MX-J) greedy assembly under ADDENDUM 6: the wave-5 S4 "
            "survivors installed one at a time in descending review order, with "
            "the full frozen 15-prompt review run after every addition."
        ),
        "selection_policy": {
            "admission": (
                "ADDENDUM 6 §6.1: a component is admitted on its sealed "
                "frozen-review verdict, not on free-run token match, which the "
                "addendum retires to a diagnostic column."
            ),
            "ranking": "descending wave-5 useful-or-partial count",
            "install_order": (
                "topological, input side -> output side; within a layer conv -> "
                "attention -> mlp -> norms (matrix_composition.install_order_key)"
            ),
            "site_scope": (
                "ADDENDUM 5 rule 2-S: phase-1 winners install at their sealed "
                "passing site only.  Phase-2 entries install at additional "
                "REGISTERED sites and say so in their own selection reason."
            ),
        },
        "install_order": [entry["cell_id"] for entry in components],
        "components": components,
        "passed_through": passed_through,
    }
    validate_recipe(recipe)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(recipe, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{arguments.variant}: {len(components)} folded "
        f"[{' -> '.join(entry['cell_id'] for entry in components)}] -> "
        f"{arguments.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
