#!/usr/bin/env python3
"""Build one wave-7 (MX-K) full-depth-ladder recipe from an explicit cell list.

Wave 6's `mxj_recipe.py`, one schema version on.  The selection logic is
unchanged -- the components are named explicitly and carry a behavioural
selection reason, because ADDENDUM 3's free-run admission floor would refuse
every S4 survivor and ADDENDUM 6 retired it -- and the one thing that is new is
what the sites are allowed to be.

Under ADDENDUM 7 a cell id may name **any architecturally valid layer**, so a
rung of this wave's ladder is almost always a site with no sealed isolation
cell behind it.  That is not a gap to be papered over: `_sealed_isolation`
reads the sealed wave-5 receipt by digest wherever one exists and otherwise
says, in the recipe itself, that the composed frozen review is the only reading
the rung has.  ADDENDUM 5 rule 2-S is why that sentence has to be in the file
rather than in a note: a level claim is site-scoped, and a site being
*registered* is not evidence about it.

Usage::

    python mxk_recipe.py --variant MXK.L04 \
        --cell C13.T8.S4.L04:639 --cell C13.T8.S4.L05:639 \
        --cell C07.T5.S4.L12:121 --out recipe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16")
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

#: The wave-5 (NOTES-MX-I §5) behavioural ladder, verbatim, so a wave-7 recipe
#: carries the review count that justified the pick rather than a free-run the
#: addendum retired.  Every entry here is at a v1-registered site by
#: construction: wave 5 could not have measured any other kind.
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

#: What wave 6 measured about a site *in composition*, which is a different
#: reading from an isolation cell and is carried separately so a recipe never
#: claims an isolation grade it does not have.
WAVE6_COMPOSED = {
    "C13.T8.S4.L00": "wave-6 phase 2 measured this site on top of COMP.MA.MXJ2: "
    "5/15, dead by degeneracy, install drift 0.422844 (NOTES-MX-J §5).",
    "C13.T8.S4.L13": "wave-6 phase 2 measured this site on top of COMP.MA.MXJ2: "
    "7/15, impaired, install drift 0.380073 (NOTES-MX-J §5).",
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
            "receipt_schema": body.get("schema"),
            "stored_parameter_count": solve.get("stored_parameter_count"),
            "parent_parameter_count": solve.get("parent_parameter_count"),
            "storage_realized_ratio": solve.get("storage_realized_ratio"),
            "review_useful_or_partial": review,
            "review_verdict": band,
            "measured_at_this_site": True,
        }
    note = (
        "No sealed isolation cell exists at this component/technique/site.  "
        "Under ADDENDUM 7 the site is newly REGISTERED, which is a statement "
        "about the receipt schema and not evidence about the layer (rule 2-S); "
        "the composed frozen review that follows this install is the only "
        "reading this rung has."
    )
    extra = WAVE6_COMPOSED.get(cell.text)
    if extra:
        note = f"{note}  {extra}"
    return {
        "campaign_wave": 7,
        "measured_at_this_site": False,
        "note": note,
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
                f"wave-7 full-depth ladder (ADDENDUM 7): {cell.component} "
                f"({COMPONENTS[cell.component]['name']}) under {cell.technique} "
                f"at {cell.level}, added at the site {cell.site}, which schema "
                "v2 registers and no wave measured in isolation.  Its only "
                "grade is the composed frozen review that follows this install."
            )
        else:
            reason = (
                f"wave-7 ladder base: {cell.text} is the wave-5 (NOTES-MX-I "
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
                    "not selected by the wave-7 ladder at this rung; installed "
                    "as parent weights (copy_exact).  A site is added only "
                    "after the previous addition has been reviewed."
                ),
            }
        )

    recipe = {
        "schema": RECIPE_SCHEMA,
        "variant": arguments.variant,
        "ranking": "behaviour-first-greedy-per-layer",
        "description": arguments.description
        or (
            "Wave 7 (MX-K) full-depth ladder under ADDENDUM 6 + ADDENDUM 7: the "
            "wave-6 best composed child COMP.MA.MXJ2 extended one SITE at a "
            "time across every architecturally valid layer, with the full "
            "frozen 15-prompt review run after every addition."
        ),
        "selection_policy": {
            "admission": (
                "ADDENDUM 6 §6.1: a component is admitted on its sealed "
                "frozen-review verdict, not on free-run token match, which the "
                "addendum retires to a diagnostic column."
            ),
            "ranking": (
                "greedy per-layer, nearest-to-the-proven-site first; a rung is "
                "kept only if the composed child stays alive-degraded or better"
            ),
            "install_order": (
                "topological, input side -> output side; within a layer conv -> "
                "attention -> mlp -> norms (matrix_composition.install_order_key)"
            ),
            "site_scope": (
                "ADDENDUM 5 rule 2-S under ADDENDUM 7's widened registry: a "
                "site that schema v2 registers has NO evidence attached to it "
                "by being registered.  Every rung is measured or it is nothing."
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
