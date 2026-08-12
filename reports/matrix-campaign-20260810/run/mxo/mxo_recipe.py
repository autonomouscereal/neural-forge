#!/usr/bin/env python3
"""Build one wave-11 (MX-O) recipe from an explicit cell list.

Wave 8's `mxl_recipe.py`, plus the one component the campaign could not name
before: **C01, the tied token embedding**.  `--cell C01.T4.S4:252` folds the
65,536 x 1,024 table that is 29.2% of the parent, and `--knob` attaches the
per-component settings its kernel reads (which weighting, and the sealed
token-frequency file plus digest when the weighting is a weighted one).

C01 has no layer, so it occupies no block and the lane-exclusivity rule below
simply does not see it -- `_block_assignment` skips a component with no layer
index, and `install_order_key` sorts it last.  That is not a special case for
the embedding; it is what a GLOBAL-site component has always meant.

Wave 7's `mxk_recipe.py`, plus the one rule wave 7's mechanism section earned
(`NOTES-MX-K` §2.1): **a block hosts at most one folded lane.**

Wave 7 measured that `C05 conv_out_proj` writes `operator.output`, which the FFN
norm of the same block then reads, so an MLP fold installed in that block is
solved against a residual the conv fold has already moved.  The readings:
C05 alone at L05 is 10/15, C05 + `mlp_gate` at L05 is 7/15, C05 + `mlp_down` at
L05 is 6/15, and `mlp_gate` + `mlp_down` together at L05 -- both on the same side
of that norm -- is 10/15.  So the conflict is not "two folds in one block", it is
"two folds on opposite sides of one block's operator/FFN boundary".

That is why the exclusivity test below is on the LANE KIND and not on the
component: two mlp components in one block are one lane and are allowed, because
that is the pair wave 7 measured at no cost at all.  A conv writer and an mlp
fold in one block are two lanes and are refused.

The refusal is machine-enforced rather than left to the caller's discipline,
because a wave that is *testing* an exclusivity rule is exactly the wave most
likely to violate it by accident and then read the damage as evidence about a
site.  `--allow-block-collision REASON` is the deliberate override, and it
writes the reason into the recipe so a collision is always something the file
admits to rather than something a reader has to notice.

Usage::

    python mxo_recipe.py --variant MXL.CTRLA \
        --cell C05.T8.S8.L05:127 --cell C13.T8.S4.L04:639 \
        --cell C07.T5.S4.L12:121 --out recipe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-10a608d5")
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

#: The wave-5 (NOTES-MX-I §5) behavioural ladder, verbatim, carried forward from
#: `mxk_recipe.py` unchanged so a wave-8 recipe cites the same isolation reading
#: a wave-7 recipe cited for the same cell.
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

#: What waves 6 and 7 measured about a site *in composition*, which is a
#: different reading from an isolation cell and is carried separately so a
#: recipe never claims an isolation grade it does not have.  Every line is a
#: sealed row, cited to the note that sealed it.
COMPOSED_EVIDENCE = {
    "C13.T8.S4.L00": "wave-6 phase 2 on top of COMP.MA.MXJ2: 5/15, dead by "
    "degeneracy, install drift 0.422844 (NOTES-MX-J §5).",
    "C13.T8.S4.L13": "wave-6 phase 2 on top of COMP.MA.MXJ2: 7/15, impaired, "
    "install drift 0.380073 (NOTES-MX-J §5).",
    "C13.T8.S4.L04": "wave-7 rung on top of COMP.MA.MXJ2: 8/15, alive-degraded, "
    "install drift 0.258949, condition 3,528.6 (NOTES-MX-K §3.1).",
    "C13.T8.S4.L06": "wave-7 rung on top of COMP.MA.MXJ2: 7/15, dead by "
    "degeneracy, install drift 0.355062 (NOTES-MX-K §3.1).",
    "C13.T8.S4.L03": "wave-7 rung on top of COMP.MA.MXJ2: 6/15, dead by "
    "degeneracy, install drift 0.363202 (NOTES-MX-K §3.1).",
    "C11.T8.S4.L05": "wave-7: the wave's one real win -- 10/15 alive-parent-band "
    "in COMP.MA.MXK5, install drift 0.168344, the smallest single install the "
    "campaign has measured (NOTES-MX-K §3.3).",
    "C11.T8.S4.L07": "wave-7 rung on top of COMP.MA.MXK5: 7/15, impaired, "
    "install drift 0.312499 (NOTES-MX-K §3.3).",
    "C11.T8.S4.L13": "wave-7 rung on top of COMP.MA.MXK5: 7/15, dead by "
    "degeneracy (the word `backend` several hundred times to the cap), install "
    "drift 0.371356 (NOTES-MX-K §3.3).",
    "C07.T5.S4.L08": "wave-7: 8/15 alive-degraded as the second attn_k fold "
    "(COMP.MA.MXK4); 7/15 dead on top of COMP.MA.MXK5 (NOTES-MX-K §3.2).",
    "C07.T5.S4.L10": "wave-7: as a THIRD attn_k fold it collapses the decoder -- "
    "COMP.MA.MXK12, 4/15, five lanes are single-token loops (NOTES-MX-K §3.2).",
    "C05.T8.S8.L05": "wave-7: 13/15 alive-parent-band in COMP.MA.MXK14, one case "
    "ABOVE the same-process parent control, install drift 0.223583, condition "
    "447.8 -- better than its own S4 arm (NOTES-MX-K §4).",
    "C05.T8.S4.L05": "wave-7: 10/15 alive-parent-band in COMP.MA.MXK6, install "
    "drift 0.181179 (NOTES-MX-K §2).",
    "C07.T5.S8.L12": "wave-7: only ever measured inside the all-S8 child "
    "COMP.MA.MXK15, which is dead by degeneracy; it has no single-site S8 arm "
    "(NOTES-MX-K §8).",
    "C11.T8.S8.L05": "wave-7: 9/15 alive-degraded in COMP.MA.MXK13 (NOTES-MX-K §4).",
    "C13.T8.S8.L05": "wave-7: only ever measured inside the all-S8 child "
    "COMP.MA.MXK15, which is dead by degeneracy (NOTES-MX-K §8).",
}

#: Which residual-stream lane a component occupies inside its block.  This is
#: the exclusivity axis, and it is read from the component's own registered kind
#: rather than restated here, so it cannot drift away from the registry.
LANE_OF_KIND = {
    "short_conv": "operator (writes operator.output, upstream of the FFN norm)",
    "full_attention": "operator (writes operator.output, upstream of the FFN norm)",
    "mlp": "feed-forward (reads the FFN norm, downstream of the operator)",
    "norm": "norm",
    "token": "token",
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
            "composed_evidence": COMPOSED_EVIDENCE.get(cell.text),
        }
    note = (
        "No sealed isolation cell exists at this component/technique/site.  "
        "Under ADDENDUM 7 the site is REGISTERED, which is a statement about the "
        "receipt schema and not evidence about the layer (rule 2-S); the composed "
        "frozen review that follows this install is the only reading this rung has."
    )
    extra = COMPOSED_EVIDENCE.get(cell.text)
    if extra:
        note = f"{note}  Composed evidence: {extra}"
    return {
        "campaign_wave": 10,
        "measured_at_this_site": False,
        "note": note,
        "composed_evidence": extra,
    }


def _block_assignment(parsed: list[tuple[CellId, int]]) -> dict:
    """Map every folded component to its block and lane, and test exclusivity.

    Returns the assignment map that goes into the recipe.  Raises if two lanes
    land in one block and the caller did not say so on the command line.
    """

    by_block: dict[int, list[tuple[CellId, str, str]]] = defaultdict(list)
    for cell, _ in parsed:
        kind = str(COMPONENTS[cell.component]["kind"])
        layer = cell.layer_index
        if layer is None:
            continue
        by_block[int(layer)].append((cell, kind, LANE_OF_KIND.get(kind, kind)))

    collisions = []
    blocks = {}
    for layer in sorted(by_block):
        entries = by_block[layer]
        lanes = sorted({lane for _, _, lane in entries})
        blocks[f"L{layer:02d}"] = {
            "folded": [cell.text for cell, _, _ in entries],
            "kinds": sorted({kind for _, kind, _ in entries}),
            "lanes": lanes,
            "lane_count": len(lanes),
            "block_exclusive": len(lanes) == 1,
        }
        if len(lanes) > 1:
            collisions.append(
                f"L{layer:02d}: {[cell.text for cell, _, _ in entries]} occupy "
                f"{len(lanes)} lanes {lanes}"
            )
    return {"blocks": blocks, "collisions": collisions}


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
    parser.add_argument(
        "--allow-block-collision",
        default="",
        metavar="REASON",
        help=(
            "Deliberately fold two lanes into one block.  The reason is written "
            "into the recipe; without it a collision is a hard refusal."
        ),
    )
    parser.add_argument(
        "--knob",
        action="append",
        default=[],
        metavar="CELL_ID:KEY=VALUE",
        help=(
            "A per-component knob the runner merges over the wave knob file, "
            "e.g. C01.T4.S4:embedding_weighting=token-frequency.  Values that "
            "parse as JSON are stored as JSON; everything else stays a string."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    knobs_by_cell: dict[str, dict] = defaultdict(dict)
    for raw in arguments.knob:
        target, _, assignment = raw.partition(":")
        key, _, value = assignment.partition("=")
        if not target or not key:
            raise SystemExit(f"--knob {raw!r} is not CELL_ID:KEY=VALUE")
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        knobs_by_cell[target.strip()][key.strip()] = parsed_value

    parsed: list[tuple[CellId, int]] = []
    for raw in arguments.cell:
        text, _, width = raw.partition(":")
        if not width:
            raise SystemExit(f"--cell {raw!r} needs an explicit :WIDTH")
        parsed.append((CellId.parse(text.strip()), int(width)))
    parsed.sort(key=lambda pair: install_order_key(pair[0]))

    assignment = _block_assignment(parsed)
    if assignment["collisions"] and not arguments.allow_block_collision:
        raise SystemExit(
            "REFUSED by the wave-8 block-exclusivity rule (NOTES-MX-K §2.1):\n  "
            + "\n  ".join(assignment["collisions"])
            + "\nA conv or attention writer and an MLP fold in one block sit on "
            "opposite sides of that block's FFN norm, so the MLP is solved "
            "against a residual the operator fold has already moved.  Pass "
            "--allow-block-collision REASON to do it on purpose."
        )
    assignment["allowed_deliberately"] = bool(arguments.allow_block_collision)
    assignment["collision_reason"] = arguments.allow_block_collision or None

    components = []
    for cell, width in parsed:
        review, band = WAVE5_REVIEW.get(cell.text, (None, None))
        if review is None:
            reason = (
                f"wave-10 assembly: {cell.component} "
                f"({COMPONENTS[cell.component]['name']}) under {cell.technique} "
                f"at {cell.level}, added at site {cell.site} in the "
                f"{LANE_OF_KIND.get(str(COMPONENTS[cell.component]['kind']))} lane.  "
                "No wave measured this cell in isolation; its only grade is the "
                "composed frozen review that follows this install."
            )
        else:
            reason = (
                f"wave-10 assembly base: {cell.text} is the wave-5 (NOTES-MX-I "
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
                "knobs": dict(knobs_by_cell.get(cell.text, {})),
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
                    "not selected by the wave-10 assembly at this rung; installed "
                    "as parent weights (copy_exact).  A site is added only after "
                    "the previous addition has been reviewed."
                ),
            }
        )

    recipe = {
        "schema": RECIPE_SCHEMA,
        "variant": arguments.variant,
        "ranking": "behaviour-first-greedy-block-exclusive",
        "description": arguments.description
        or (
            "Wave 8 (MX-L) block-exclusive assembly under ADDENDUM 6 + ADDENDUM "
            "7: the wave-7 best composed child extended one SITE at a time under "
            "the rule that a block hosts at most one folded lane, with the full "
            "frozen 15-prompt review run after every addition."
        ),
        "selection_policy": {
            "admission": (
                "ADDENDUM 6 §6.1: a component is admitted on its sealed "
                "frozen-review verdict, not on free-run token match, which the "
                "addendum retires to a diagnostic column."
            ),
            "ranking": (
                "greedy, block-exclusive; a rung is kept only if the composed "
                "child stays alive-degraded or better (>= 8/15) and is not dead "
                "by the degeneracy clause"
            ),
            "block_exclusivity": (
                "NOTES-MX-K §2.1: a block hosts at most one folded LANE.  The "
                "operator lane (short_conv / full_attention) writes "
                "operator.output upstream of the block's FFN norm; the "
                "feed-forward lane reads that norm.  Folding both in one block "
                "solves the MLP against a residual the operator fold has already "
                "moved -- measured at -3 review cases for mlp_gate and -4 for "
                "mlp_down against the same block's conv writer, while the two "
                "MLP tensors together in one block cost nothing.  Two components "
                "in the SAME lane are therefore allowed; two lanes are refused."
            ),
            "install_order": (
                "topological, input side -> output side; within a layer conv -> "
                "attention -> mlp -> norms (matrix_composition.install_order_key)"
            ),
            "site_scope": (
                "ADDENDUM 5 rule 2-S under ADDENDUM 7's widened registry: a site "
                "that schema v2 registers has NO evidence attached to it by being "
                "registered.  Every rung is measured or it is nothing."
            ),
            "storage_floor": (
                "Cyril's hard rule: every folded component stores at or below a "
                "true quarter (S4) or eighth (S8) of its OWN parent tensor.  No "
                "sub-4x level is solved, sealed, or used as a fallback; a cell "
                "whose technique cannot express 4x at its site is dropped, never "
                "folded shallower."
            ),
        },
        "block_assignment": assignment,
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
    lanes = "; ".join(
        f"{block}={info['folded']}" for block, info in assignment["blocks"].items()
    )
    print(
        f"{arguments.variant}: {len(components)} folded "
        f"[{' -> '.join(entry['cell_id'] for entry in components)}] -> "
        f"{arguments.out}"
    )
    print(f"  blocks: {lanes}")
    print(
        f"  block-exclusive: "
        f"{all(info['block_exclusive'] for info in assignment['blocks'].values())}"
        + (
            f"  (deliberate collision: {arguments.allow_block_collision})"
            if arguments.allow_block_collision
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
