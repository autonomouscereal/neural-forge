#!/usr/bin/env python3
"""Render the two derived tables the attribution report quotes.

`branch_decomposition.md` — the stage-delta decomposition of each layer's
composed post-layer endpoint (operator stages vs the MLP refit stage).
`capacity_risk.md` — the planner's own cached per-layer capacity evidence at
the selected layout, which is where the 2560 -> ~320 contraction is quantified.
"""

from __future__ import annotations

import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT = os.path.normpath(
    os.path.join(HERE, "..", "armA-review-20260808", "receipt.json")
)


def main() -> int:
    with open(
        os.path.join(HERE, "branch_decomposition.json"), "r", encoding="utf-8"
    ) as handle:
        branch = json.load(handle)
    with open(
        os.path.join(HERE, "budget_and_lattice.json"), "r", encoding="utf-8"
    ) as handle:
        budget = json.load(handle)
    with open(RECEIPT, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)

    lines = [
        "| L | operator | baseline (held out, nothing installed) | after operator "
        "installs, old MLP (fit) | after MLP refit (fit) | after everything (held "
        "out) | MLP-stage delta | whole-layer delta | operator-stage delta "
        "(residual) | fit-vs-heldout gap |",
        "|" + "---|" * 10,
    ]
    for entry in branch["layers"]:
        lines.append(
            "| {L} | {op} | {b:.4f} | {c:.4f} | {cf:.4f} | {ch:.4f} | {dm:+.4f} | "
            "{dl:+.4f} | {do:+.4f} | {g:+.4f} |".format(
                L=entry["layer_index"],
                op=entry["operator_kind"],
                b=entry["baseline_heldout"],
                c=entry["current_fit_operator_installed_old_mlp"],
                cf=entry["candidate_fit_all_installed"],
                ch=entry["candidate_heldout_all_installed"],
                dm=entry["mlp_stage_delta_fit"],
                dl=entry["whole_layer_delta_heldout"],
                do=entry["operator_stage_delta_residual"],
                g=entry["fit_minus_heldout_at_candidate"],
            )
        )
    totals = {
        "mlp": sum(entry["mlp_stage_delta_fit"] for entry in branch["layers"]),
        "layer": sum(entry["whole_layer_delta_heldout"] for entry in branch["layers"]),
        "operator": sum(
            entry["operator_stage_delta_residual"] for entry in branch["layers"]
        ),
    }
    lines.append(
        "| **sum** | | | | | | **{m:+.4f}** | **{l:+.4f}** | **{o:+.4f}** | |".format(
            m=totals["mlp"], l=totals["layer"], o=totals["operator"]
        )
    )
    with open(
        os.path.join(HERE, "branch_decomposition.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    bank = receipt["planned_target"]["heterogeneous_capacity_option_bank"]
    selected = [
        int(value)
        for _key, value in sorted(
            budget["per_layer_mlp_intermediate_width"].items(),
            key=lambda kv: int(kv[0]),
        )
    ]
    risk_lines = [
        "| L | type | selected MLP width | post-SwiGLU tail | collision fraction "
        "| tangent tail | predicted defect | operator delta-energy | MLP "
        "delta-energy |",
        "|" + "---|" * 9,
    ]
    for evidence in bank["layer_evidence"]:
        index = int(evidence["layer_index"])
        risks = {
            int(option["intermediate_size"]): option
            for option in evidence["mlp_capacity_risks"]
        }
        width = selected[index]
        option = risks.get(width) or risks[min(risks)]
        risk_lines.append(
            "| {L} | {t} | {w} | {p:.4f} | {c:.4f} | {g:.4f} | {d:.5f} | {oe:.4f} "
            "| {me:.4f} |".format(
                L=index,
                t=evidence["layer_type"],
                w=width,
                p=option["post_swiglu_tail_fraction"],
                c=option["collision_fraction"],
                g=option["tangent_tail_fraction"],
                d=option["predicted_defect"],
                oe=evidence["operator_delta_energy_fraction"],
                me=evidence["mlp_delta_energy_fraction"],
            )
        )
    with open(os.path.join(HERE, "capacity_risk.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(risk_lines) + "\n")
    print()
    print("\n".join(risk_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
