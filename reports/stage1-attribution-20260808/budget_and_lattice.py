#!/usr/bin/env python3
"""Cost the MLP-width remedy against the sealed exact-4x parameter budget.

Reads the arm-A run's own child graph (`remote/child_graph.json`) and its own
heterogeneous capacity option bank (from `receipt.json`), then answers one
question with the run's own numbers: how much MLP intermediate width can the
exact-4x budget actually buy, and what does the planner's cached tail evidence
say that width is worth.

Nothing here is a new experiment; every input is a sealed field of the run.
"""

from __future__ import annotations

import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "remote", "child_graph.json")
RECEIPT = os.path.normpath(
    os.path.join(HERE, "..", "armA-review-20260808", "receipt.json")
)
HIDDEN = 816
# w1 (I x H), w3 (I x H), w2 (H x I) -> three H-sized rows per intermediate unit.
PER_INTERMEDIATE_UNIT = 3 * HIDDEN


def main() -> int:
    with open(GRAPH, "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    with open(RECEIPT, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)

    by_kind: dict[str, int] = {}
    mlp_by_layer: dict[int, int] = {}
    conv_by_layer: dict[int, int] = {}
    attention_by_layer: dict[int, int] = {}
    for node in graph["nodes"]:
        count = int(node["logical_parameter_count"])
        by_kind[node["kind"]] = by_kind.get(node["kind"], 0) + count
        index = node["layer_index"]
        if index is None:
            continue
        if node["kind"] == "gated_mlp":
            mlp_by_layer[int(index)] = count
        elif node["kind"] == "short_conv":
            conv_by_layer[int(index)] = count
        elif node["kind"] == "attention":
            attention_by_layer[int(index)] = count

    total = sum(by_kind.values())
    operator = by_kind.get("short_conv", 0) + by_kind.get("attention", 0)
    mlp = by_kind.get("gated_mlp", 0)
    interface = by_kind.get("token_embedding", 0) + by_kind.get("output_head", 0)

    print(f"exact unique parameters      {total:,}")
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<18} {count:>12,}  {100.0 * count / total:5.2f}%")
    print()
    print(
        f"operator stack (short_conv+attention) {operator:,} "
        f"({100.0 * operator / total:.2f}%)"
    )
    print(f"MLP stack                             {mlp:,} ({100.0 * mlp / total:.2f}%)")
    print(
        f"token interface (embed+head)          {interface:,} "
        f"({100.0 * interface / total:.2f}%)"
    )
    print()

    # Per-layer intermediate width, recovered from the MLP parameter count.
    widths = {
        index: count // PER_INTERMEDIATE_UNIT for index, count in mlp_by_layer.items()
    }
    print("per-layer MLP intermediate width:", [widths[i] for i in sorted(widths)])
    width_sum = sum(widths.values())
    print(f"sum of intermediate widths across 14 layers: {width_sum}")
    print(f"cost of +1 intermediate unit on one layer:   {PER_INTERMEDIATE_UNIT}")
    print()

    # Short-conv internal widths, recovered from the graph counts.
    #   dense writer:   (3H + H + 3)*w                      = 3267*w
    #   rank-r writer:  (3H + 3)*w + H*r + r*w              = (2451+r)*w + H*r
    for index in sorted(conv_by_layer):
        count = conv_by_layer[index]
        dense = count / 3267.0
        note = f"dense w={dense:.0f}" if abs(dense - round(dense)) < 1e-6 else ""
        if not note:
            rank = 40
            width = (count - HIDDEN * rank) / (2451.0 + rank)
            note = f"rank-{rank} factor writer, w={width:.0f}"
        print(f"  L{index:<2} short_conv {count:>10,}  {note}")
    print()

    # The only lattice move that frees parameters is the narrow short-conv rung.
    narrow = 3267 * 704
    freed = 0
    for index, count in conv_by_layer.items():
        if count > narrow:
            freed += count - narrow
    print(
        f"parameters freed by moving every short_conv to the lattice's narrow "
        f"704 rung: {freed:,}"
    )
    bought = freed // (PER_INTERMEDIATE_UNIT * len(mlp_by_layer))
    print(
        f"  -> uniform MLP intermediate width increase across 14 layers: "
        f"+{bought} (to ~{min(widths.values()) + bought})"
    )
    print()

    # What would the 1280 rung cost, and what is the absolute arithmetic ceiling?
    for target in (483, 578, 1280, 1920, 2559):
        cost = (target * len(mlp_by_layer) - width_sum) * PER_INTERMEDIATE_UNIT
        print(
            f"uniform MLP width {target:>4}: +{cost:>12,} parameters "
            f"({100.0 * cost / total:6.1f}% of the exact budget)"
        )
    body_available = total - interface - by_kind.get("norm", 0)
    ceiling = body_available // (PER_INTERMEDIATE_UNIT * len(mlp_by_layer))
    print()
    print(
        "absolute arithmetic ceiling — spend every non-interface parameter on the "
        f"MLP and give the operator stack nothing: intermediate width {ceiling}"
    )

    # The planner's own cached tail evidence for those widths, at layer 0.
    bank = receipt["planned_target"]["heterogeneous_capacity_option_bank"]
    evidence = bank["layer_evidence"][0]
    print()
    print("layer-0 cached MLP capacity risk ladder (planner evidence, 2560 source):")
    print(
        "| intermediate | collision | post-SwiGLU tail | tangent tail | "
        "downstream-weighted tail | predicted defect |"
    )
    print("|" + "---|" * 6)
    for option in evidence["mlp_capacity_risks"]:
        print(
            "| {i} | {c:.4f} | {p:.4f} | {t:.4f} | {d:.4f} | {q:.5f} |".format(
                i=option["intermediate_size"],
                c=option["collision_fraction"],
                p=option["post_swiglu_tail_fraction"],
                t=option["tangent_tail_fraction"],
                d=option["downstream_weighted_tail_fraction"],
                q=option["predicted_defect"],
            )
        )
    print()
    print(
        "layer-0 short_conv writer risk:",
        json.dumps(evidence["short_conv_writer_risks"], sort_keys=True),
    )
    print(
        "layer-0 delta-energy fractions: operator "
        f"{evidence['operator_delta_energy_fraction']:.4f}  mlp "
        f"{evidence['mlp_delta_energy_fraction']:.4f}"
    )

    out = {
        "schema": "neural-forge-stage-a-budget-lattice-v1",
        "exact_unique_parameters": total,
        "by_kind": by_kind,
        "operator_stack": operator,
        "mlp_stack": mlp,
        "token_interface": interface,
        "per_layer_mlp_intermediate_width": {
            str(k): v for k, v in sorted(widths.items())
        },
        "cost_per_intermediate_unit_per_layer": PER_INTERMEDIATE_UNIT,
        "freed_by_narrow_short_conv_rung": freed,
        "uniform_width_bought_by_that_move": bought,
        "uniform_width_costs": {
            str(target): (target * len(mlp_by_layer) - width_sum)
            * PER_INTERMEDIATE_UNIT
            for target in (483, 578, 1280, 1920, 2559)
        },
        "absolute_arithmetic_ceiling_width": ceiling,
        "layer0_mlp_capacity_risks": evidence["mlp_capacity_risks"],
        "layer0_short_conv_writer_risks": evidence["short_conv_writer_risks"],
        "layer0_operator_delta_energy_fraction": evidence[
            "operator_delta_energy_fraction"
        ],
        "layer0_mlp_delta_energy_fraction": evidence["mlp_delta_energy_fraction"],
    }
    with open(
        os.path.join(HERE, "budget_and_lattice.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(out, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
