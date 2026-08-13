#!/usr/bin/env python3
"""The per-case attribution artifact the deck has carried unbuilt since wave 14.

It needs no GPU, no evidence and no build: every input is already sealed in the
repository's own human-review packages -- one ``per-case-<TAG>.json`` per graded
child, which is a reading of that child's frozen dossier case by case, and one
``build-receipt-<CHILD>.json`` beside it, which says which axis moved.

Three questions, none of which any single wave could answer:

1. **Is the review a ladder or a set?**  If the fifteen cases fell in a fixed
   order, a child holding k cases would hold the k easiest ones, and the whole
   campaign's scores would be one number wide.  The Guttman coefficient below is
   exactly that test: the fraction of (child, case) cells that a strict
   difficulty order would have predicted.
2. **Which cases does each AXIS take first?**  A case that dies under every axis
   is measuring depth; a case that dies under one is measuring that axis.  The
   per-axis hold rate, against the campaign-wide rate for the same case, is the
   attribution.
3. **What survives at the bottom?**  Which cases the deepest children still
   hold, which is the only place a fold's residual competence is visible.

Nothing here grades anything and nothing here is a gate.  It prices what has
already been graded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HELD = ("Useful", "Partial")


def axis_of(receipt: dict[str, Any]) -> str:
    """Which axis a child moved, from its own build receipt.

    Wave 14's receipts predate the ``axes`` array, so the fallback reads the
    same fields the array is built from.  A child that moved two axes is named
    by both, in a fixed order, because a composed child is its own class.
    """

    names: list[str] = []
    if isinstance(receipt.get("axes"), list):
        names = [str(a.get("axis")) for a in receipt["axes"] if a.get("axis")]
    else:
        if receipt.get("keep") is not None or receipt.get("keep_ff") is not None:
            names.append("intermediate_size")
        if receipt.get("excised"):
            names.append("num_hidden_layers")
        if receipt.get("keep_kv_groups") is not None:
            names.append("num_attention_heads")
        if receipt.get("keep_head_dim") is not None:
            names.append("head_dim")
        if receipt.get("keep_hidden_size") is not None:
            names.append("hidden_size")
    order = (
        "hidden_size",
        "intermediate_size",
        "head_dim",
        "num_attention_heads",
        "num_hidden_layers",
    )
    names = [n for n in order if n in names]
    return "+".join(names) if names else "unattributed"


def collect(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, str]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("wave*/human-review/per-case-*.json")):
        tag = path.name[len("per-case-") : -len(".json")]
        wave = path.parts[-3]
        cases[f"{wave}/{tag}"] = json.loads(path.read_text())
        receipts = sorted(path.parent.glob(f"build-receipt-*{tag}.json"))
        receipt = json.loads(receipts[0].read_text()) if receipts else {}
        meta[f"{wave}/{tag}"] = {
            "wave": wave,
            "tag": tag,
            "axis": axis_of(receipt),
            "net_ratio": receipt.get("net_ratio"),
        }
    return cases, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="the campaign report root")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cases, meta = collect(args.root)
    if not cases:
        raise SystemExit(f"no per-case artifacts under {args.root}")
    case_names: list[str] = []
    for reading in cases.values():
        for name in reading:
            if name not in case_names:
                case_names.append(name)

    held: dict[str, set[str]] = {
        child: {c for c, verdict in reading.items() if verdict in HELD}
        for child, reading in cases.items()
    }
    rate = {
        case: sum(1 for child in held if case in held[child]) / len(held)
        for case in case_names
    }
    difficulty = sorted(case_names, key=lambda c: -rate[c])

    # ---- the Guttman test: would a strict difficulty order have predicted this?
    agree = total = 0
    for child, kept in held.items():
        predicted = set(difficulty[: len(kept)])
        for case in case_names:
            total += 1
            agree += int((case in kept) == (case in predicted))
    guttman = agree / total

    by_axis: dict[str, list[str]] = defaultdict(list)
    for child, info in meta.items():
        by_axis[info["axis"]].append(child)
    axis_rate: dict[str, dict[str, float]] = {}
    for axis, children in sorted(by_axis.items()):
        axis_rate[axis] = {
            case: sum(1 for c in children if case in held[c]) / len(children)
            for case in case_names
        }

    deepest = sorted(
        (c for c in meta if meta[c]["net_ratio"]),
        key=lambda c: meta[c]["net_ratio"],
    )[:8]

    report = {
        "schema": "neural-forge-matrix-wave28-percase-attribution-v1",
        "children": len(held),
        "cases": len(case_names),
        "difficulty_order": [
            {"case": c, "hold_rate": round(rate[c], 4)} for c in difficulty
        ],
        "guttman_agreement": round(guttman, 4),
        "guttman_note": (
            "the fraction of (child, case) cells a strict difficulty order predicts; "
            "1.0 would mean the review is one number wide and 0.5 is chance on a "
            "balanced matrix"
        ),
        "per_axis_hold_rate": {
            axis: {
                "children": len(by_axis[axis]),
                "mean_held": round(
                    sum(len(held[c]) for c in by_axis[axis]) / len(by_axis[axis]), 3
                ),
                "cases": {c: round(v, 3) for c, v in axis_rate[axis].items()},
            }
            for axis in sorted(by_axis)
        },
        "deepest_children": [
            {
                "child": c,
                "axis": meta[c]["axis"],
                "net_ratio": meta[c]["net_ratio"],
                "held": sorted(held[c]),
            }
            for c in deepest
        ],
    }
    print(f"children {report['children']}  cases {report['cases']}")
    print(f"Guttman agreement {report['guttman_agreement']}")
    print("\ncase                            campaign  " + "  ".join(
        f"{a[:14]:>14s}" for a in sorted(by_axis)
    ))
    for case in difficulty:
        row = "  ".join(f"{axis_rate[a][case]:>14.2f}" for a in sorted(by_axis))
        print(f"{case:<30s} {rate[case]:>8.2f}  {row}")
    print("\naxis                            children  mean cases held")
    for axis in sorted(by_axis):
        block = report["per_axis_hold_rate"][axis]
        print(f"{axis:<30s} {block['children']:>8d}  {block['mean_held']:>15.2f}")
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
