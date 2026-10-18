#!/usr/bin/env python3
"""s8_compare.py -- side-by-side bank-v1 table for parent / S3 / S5 / OMNI1 (S8-TEXT lane, 2026-10-13).

Reads each model's dossier (run_omni_frozen_review schema) and its oc_assess assessment (run WITHOUT --per-case,
so columns A/B are UPPER BOUNDS) and writes one JSON + prints a Markdown table:
  * per case: termination, generated tokens, form box (EMPTY / LOOP / CAP_RIDE / ECHO / CONTENT ...) per model;
  * columns A / B (of the scored lanes) per model, and the form-box histogram;
  * per case, whether two named models decoded the SAME token ids (e.g. OMNI1 vs S5: the composite's charted blocks
    are the identity on text and its band rows are masked, so on the same trunk placement the streams can be equal).
Grades nothing.  Usage:
  s8_compare.py --model NAME DOSSIER ASSESSMENT [--model ...] [--same A B] --out OUT.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "DOSSIER", "ASSESSMENT"), required=True)
    ap.add_argument("--same", nargs=2, action="append", metavar=("A", "B"), default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    models: dict[str, dict] = {}
    order: list[str] = []
    for name, dossier, assessment in args.model:
        d = json.loads(Path(dossier).read_text())
        a = json.loads(Path(assessment).read_text())
        recs = {r["id"]: r for r in d["child_responses"]}
        lanes = {l["lane"]: l for l in a["lanes"]}
        c = a["columns"]
        models[name] = {
            "dossier": dossier, "dossier_file_sha256": sha(dossier), "assessment": assessment, "assessment_file_sha256": sha(assessment),
            "label": d.get("label"), "dtype": d["generation"].get("dtype"), "split": d["generation"].get("split"),
            "compute": d["generation"].get("compute_architecture"), "wall_time_seconds": d.get("wall_time_seconds"),
            "column_a": c["column_a_copy_back_counts"], "column_b": c["column_b_copy_back_refused"],
            "scored": c["denominator_scored_lanes"], "is_upper_bound": c.get("is_upper_bound"),
            "voided_lanes": c.get("voided_lanes"),
            "form_boxes": dict(collections.Counter(l["form_box"] for l in a["lanes"])),
            "terminations": dict(collections.Counter(r.get("termination") for r in d["child_responses"])),
            "cases": {cid: {"termination": r.get("termination"), "generated_tokens": r.get("generated_tokens"),
                            "loop_detected": r.get("loop_detected"), "form_box": lanes.get(cid, {}).get("form_box"),
                            "ids_sha256": r.get("generated_token_ids_sha256")} for cid, r in recs.items()},
            "case_order": [r["id"] for r in d["child_responses"]],
        }
        order.append(name)

    same = {}
    for a_name, b_name in args.same:
        a_cases, b_cases = models[a_name]["cases"], models[b_name]["cases"]
        per = {cid: a_cases[cid]["ids_sha256"] == b_cases.get(cid, {}).get("ids_sha256") for cid in a_cases}
        same[f"{a_name}=={b_name}"] = {"identical_token_ids": sum(per.values()), "cases": len(per), "per_case": per}

    out = {"schema": "s8-text-compare-v1", "models": models, "order": order, "same_token_ids": same,
           "note": "columns A/B are oc_assess upper bounds (no --per-case reading); grades nothing"}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    cases = models[order[0]]["case_order"]
    print("| case | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    for cid in cases:
        cells = []
        for name in order:
            c = models[name]["cases"].get(cid, {})
            cells.append(f"{c.get('termination')} {c.get('generated_tokens')} {c.get('form_box')}")
        print(f"| {cid} | " + " | ".join(cells) + " |")
    print()
    print("| model | A | B | scored | form boxes | terminations | dtype | split | wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in order:
        m = models[name]
        print(f"| {name} | {m['column_a']} | {m['column_b']} | {m['scored']} | {m['form_boxes']} | {m['terminations']} | {m['dtype']} | {m['split']} | {m['wall_time_seconds']} |")
    for key, v in same.items():
        print(f"same token ids {key}: {v['identical_token_ids']}/{v['cases']}")
    print("out", args.out, sha(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
