#!/usr/bin/env python3
"""Assemble the campaign-wide frontier table from the sealed record, not prose.

Every wave's note quotes its own children.  Nobody has ever built the table of
ALL of them, and the one time a wave tried to name the frontier from memory it
named the wrong child for three waves (`NOTES-MX-W` §9).  So this reads the
authoritative ledger, joins it to the per-wave child indexes for the ADDENDUM 9
per-lane reading and to the sealed `lm-eval` output for the benchmark columns,
and writes one artifact the consolidated report is generated from.

Sources, in order of authority:

* ``ledger.jsonl`` -- one canonical row per sealed receipt.  The behaviour
  verdict rows carry ``behavior_useful_or_partial``, ``behavior_verdict``,
  ``stored_parameter_ratio`` and the reviewer's own note.  Where a cell has more
  than one reviewed row (a re-seal, a re-run), the LATEST by ``sealed_at`` wins
  and the superseded ones are reported rather than dropped silently.
* ``wave*/wave*-child-index.json`` -- waves 15-21 only; this is where the
  per-lane ADDENDUM 9 reading and the export class live.  Earlier waves predate
  the artifact and their per-lane column is ``unstated``, which is what
  ADDENDUM 9 §4 says a review that names no lanes seals.
* ``wave*/bench/<label>/**/results*.json`` -- the sealed `lm-eval` output.  The
  parent row is re-run per wave and is the campaign's third calibration; this
  script checks every parent row against the sealed reference and refuses to
  write a table if one disagrees.

Whole-model children only: ``cell_kind`` in {architecture, composition}.  A
component cell's ``stored_parameter_ratio`` is the ratio for that COMPONENT and
putting it in a frontier table would claim a 0.124x model that does not exist.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any

PARENT_REFERENCE = {
    "arc_easy": 0.5450,
    "hellaswag": 0.5050,
    "piqa": 0.6800,
    "winogrande": 0.5500,
    "lambada": 0.4600,
    "lam_ppl": 24.39,
    "wiki": 81.80,
}
PARENT_PARAMETERS = 229_693_184
PARENT_REVIEW = 12


def bench_rows(root: Path) -> dict[str, dict[str, Any]]:
    """Every sealed benchmark row, keyed by label, newest wave wins."""

    out: dict[str, dict[str, Any]] = {}
    patterns = [
        str(root / "wave*" / "bench" / "*" / "**" / "results*.json"),
        str(root / "wave*" / "bench" / "*" / "results*.json"),
    ]
    seen: list[tuple[int, str, str]] = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            match = re.search(r"/wave(\d+)/bench/([^/]+)/", path.replace(os.sep, "/"))
            if match:
                seen.append((int(match.group(1)), match.group(2), path))
    for wave, label, path in sorted(seen):
        try:
            body = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # The parent is re-run every wave and every one of those rows is a
        # calibration, so they are kept apart.  A child label collides only if a
        # cell was re-benchmarked, and then the later wave is the one to keep.
        key = f"PARENT@w{wave}" if label == "PARENT" else label
        results = body.get("results", {})

        def pick(task: str, prefix: str) -> float | None:
            for metric, value in results.get(task, {}).items():
                if metric.startswith(prefix):
                    return value
            return None

        row = {
            "wave": wave,
            "arc_easy": pick("arc_easy", "acc,"),
            "hellaswag": pick("hellaswag", "acc_norm,"),
            "piqa": pick("piqa", "acc_norm,"),
            "winogrande": pick("winogrande", "acc,"),
            "lambada": pick("lambada_openai", "acc,"),
            "lam_ppl": pick("lambada_openai", "perplexity,"),
            "wiki": pick("wikitext", "word_perplexity,"),
        }
        if all(row[k] is None for k in PARENT_REFERENCE):
            continue
        out[key] = row
    return out


def parent_calibration(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every wave's own parent row against the sealed reference, to 4 decimals."""

    report: dict[str, Any] = {"reference": PARENT_REFERENCE, "waves": {}}
    for label, row in rows.items():
        if not label.startswith("PARENT"):
            continue
        report["waves"][str(row["wave"])] = row
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="the wave root on disk")
    ap.add_argument(
        "--repo",
        type=Path,
        action="append",
        required=True,
        help=(
            "a directory holding wave<N>/wave<N>-child-index.json.  Repeatable, "
            "because the current wave's index is written to the wave root on the "
            "share before it is committed, and a table generated during the wave "
            "that silently omitted the wave's own per-lane readings would be a "
            "table with a hole in exactly the rows the wave is about"
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ledger = [
        json.loads(line)
        for line in (args.root / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]

    latest: dict[str, dict] = {}
    superseded: list[dict] = []
    for row in ledger:
        if row.get("cell_kind") not in ("architecture", "composition"):
            continue
        if row.get("behavior_useful_or_partial") is None:
            continue
        cell = str(row["cell_id"])
        if cell in latest:
            older, newer = sorted(
                (latest[cell], row), key=lambda r: str(r["sealed_at"])
            )
            superseded.append(
                {
                    "cell_id": cell,
                    "dropped_receipt_sha": older["receipt_sha"],
                    "dropped_score": older["behavior_useful_or_partial"],
                    "kept_receipt_sha": newer["receipt_sha"],
                }
            )
            latest[cell] = newer
        else:
            latest[cell] = row

    index: dict[str, dict] = {}
    for repo in args.repo:
        for path in sorted(glob.glob(str(repo / "wave*" / "wave*-child-index.json"))):
            body = json.loads(Path(path).read_text())
            for child in body.get("children", []):
                index[str(child["cell_id"])] = child

    benches = bench_rows(args.root)

    children = []
    for cell, row in latest.items():
        score = int(row["behavior_useful_or_partial"])
        extra = index.get(cell, {})
        label = cell if cell in benches else None
        children.append(
            {
                "cell_id": cell,
                "wave": row["campaign_wave"],
                "net_ratio": row["stored_parameter_ratio"],
                "stored_parameters": row["stored_parameter_count"],
                "review": score,
                "parent_review": row["behavior_parent_useful_or_partial"],
                "per_child_verdict": row["behavior_verdict"],
                "per_lane_verdict": extra.get("per_lane_verdict", "unstated"),
                "degenerate_lane_count": extra.get("degenerate_lane_count"),
                "degeneracy_reading": extra.get("degeneracy_reading", "unstated"),
                "smoke": row.get("decode_smoke_review"),
                "standard_shape": bool(row.get("standard_shape")),
                "export_class": extra.get("export_class"),
                "trust_remote_code": extra.get("trust_remote_code"),
                "token_match_diagnostic": row.get("token_match_vs_control"),
                "drift_vs_control": row.get("vs_copy_exact_control_delta"),
                "write_heldout_cos": row.get("write_heldout_cos"),
                "receipt_sha": row["receipt_sha"],
                "cell_instance_id": row["cell_instance_id"],
                "notes": row.get("notes"),
                "benchmark": benches.get(label),
                "dossier": f"reports/matrix-campaign-20260810/wave{row['campaign_wave']}"
                f"/human-review/READ-{cell.split('.')[-1]}.txt"
                if str(row["campaign_wave"]).isdigit()
                and int(row["campaign_wave"]) >= 15
                else None,
            }
        )
    children.sort(key=lambda c: (c["net_ratio"], c["cell_id"]))

    calibration = parent_calibration(benches)
    disagreeing = []
    for wave, row in calibration["waves"].items():
        for key, want in PARENT_REFERENCE.items():
            got = row.get(key)
            if got is None:
                continue
            if abs(float(got) - want) > (0.01 if want > 1 else 0.0005):
                disagreeing.append({"wave": wave, "column": key, "got": got, "want": want})
    if disagreeing:
        raise SystemExit(
            "REFUSED: a wave's parent benchmark row disagrees with the sealed "
            f"reference, so no column in this table is comparable: {disagreeing}"
        )

    payload = {
        "schema": "neural-forge-matrix-wave43-campaign-frontier-v1",
        "parent_parameters": PARENT_PARAMETERS,
        "parent_review": PARENT_REVIEW,
        "ledger_rows": len(ledger),
        "whole_model_children_reviewed": len(latest),
        "children_scoring_at_least_one": sum(1 for c in children if c["review"] >= 1),
        "superseded_reviewed_rows": superseded,
        "parent_benchmark_calibration": calibration,
        "benchmark_labels": sorted(benches),
        "children": children,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(
        f"{len(children)} reviewed whole-model children, "
        f"{payload['children_scoring_at_least_one']} scoring >= 1/15, "
        f"{len(benches)} benchmark rows, "
        f"{len(calibration['waves'])} parent rows all matching the reference "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
