#!/usr/bin/env python3
"""Render the consolidated report's tables from the sealed artifact, not by hand.

`NOTES-MX-W` §9 is the reason this file exists: three waves quoted a frontier
child that the ledger does not say is the frontier, because the number was
carried forward in prose instead of recomputed.  Every table this emits is a
function of `campaign-frontier-table.json`, which is itself a function of the
ledger, the per-wave child indexes and the sealed `lm-eval` output.

Two tables:

``--table frontier``
    Every reviewed whole-model child scoring at least 1 of 15, sorted by net
    stored ratio, with both ADDENDUM 6 and ADDENDUM 9 readings, the export
    class, the benchmark row where one exists, and the dossier path.

``--table scorecard``
    The top fifteen by Cyril's own criterion -- any intelligence per
    compression -- which is the deepest children that still answer something.
    Ranked by net ratio ascending among children scoring >= 1/15, because a
    child that answers nothing has no intelligence to price and a child that
    answers at 0.99x has bought nothing with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float | None, spec: str = "%.4f") -> str:
    return "--" if value is None else spec % value


def bench_cells(child: dict) -> list[str]:
    bench = child.get("benchmark")
    if not bench:
        return ["--"] * 7
    return [
        fmt(bench["arc_easy"]),
        fmt(bench["hellaswag"]),
        fmt(bench["piqa"]),
        fmt(bench["winogrande"]),
        fmt(bench["lambada"]),
        fmt(bench["lam_ppl"], "%.2f"),
        fmt(bench["wiki"], "%.2f"),
    ]


def lane_cell(child: dict) -> str:
    verdict = child.get("per_lane_verdict") or "unstated"
    count = child.get("degenerate_lane_count")
    if verdict == "unstated":
        return "`unstated`"
    return f"{verdict} ({count}/16)" if count is not None else verdict


def frontier(body: dict) -> str:
    rows = [c for c in body["children"] if c["review"] >= 1]
    rows.sort(key=lambda c: (c["net_ratio"], c["cell_id"]))
    out = [
        "| # | child | wave | net stored | stored params | review | per-child | "
        "per-lane (reading) | shape | arc_e | hella_n | piqa_n | wino | lambada | "
        "lam ppl | wiki wppl |",
        "|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, child in enumerate(rows, 1):
        shape = "standard" if child["standard_shape"] else "parent-shaped"
        out.append(
            "| %d | `%s` | %s | **%.6f** | %s | **%d/15** | %s | %s | %s | %s |"
            % (
                index,
                child["cell_id"],
                child["wave"],
                child["net_ratio"],
                f"{child['stored_parameters']:,}",
                child["review"],
                child["per_child_verdict"],
                lane_cell(child),
                shape,
                " | ".join(bench_cells(child)),
            )
        )
    return "\n".join(out)


def scorecard(body: dict, characters: dict[str, str], limit: int = 15) -> str:
    rows = [c for c in body["children"] if c["review"] >= 1]
    rows.sort(key=lambda c: (c["net_ratio"], -c["review"]))
    out = [
        "| rank | child | net | review | shape | the character of its failure | READ- dossier |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    for index, child in enumerate(rows[:limit], 1):
        dossier = child.get("dossier") or "not packaged (pre-wave-15)"
        out.append(
            "| %d | `%s` | **%.6f** | **%d/15** | %s | %s | `%s` |"
            % (
                index,
                child["cell_id"],
                child["net_ratio"],
                child["review"],
                "standard" if child["standard_shape"] else "parent-shaped",
                characters.get(child["cell_id"], "**UNWRITTEN**"),
                dossier,
            )
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--table", choices=["frontier", "scorecard"], required=True)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument(
        "--characters",
        type=Path,
        help=(
            "`--table scorecard`: a JSON map from cell id to the one-line "
            "character of that child's failure.  It is PROSE and it is kept in "
            "its own committed file rather than generated, because no artifact "
            "in this campaign knows what a failure reads like -- but the file is "
            "keyed by cell id so a child with no line written prints "
            "**UNWRITTEN** instead of silently printing nothing"
        ),
    )
    args = ap.parse_args()
    body = json.loads(args.artifact.read_text(encoding="utf-8"))
    if args.table == "frontier":
        print(frontier(body))
    else:
        characters = (
            json.loads(args.characters.read_text(encoding="utf-8"))
            if args.characters
            else {}
        )
        print(scorecard(body, characters, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
