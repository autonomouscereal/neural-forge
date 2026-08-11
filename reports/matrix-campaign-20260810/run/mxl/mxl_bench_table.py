#!/usr/bin/env python3
"""The wave-8 benchmark frontier as one markdown table, read from lm-eval output.

Rows are ordered by net stored ratio, deepest first, because that is ADDENDUM
8's ranking; the review band rides along as an annotation so the compression /
retained-intelligence trade is visible on one line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix")
BENCH = ROOT / "wave8" / "bench"

COLUMNS = [
    ("arc_easy", "acc,none"),
    ("hellaswag", "acc_norm,none"),
    ("piqa", "acc_norm,none"),
    ("winogrande", "acc,none"),
    ("lambada_openai", "acc,none"),
    ("lambada_openai", "perplexity,none"),
    ("wikitext", "word_perplexity,none"),
]
HEADERS = [
    "arc_easy acc", "hellaswag acc_n", "piqa acc_n", "wino acc",
    "lambada acc", "lambada ppl", "wikitext wppl",
]


def results(label: str):
    root = BENCH / label
    if not root.is_dir():
        return None
    files = sorted(root.rglob("results_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8")).get("results") or {}


def receipt(spec: str):
    wave, _, cell = spec.partition(":")
    path = ROOT / wave / "cells" / cell / "cell-receipt-behavior-reviewed.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for spec in sys.argv[1:]:
        wave, _, cell = spec.partition(":")
        body = receipt(spec)
        data = results(cell)
        if body is None or data is None:
            print(f"!! missing {spec} (receipt={body is not None} bench={data is not None})",
                  file=sys.stderr)
            continue
        rows.append((
            cell,
            len(body["composition"]["components"]),
            float(body["composition"]["storage"]["net_ratio"]),
            int(body["addendum6"]["useful_or_partial"]),
            str(body["addendum6"]["verdict"]),
            data,
        ))
    rows.sort(key=lambda r: r[2])

    parent = results("PARENT")
    print("| model | n | net stored | review | verdict | " + " | ".join(HEADERS) + " |")
    print("|---|---:|---:|---:|---|" + "---:|" * len(HEADERS))
    cells = []
    for task, key in COLUMNS:
        value = parent.get(task, {}).get(key)
        cells.append(f"{value:.4f}" if value is not None else "n/a")
    print("| **parent (unfolded)** | 0 | 1.000000 | 12/15 | — | " + " | ".join(cells) + " |")
    for cell, folded, ratio, review, verdict, data in rows:
        cells = []
        for task, key in COLUMNS:
            value = data.get(task, {}).get(key)
            cells.append(f"{value:.4f}" if value is not None else "n/a")
        print(
            f"| `{cell}` | {folded} | **{ratio:.6f}×** | {review}/15 | {verdict} | "
            + " | ".join(cells) + " |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
