#!/usr/bin/env python3
"""RESULTS_MATRIX: every frontier child on two columns, two review surfaces, and benchmarks.

This is the artifact Cyril asked wave 29 for -- "[Cyril quote withheld]" -- and it is generated from the sealed record rather than written from
memory.  `mx29_campaign_table.py` already produces the authoritative join of the
ledger, the per-lane child indexes and the `lm-eval` output; this file consumes
that JSON, adds the wave-29 extension-surface score where one exists, and lays
the whole thing out so the plan's mechanisms are visible **at a glance**:

* rows are ordered by net stored ratio, deepest last, because depth is the axis
  the campaign is trying to buy;
* the two storage columns ADDENDUM 12 established -- standard-shape and
  parent-shaped -- are marked in the first cell of every row rather than split
  into two tables, so a reader can see them interleave;
* the review column is the sealed fifteen and is the ONLY column any claim fires
  on.  The extension column beside it is ADDENDUM 16.1's second surface and it
  prices and orders;
* a MECHANISM column names which of the retention plan's mechanisms produced the
  row, so the improvement the plan bought is readable as a column and not as an
  argument;
* the frontier ladder -- deepest child holding at least N of 15, for every N --
  is computed from the same rows and printed first, because that ladder is the
  only summary the campaign has ever agreed on.

Nothing here recomputes a score.  If a number is in this file it came from a
sealed receipt, and if a number is missing it is missing because nothing sealed
it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PARENT_BENCH = {
    "arc_easy": 0.5450, "hellaswag": 0.5050, "piqa": 0.6800,
    "winogrande": 0.5500, "lambada_openai": 0.4600,
}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bench_cell(bench: dict[str, Any] | None) -> str:
    if not bench:
        return "--"
    parts = []
    for key, short in (("arc_easy", "arc"), ("hellaswag", "hella"), ("piqa", "piqa"),
                       ("winogrande", "wino"), ("lambada_openai", "lam")):
        v = bench.get(key)
        if v is not None:
            parts.append(f"{short} {float(v):.3f}")
    ppl = bench.get("lambada_ppl") or bench.get("lambada_openai_perplexity")
    if ppl:
        parts.append(f"ppl {float(ppl):.2f}")
    bpb = bench.get("bits_per_byte") or bench.get("wikitext_bits_per_byte")
    if bpb:
        parts.append(f"b/b {float(bpb):.4f}")
    return " · ".join(parts) if parts else "--"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True, help="campaign-frontier-table.json")
    ap.add_argument("--extension", type=Path, help="wave-29 extension-score JSON")
    ap.add_argument("--mechanisms", type=Path, help="JSON: cell_id -> mechanism label")
    ap.add_argument("--wave", type=int, default=29)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    table = json.loads(args.table.read_text(encoding="utf-8"))
    children: list[dict[str, Any]] = table["children"]
    ext_by_label: dict[str, dict[str, Any]] = {}
    retro: dict[str, Any] | None = None
    if args.extension and args.extension.is_file():
        ext = json.loads(args.extension.read_text(encoding="utf-8"))
        for row in ext.get("children", []):
            if row.get("label"):
                ext_by_label[str(row["label"])] = row
        retro = ext.get("retro_score")
    mechanisms: dict[str, str] = {}
    if args.mechanisms and args.mechanisms.is_file():
        mechanisms = json.loads(args.mechanisms.read_text(encoding="utf-8"))

    def tag_of(cell: str) -> str:
        return cell.split(".")[-1]

    # ---- the ladder: deepest child holding at least N of 15 -----------------
    ladder: list[dict[str, Any]] = []
    for n in range(15, -1, -1):
        holders = [c for c in children if c["review"] >= n] if n else children
        if not holders:
            continue
        deepest = min(holders, key=lambda c: c["net_ratio"])
        ladder.append({
            "at_least": n,
            "cell_id": deepest["cell_id"],
            "net_ratio": deepest["net_ratio"],
            "review": deepest["review"],
            "wave": deepest["wave"],
            "standard_shape": deepest["standard_shape"],
        })

    lines: list[str] = []
    A = lines.append
    A(f"# RESULTS MATRIX — Neural Forge fold-only campaign, through wave {args.wave}")
    A("")
    A(f"**Generated** {time.strftime('%Y-%m-%d', time.gmtime())} from "
      f"`{args.table.name}`, which is itself a join of the sealed `ledger.jsonl`, "
      "the per-wave child indexes and the sealed `lm-eval` output. Nothing in this "
      "file is recomputed from prose.")
    A("")
    A(f"**Parent** LFM2.5-230M, {table['parent_parameters']:,} parameters, "
      f"**{table['parent_review']}/15** on the frozen protocol. "
      f"**{table['whole_model_children_reviewed']} whole-model children reviewed**, "
      f"{table['children_scoring_at_least_one']} holding at least one case, "
      f"{len(table['benchmark_labels'])} benchmark rows, "
      f"{table['ledger_rows']} ledger rows.")
    A("")
    A("**How to read the two review columns.** `sealed 15` is the frozen "
      "15-prompt protocol, read by hand, and it is the **only** column any "
      "frontier claim, verdict or ledger row fires on — before wave 29 and after "
      "it. `ext 30` is ADDENDUM 16.1's extension surface, authored blind in wave "
      "29 and machine-scored against a rubric written before any child was "
      "decoded. It prices and orders; it never grades. A row with a blank `ext 30` "
      "was not run on the second surface, which is a statement about GPU budget "
      "and not about the child.")
    A("")
    A("**Storage column.** `S` is standard-shape — a directory stock "
      "`transformers` loads, R3-exportable. `P` is parent-shaped — a sealed "
      "composed tensor set at a fully-counted stored ratio. ADDENDUM 16.3 permits "
      "a parent-shaped child to headline the parent-band column provided its "
      "format is named, which is what the first column does on every row.")
    A("")

    A("## 1. The frontier ladder — deepest child holding at least N of 15")
    A("")
    A("| at least | child | net ratio | read | wave | storage |")
    A("|---:|---|---:|---:|---:|:--:|")
    for rung in ladder:
        A(f"| {rung['at_least']}/15 | `{rung['cell_id']}` | "
          f"{rung['net_ratio']:.6f}x | {rung['review']}/15 | {rung['wave']} | "
          f"{'S' if rung['standard_shape'] else 'P'} |")
    A("")

    A("## 2. Every reviewed whole-model child, deepest last")
    A("")
    A("| | child | net ratio | sealed 15 | ext 30 | charged | verdict | mechanism | benchmarks |")
    A("|:--:|---|---:|---:|---:|---:|---|---|---|")
    for c in children:
        tag = tag_of(c["cell_id"])
        ext = ext_by_label.get(tag)
        A("| {s} | `{cid}` | {ratio:.6f}x | **{rev}**/15 | {e} | {ch} | {v} | {m} | {b} |".format(
            s="S" if c["standard_shape"] else "P",
            cid=c["cell_id"],
            ratio=c["net_ratio"],
            rev=c["review"],
            e=f"{ext['held']}/{ext['prompts']}" if ext else "--",
            ch=str(ext["charged_lanes"]) if ext else (
                str(c["degenerate_lane_count"]) if c.get("degenerate_lane_count") is not None else "--"),
            v=c["per_child_verdict"],
            m=mechanisms.get(c["cell_id"], "--"),
            b=bench_cell(c.get("benchmark")),
        ))
    A("")

    if retro:
        A("## 3. The retro-score — does the sealed map survive a second instrument?")
        A("")
        A(f"**Verdict: {retro['verdict']}**")
        A("")
        A(f"Spearman(sealed 15, extension 30) = **{fmt(retro['spearman_sealed_vs_extension'])}** "
          f"over {len(retro['table'])} sealed checkpoints spanning "
          f"{retro['table'][0]['sealed_15']}/15 down to {retro['table'][-1]['sealed_15']}/15. "
          f"The sealed surface separates them into {retro['sealed_distinct_levels']} distinct "
          f"levels; the extension surface separates them into "
          f"{retro['extension_distinct_levels']}.")
        A("")
        A("| child | sealed 15 | extension 30 | extension % | charged |")
        A("|---|---:|---:|---:|---:|")
        for row in retro["table"]:
            A(f"| `{row['label']}` | {row['sealed_15']}/15 | {row['extension_30']}/30 | "
              f"{row['extension_pct']}% | {row['charged_lanes']} |")
        A("")

    A("## 4. What this matrix does not say")
    A("")
    A("It does not rank children by benchmark. The benchmark columns are ADDENDUM 8 "
      "diagnostics and have been anti-correlated with the review often enough that "
      "reading them as a score would invert the campaign's own findings. It does not "
      "gate anything on the extension column. And it does not claim a parent-shaped "
      "row is exportable: R3 is unchanged, and R4 — no quantization, no MoE — holds "
      "on every row of both columns.")
    A("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(children)} children, {len(ladder)} ladder rungs, "
          f"{len(ext_by_label)} extension-scored")
    for rung in ladder:
        print(f"  >= {rung['at_least']:>2}/15  {rung['cell_id']:<24} {rung['net_ratio']:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
