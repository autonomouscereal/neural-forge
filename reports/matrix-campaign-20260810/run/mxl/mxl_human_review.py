#!/usr/bin/env python3
"""Build the wave-8 human-review package: Cyril scores the results himself.

One markdown file per coherent frontier child, plus one for the parent, each
carrying the recipe, the FULL verbatim responses for all fifteen graded prompts
(and the safety prompt) paired with the parent's, the benchmark table, and the
machine verdict as an ANNOTATION rather than a conclusion.

Two rules this file exists to obey:

* **No truncation and no summarizing of responses.**  Every response is written
  out byte-for-byte as the sealed dossier holds it, inside a fenced block, with
  its own termination and token count.  The whole point is that the person
  reading it is not being shown a filtered view.
* **The machine verdict is an annotation.**  ADDENDUM 8 makes coherence the gate
  and accuracy the tie-break, but the decision about whether phase 1's goal is
  hit is Cyril's, so the verdict, the band and the counts are reported next to
  the text that produced them and never in place of it.

Reads only sealed artifacts: `cell-receipt-behavior-reviewed.json`,
`frozen-review-dossier.json`, and the lm-eval `results_*.json` written by
`run-bench.sh`.  Computes nothing about behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix")
BENCH = ROOT / "wave8" / "bench"

#: The metric of record for each benchmark task, in the order the table prints.
METRICS = [
    ("arc_easy", "acc,none", "acc"),
    ("arc_easy", "acc_norm,none", "acc_norm"),
    ("hellaswag", "acc,none", "acc"),
    ("hellaswag", "acc_norm,none", "acc_norm"),
    ("piqa", "acc,none", "acc"),
    ("piqa", "acc_norm,none", "acc_norm"),
    ("winogrande", "acc,none", "acc"),
    ("lambada_openai", "acc,none", "acc"),
    ("lambada_openai", "perplexity,none", "perplexity (lower better)"),
    ("wikitext", "word_perplexity,none", "word_perplexity (lower better)"),
    ("wikitext", "bits_per_byte,none", "bits_per_byte (lower better)"),
]


def load_bench(label: str) -> dict | None:
    """The most recent lm-eval results file for one label, or None."""

    root = BENCH / label
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob("results_*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def bench_table(child_label: str | None) -> str:
    parent = load_bench("PARENT")
    child = load_bench(child_label) if child_label else None
    if parent is None:
        return (
            "_No parent benchmark baseline was produced, so no benchmark table "
            "can be honest here._\n"
        )
    lines = []
    if child is None and child_label:
        lines.append(
            f"_No benchmark run resolved for `{child_label}`; the parent column "
            "is shown alone rather than paired with a number that does not exist._\n"
        )
    header = "| task | metric | parent |" + (" child | delta |" if child else "")
    rule = "|---|---|---:|" + ("---:|---:|" if child else "")
    lines += [header, rule]
    for task, key, pretty in METRICS:
        p = (parent.get("results") or {}).get(task, {}).get(key)
        if p is None:
            continue
        row = f"| {task} | {pretty} | {p:.4f} |"
        if child:
            c = (child.get("results") or {}).get(task, {}).get(key)
            if c is None:
                row += " n/a | n/a |"
            else:
                row += f" {c:.4f} | {c - p:+.4f} |"
        lines.append(row)
    config = (child or parent).get("config") or {}
    lines.append("")
    lines.append(
        f"Sampled evaluation — `limit={config.get('limit')}`, "
        f"`seed={config.get('random_seed')}`, `batch_size={config.get('batch_size')}`, "
        "`max_length=2048`, dtype `float32`, one pinned sm86 RTX 3090. "
        "These are PARTIAL evaluations by construction (200 documents per task): "
        "they rank models against each other under identical settings and they are "
        "not comparable to published full-split numbers."
    )
    versions = (child or parent).get("versions") or {}
    if versions:
        lines.append("")
        lines.append(
            "Task versions: "
            + ", ".join(f"`{k}`={v}" for k, v in sorted(versions.items()))
        )
    return "\n".join(lines) + "\n"


def recipe_table(receipt: dict) -> str:
    composition = receipt["composition"]
    storage = composition["storage"]
    lines = [
        "| # | cell | component | level | site | width | stored | owned | stored/owned |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for index, entry in enumerate(composition["components"], start=1):
        s = entry["storage"]
        stored = int(s["stored_parameter_count"])
        owned = int(s["owned_parameter_count"])
        lines.append(
            f"| {index} | `{entry['isolation_cell_id']}` | {entry['component']} | "
            f"{entry['level']} | {entry['site']} | {entry['level_width']} | "
            f"{stored:,} | {owned:,} | **{stored / owned:.6f}** |"
        )
    removed = int(storage["parent_parameter_count"]) - int(
        storage["child_parameter_count"]
    )
    lines.append("")
    lines.append(
        f"**Net whole-model stored ratio {float(storage['net_ratio']):.6f}×** — "
        f"{int(storage['child_parameter_count']):,} unique parameters against the "
        f"parent's {int(storage['parent_parameter_count']):,} "
        f"(**−{removed:,}**).  Every folded component above stores at or below a "
        "true quarter (S4) or eighth (S8) of its own parent tensor; the net "
        "whole-model number is larger than any per-component number because the "
        "67,108,864-parameter tied embedding table (29.2% of the parent) is "
        "untouched by every component in the matrix."
    )
    return "\n".join(lines) + "\n"


def responses_section(dossier: dict, parent_dossier: dict) -> str:
    parent_by_id = {row["id"]: row for row in parent_dossier["parent_responses"]}
    out = []
    for index, row in enumerate(dossier["child_responses"], start=1):
        parent_row = parent_by_id.get(row["id"])
        out.append(f"### {index}. `{row['id']}` — {row['category']}")
        out.append("")
        out.append(
            f"**Child** — {row['termination']}, {row['generated_tokens']} tokens"
        )
        out.append("")
        out.append("```text")
        out.append(row["response_verbatim"])
        out.append("```")
        out.append("")
        if parent_row is not None:
            out.append(
                f"**Parent** — {parent_row['termination']}, "
                f"{parent_row['generated_tokens']} tokens"
            )
            out.append("")
            out.append("```text")
            out.append(parent_row["response_verbatim"])
            out.append("```")
            out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out)


def build_child(cell: str, wave: str, out_root: Path) -> None:
    root = ROOT / wave / "cells" / cell
    receipt = json.loads(
        (root / "cell-receipt-behavior-reviewed.json").read_text(encoding="utf-8")
    )
    dossier = json.loads(
        (root / "frozen-review-dossier.json").read_text(encoding="utf-8")
    )
    block = receipt["addendum6"]
    composition = receipt["composition"]

    body = []
    body.append(f"# {cell} — human review")
    body.append("")
    body.append(
        f"**Net stored ratio {float(composition['storage']['net_ratio']):.6f}× · "
        f"{len(composition['components'])} folded components · "
        f"machine verdict `{block['verdict']}` · "
        f"{block['useful_or_partial']}/15 useful-or-partial against the parent's "
        f"{block['parent_useful_or_partial']}/15**"
    )
    body.append("")
    body.append(
        "> The verdict and the count below are ANNOTATIONS.  ADDENDUM 8 makes "
        "retained coherence the success criterion and accuracy the tie-break, and "
        "the decision about whether this is a good enough fold-init is Cyril's, "
        "not the machine's.  Every response is reproduced in full and unedited."
    )
    body.append("")

    body.append("## 1. What was folded")
    body.append("")
    body.append(recipe_table(receipt))
    body.append("")
    body.append(f"Install order: {' → '.join(composition['install_order'])}")
    body.append("")
    body.append(
        f"Global install drift (terminal relL2 vs the copy_exact control): "
        f"{composition['global_drift']['terminal_relL2_vs_control']:.6f}"
    )
    body.append("")

    body.append("## 2. Benchmarks against the parent")
    body.append("")
    body.append(bench_table(cell))
    body.append("")

    body.append("## 3. The frozen review, in full")
    body.append("")
    body.append(
        f"Protocol `{dossier['protocol']['protocol_sha256'][:16]}…`, prompt pack "
        f"`{dossier['protocol']['prompt_pack_sha256'][:16]}…`, greedy decoding at "
        "the 1,024-token cap, parent decoded on the same prompts in the same "
        "process.  Fifteen prompts are graded; `safety-household-chemical` is "
        "reported and never scored."
    )
    body.append("")
    per_case = block.get("per_case") or {}
    if per_case:
        body.append("| case | machine grade |")
        body.append("|---|---|")
        for key in sorted(per_case):
            body.append(f"| `{key}` | {per_case[key]} |")
        body.append("")
    if block.get("notes"):
        body.append("**Reviewer notes sealed with this verdict:**")
        body.append("")
        body.append("> " + str(block["notes"]).replace("\n", "\n> "))
        body.append("")
    body.append(responses_section(dossier, dossier))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{cell}.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {out_root / f'{cell}.md'}")


def build_parent(reference_cell: str, wave: str, out_root: Path) -> None:
    dossier = json.loads(
        (ROOT / wave / "cells" / reference_cell / "frozen-review-dossier.json").read_text(
            encoding="utf-8"
        )
    )
    body = []
    body.append("# PARENT (LFM2.5-230M, unfolded) — human review baseline")
    body.append("")
    body.append(
        "The parent decoded on the same sixteen prompts, in the same process as "
        "every child, under the same frozen protocol.  Its sixteen responses are "
        "**byte-identical across every wave-7 and wave-8 dossier** (lane digest "
        "`865f006e3813712e…`), which is what makes the child comparisons "
        "comparisons rather than two separate experiments."
    )
    body.append("")
    body.append(
        "Machine calibration, reproduced before any wave-8 child was graded: "
        "**5 useful / 7 partial / 3 observed miss = 12/15**, with the misses at "
        "`arithmetic-multi-step` (it computes 94 kits and then answers "
        "\"94 kits left over\"), `logic-ordering` (it never states an order) and "
        "`multi-turn-correction` (it offers peanut butter to a severe peanut "
        "allergy).  The parent is not a strong model and the children are not "
        "measured against a strong model."
    )
    body.append("")
    body.append("## Benchmarks (parent baseline)")
    body.append("")
    body.append(bench_table(None))
    body.append("")
    body.append("## The parent's sixteen responses, in full")
    body.append("")
    for index, row in enumerate(dossier["parent_responses"], start=1):
        body.append(f"### {index}. `{row['id']}` — {row['category']}")
        body.append("")
        body.append(f"_{row['termination']}, {row['generated_tokens']} tokens_")
        body.append("")
        body.append("```text")
        body.append(row["response_verbatim"])
        body.append("```")
        body.append("")
        body.append("---")
        body.append("")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "PARENT.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {out_root / 'PARENT.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-from", default="wave8:COMP.MA.MXL5")
    parser.add_argument("--child", action="append", default=[], metavar="WAVE:CELL")
    arguments = parser.parse_args()
    wave, _, cell = arguments.parent_from.partition(":")
    build_parent(cell, wave, arguments.out)
    for spec in arguments.child:
        child_wave, _, child_cell = spec.partition(":")
        build_child(child_cell, child_wave, arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
