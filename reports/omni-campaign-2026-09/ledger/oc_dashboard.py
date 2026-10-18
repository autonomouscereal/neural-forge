#!/usr/bin/env python3
"""Regenerate the campaign's results dashboard and review queue FROM the ledger.

`STANDING_STATE`, `SCORECARD.md` and `RESULTS_MATRIX` in the LFM2 campaign are
regenerated from the ledger and never typed, because a hand-maintained table is
a second source of truth that drifts the first time someone is in a hurry.  This
does the same for the omni campaign: it reads the sealed registry (and, when
given one, the ledger, whose digest it prints so the table can be tied to a
ledger state), and writes markdown.

Two artifacts:

* ``--out`` (default `SCORECARD.md`): every checkpoint by stage, with both
  review columns, the canonical count, the status and the package path, sorted
  by intelligence-per-compression within each stage -- the LFM2 scorecard's own
  ordering, which puts "held something at a deep ratio" above "held a lot at no
  compression".
* ``--queue`` (default `REVIEW_QUEUE.md`): one row per checkpoint whose status
  is `partially_capable`, i.e. the ones R5 sends to Cyril, plus the bad ones
  waiting on retention and the clones that are never deleted.

Neither file is an input to anything.  Nothing here grades; every number is
copied from a sealed record.

Usage::

    oc_dashboard.py --registry R [--ledger L] [--out SCORECARD.md] [--queue REVIEW_QUEUE.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from oc_checkpoint import OcCheckpointError, all_current
from oc_ledger import read_rows
from oc_seal import CAMPAIGN, OcSealError, file_sha256, human_bytes, now_utc, oc_root

STAGE_ORDER = ("parent", "fold", "distill", "finetune", "clone")


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}" if value >= 10000 else str(value)
    return str(value)


def _intelligence_per_compression(record: Mapping[str, Any]) -> float:
    """Canonical signal divided by the ratio: the scorecard's sort key.

    A child that holds 3 lanes at 0.20x is ordered above one that holds 6 at
    0.90x, which is the whole point of the campaign.  An unreviewed or
    unratioed row sorts last rather than being invented a score.
    """

    review = record.get("review") or {}
    bank1 = review.get("bank1") or {}
    signal = bank1.get("canonical_signal")
    ratio = record["checkpoint"].get("net_ratio")
    if signal is None or not ratio:
        return -1.0
    return float(signal) / float(ratio)


def rows_for(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for checkpoint_id, record in records.items():
        checkpoint = record["checkpoint"]
        review = record.get("review") or {}
        bank1 = review.get("bank1") or {}
        columns = review.get("columns") or {}
        bank2 = review.get("bank2") or {}
        media = review.get("media_out") or []
        out.append(
            {
                "checkpoint_id": checkpoint_id,
                "tag": checkpoint.get("tag"),
                "stage": checkpoint["stage"],
                "status": record["status"],
                "status_reason": record["status_reason"],
                "parent_code": checkpoint.get("parent_code"),
                "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
                "wave": checkpoint.get("wave"),
                "parameters": checkpoint.get("parameter_count"),
                "net_ratio": checkpoint.get("net_ratio"),
                "ratio_measured": (
                    checkpoint.get("parameter_count_source")
                    == "derived-from-safetensors-headers"
                    and checkpoint.get("parent_parameter_count_source")
                    == "derived-from-safetensors-headers"
                ),
                "sealed": bank1.get("useful_or_partial"),
                "parent_sealed": bank1.get("parent_useful_or_partial"),
                "verdict": bank1.get("sealed_verdict"),
                "canonical": bank1.get("canonical_signal"),
                "column_a": columns.get("column_a"),
                "column_b": columns.get("column_b"),
                "bank2_sealed": bank2.get("sealed_useful_or_partial"),
                "bank2_canonical": bank2.get("canonical_signal"),
                "media": len(media),
                "media_scored": sum(1 for m in media if m.get("cyril_score") is not None),
                "package": review.get("package_path"),
                "directory": checkpoint.get("directory"),
                "bytes": checkpoint.get("directory_bytes"),
                "deleted": bool(checkpoint.get("deleted")),
                "frontier_holder": bool(checkpoint.get("frontier_holder")),
                "records": record["record_seq"],
                "sort_key": _intelligence_per_compression(record),
            }
        )
    return out


def scorecard_markdown(
    rows: Sequence[Mapping[str, Any]], *, ledger: Path | None, ledger_rows: int
) -> str:
    lines = [
        f"# Chimera Omni campaign scorecard — `{CAMPAIGN}`",
        "",
        f"Regenerated {now_utc()} from the sealed registry. **Nothing in this "
        "file is typed by hand; every number is copied from a sealed record.**",
        "",
    ]
    if ledger and Path(ledger).is_file():
        lines += [
            f"Ledger `{ledger}` — {ledger_rows} rows, sha256 "
            f"`{file_sha256(ledger)}`.",
            "",
        ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines += [
        "| status | checkpoints |",
        "|---|---:|",
        *[f"| {status} | {count} |" for status, count in sorted(counts.items())],
        f"| **total** | **{len(rows)}** |",
        "",
        "Every threshold below is a proposed decision line for Cyril. The only "
        "grade is measured intelligence read by a human.",
        "",
    ]
    for stage in STAGE_ORDER:
        stage_rows = [r for r in rows if r["stage"] == stage]
        if not stage_rows:
            continue
        stage_rows.sort(key=lambda r: (-r["sort_key"], r["checkpoint_id"]))
        lines += [
            f"## Stage: {stage}  ({len(stage_rows)})",
            "",
            "| checkpoint | tag | parent | ratio | ratio from | params "
            "| sealed/15 | parent/15 | canon | A | B | bank2 | media | status "
            "| package |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
        ]
        for row in stage_rows:
            package = f"`{row['package']}`" if row["package"] else "—"
            bank2 = (
                f"{_cell(row['bank2_sealed'])}/{_cell(row['bank2_canonical'])}"
                if row["bank2_sealed"] is not None or row["bank2_canonical"] is not None
                else "—"
            )
            lines.append(
                f"| `{row['checkpoint_id']}`{' 🗑' if row['deleted'] else ''}"
                f"{' ★' if row['frontier_holder'] else ''} "
                f"| {_cell(row['tag'])} | {_cell(row['parent_code'])} "
                f"| {_cell(row['net_ratio'])} "
                f"| {'headers' if row['ratio_measured'] else '**asserted**'} "
                f"| {_cell(row['parameters'])} "
                f"| {_cell(row['sealed'])} | {_cell(row['parent_sealed'])} "
                f"| {_cell(row['canonical'])} | {_cell(row['column_a'])} "
                f"| {_cell(row['column_b'])} | {bank2} "
                f"| {row['media_scored']}/{row['media']} | {row['status']} "
                f"| {package} |"
            )
        lines.append("")
    lines += [
        "Legend: **ratio** = child parameters / parent parameters (storage counted "
        "in PARAMETERS); **ratio from** = `headers` when BOTH counts were read "
        "from the safetensors headers on disk, **asserted** when one of them was "
        "typed by the operator and therefore proves nothing. **sealed/15** = the sealed per-case useful-or-partial "
        "count on the frozen 15-case bank. **canon** = canonical three-box signal "
        "lanes (artifact misses count for nothing; only content misses count "
        "against). **A** = copy-backs count; **B** = copy-backs refused, the "
        "machine-verified intelligence floor. ★ = frontier/floor holder, never "
        "dropped. 🗑 = bytes deleted, evidence retained. Status **unmeasured** = "
        "zero signal but zero content misses too — every lane was an artifact, so "
        "nothing was measured and nothing may be deleted; re-run it.",
        "",
    ]
    return "\n".join(lines)


def queue_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    partially = [r for r in rows if r["status"] == "partially_capable"]
    bad = [r for r in rows if r["status"] == "bad" and not r["deleted"]]
    unmeasured = [r for r in rows if r["status"] == "unmeasured"]
    clones = [r for r in rows if r["status"] == "cloned"]
    unscored = [
        r for r in rows if r["media"] and r["media_scored"] < r["media"]
    ]
    partially.sort(key=lambda r: (-r["sort_key"], r["checkpoint_id"]))
    lines = [
        "# REVIEW QUEUE — Chimera Omni",
        "",
        f"Regenerated {now_utc()} from the sealed registry.",
        "",
        "R5: a child is *at least partially capable* if it holds at least one "
        "canonical signal lane on ANY surface — the 15-case pack, bank v2, or a "
        "media-out artifact with non-zero content. Those are packaged and queued "
        "here; they are never deleted.",
        "",
        f"## For Cyril ({len(partially)})",
        "",
        "| checkpoint | stage | ratio | canon | A | B | media scored | package |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in partially:
        lines.append(
            f"| `{row['checkpoint_id']}` | {row['stage']} | "
            f"{_cell(row['net_ratio'])} | {_cell(row['canonical'])} | "
            f"{_cell(row['column_a'])} | {_cell(row['column_b'])} | "
            f"{row['media_scored']}/{row['media']} | "
            f"{('`' + row['package'] + '`') if row['package'] else '**NOT PACKAGED**'} |"
        )
    lines += [
        "",
        f"## Media artifacts awaiting a blind 1-5 ({len(unscored)})",
        "",
        "| checkpoint | artifacts | scored |",
        "|---|---:|---:|",
        *[
            f"| `{r['checkpoint_id']}` | {r['media']} | {r['media_scored']} |"
            for r in unscored
        ],
        "",
        f"## Unmeasured — re-run, never delete ({len(unmeasured)})",
        "",
        "Zero canonical signal AND zero content misses: every lane was an "
        "artifact (loop, cap-ride, empty generation, missing EOS). An artifact "
        "measures nothing either way, so these have not been graded — they were "
        "not run properly. They are not deletable by any path; fix the run "
        "(cap, EOS, decode) and review them again.",
        "",
        "| checkpoint | ratio | artifacts | bytes | why |",
        "|---|---:|---:|---:|---|",
        *[
            f"| `{r['checkpoint_id']}` | {_cell(r['net_ratio'])} | "
            f"{r['media']} | {human_bytes(int(r['bytes'] or 0))} | "
            f"{r['status_reason']} |"
            for r in unmeasured
        ],
        "",
        f"## Bad, awaiting retention ({len(bad)})",
        "",
        "A DROP needs every R4 condition: canonical 0 on every surface AND at "
        "least one measured content miss, no unscored media, not a frontier "
        "holder, rebuildable with the exact command from a retained source, not a "
        "clone or parent, nothing descending from it, and the evidence already "
        "packaged under a manifest that names THIS checkpoint and the digest the "
        "review committed to.",
        "",
        "| checkpoint | ratio | bytes | why bad |",
        "|---|---:|---:|---|",
        *[
            f"| `{r['checkpoint_id']}` | {_cell(r['net_ratio'])} | "
            f"{human_bytes(int(r['bytes'] or 0))} | {r['status_reason']} |"
            for r in bad
        ],
        "",
        f"## Clones — never deleted while a descendant exists ({len(clones)})",
        "",
        "| clone | of | bytes |",
        "|---|---|---:|",
        *[
            f"| `{r['checkpoint_id']}` | `{r['parent_checkpoint_id']}` | "
            f"{human_bytes(int(r['bytes'] or 0))} |"
            for r in clones
        ],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--queue", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the rows as JSON for another tool to read")
    args = ap.parse_args(argv)
    root = oc_root(args.root)
    registry = args.registry or root / "registry"
    ledger = args.ledger or (root / "ledger.jsonl")
    out = args.out or (root / "SCORECARD.md")
    queue = args.queue or (root / "REVIEW_QUEUE.md")

    try:
        rows = rows_for(all_current(registry))
        ledger_rows = len(read_rows(ledger)) if Path(ledger).is_file() else 0
    except (OcCheckpointError, OcSealError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        scorecard_markdown(rows, ledger=ledger, ledger_rows=ledger_rows),
        encoding="utf-8",
    )
    Path(queue).write_text(queue_markdown(rows), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    # Re-read both, so a truncated write is this command's failure and not the
    # next reader's confusion.
    for path in (out, queue):
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip().startswith("#"):
            print(f"REFUSED: {path} does not read back as markdown", file=sys.stderr)
            return 4
    print(f"checkpoints {len(rows)}  ledger rows {ledger_rows}")
    print(f"scorecard   {out}")
    print(f"queue       {queue}")
    for status in sorted({r["status"] for r in rows}):
        print(f"  {status:<18} {sum(1 for r in rows if r['status'] == status)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
