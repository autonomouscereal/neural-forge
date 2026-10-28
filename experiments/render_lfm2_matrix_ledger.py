#!/usr/bin/env python3
"""Render the authenticated LFM2 component-matrix ledger without scoring it."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.hf_lfm2_matrix_cell import read_matrix_ledger  # noqa: E402


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_ledger(records: Sequence[Mapping[str, object]]) -> str:
    """Return a human-auditable report; no numeric field decides intelligence."""

    latest: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in records:
        key = (str(row["instance_id"]), str(row["cell_id"]))
        latest[key] = row
    rows = sorted(
        latest.values(),
        key=lambda row: (
            str(row["component_ids"]),
            str(row["technique_id"]),
            str(row["site_id"]),
            str(row["level_id"]),
            int(row["sequence"]),
        ),
    )
    complete_child = sum(
        row["evidence_class"] == "complete_child" for row in rows
    )
    isolation = sum(row["evidence_class"] == "isolation_only" for row in rows)
    manual_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["manual_intelligence_status"])
        manual_counts[status] = manual_counts.get(status, 0) + 1

    lines = [
        "# LFM2 component × technique matrix",
        "",
        "Manual review of whole decoded responses is authoritative. Numeric triage "
        "only localizes the first measurable divergence; exact-token match does not "
        "accept or reject intelligence. Safety observations are separate, and a "
        "token-cap termination is censored rather than failed.",
        "",
        f"- Authenticated ledger records: {len(records)}",
        f"- Latest cell instances rendered: {len(rows)}",
        f"- Complete blank-child evidence rows: {complete_child}",
        f"- Isolation-only diagnostic rows: {isolation}",
        "- Manual states: "
        + (", ".join(f"{key}={value}" for key, value in sorted(manual_counts.items())) or "none"),
        "",
        "Only `complete_child` rows can support a compaction claim. An "
        "`isolation_only` row can identify a working component technique but is not "
        "a compact model.",
        "",
        "| Cell | Components | Technique | Level | Site | Evidence | Execution | Manual intelligence | Numeric first divergence | Free-run token match | Mean LCP | Terminal relL2 | Receipt |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        receipt = str(row["receipt_sha256"])
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(row["cell_id"]),
                    _cell(",".join(str(value) for value in row["component_ids"])),
                    _cell(row["technique_id"]),
                    _cell(row["level_id"]),
                    _cell(row["site_id"]),
                    _cell(row["evidence_class"]),
                    _cell(row["execution_status"]),
                    _cell(row["manual_intelligence_status"]),
                    _cell(row["numeric_first_divergence"]),
                    f"{float(row['free_run_token_match_vs_control']):.6f}",
                    f"{float(row['free_run_mean_lcp']):.6f}",
                    f"{float(row['terminal_heldout_rel_l2']):.6f}",
                    f"`{receipt[:12]}`",
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = render_ledger(read_matrix_ledger(args.ledger))
    if args.out is None:
        print(report)
    else:
        _atomic_write(args.out.expanduser().absolute(), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
