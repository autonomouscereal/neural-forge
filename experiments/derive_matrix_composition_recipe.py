#!/usr/bin/env python3
"""Derive the composition wave's recipes from the SEALED receipts.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` ADDENDUM 1 names the selection rule
and ADDENDUM 3 names the admission band, and both are statements about numbers
that live in sealed receipts.  The winners table in `NOTES-MX-G` §5 is a *render*
of those numbers; a recipe built from the prose would inherit every transcription
error in it and none of the receipts' authority.  So this script does the
derivation mechanically:

1. read the append-only ledger and take the newest instance per cell id
   (§3: a re-run gets a new instance, never an overwrite -- so "the state of a
   cell" is its newest row, and the review instance is newer than the run);
2. open the sealed receipt each of those rows names, **by digest**, and verify
   the seal before reading a number out of it;
3. flatten each into a candidate, grade it (ADDENDUM 3) and admit the promising
   and strong ones;
4. rank per component under each of the two pre-registered rankings and write
   one recipe per ranking, with the reason for every pick and every pass-through
   carried into the file.

Storage is read from the receipt rather than the ledger row on purpose: waves 1
and 2 sealed their storage counts under `solve` before `ledger_line` had columns
for them, so a ledger-only derivation loses the storage honesty of exactly the
cells the composition wave most wants it for.

Usage::

    python experiments/derive_matrix_composition_recipe.py \
        --ledger reports/matrix-campaign-20260810/ledger.jsonl \
        --cells  reports/matrix-campaign-20260810/cells \
        --out-directory reports/matrix-campaign-20260810/run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from neural_forge.dynamic_circuit_ir import canonical_json_bytes  # noqa: E402
from neural_forge.matrix_cell_receipt import (  # noqa: E402
    SCHEMA,
    CellId,
    read_ledger,
    verify_seal,
)
from neural_forge.matrix_composition import (  # noqa: E402
    RANKINGS,
    build_recipe,
    candidate_from_receipt,
    eligible,
    validate_recipe,
)

#: Which ranking produces which recipe variant, and what the variant is for.
VARIANTS = {
    "RBEH": (
        "behaviour-first",
        "ADDENDUM 1's rule -- highest free-run at the largest passing level, "
        "ties on composed-terminal margin -- with the storage-honesty clause "
        "applied only where an honest variant is within 0.05 free-run.",
    ),
    "RSTO": (
        "storage-honest",
        "Only folds whose sealed stored-parameter count is below the parent "
        "tensor they replace.  A composed child that stores more than its "
        "parent is a failure regardless of behaviour, so this variant measures "
        "what the campaign can build under that constraint alone.",
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def index_receipts(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every sealed cell receipt under ``root``, keyed by its own digest."""

    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(body, Mapping) or body.get("schema") != SCHEMA:
            continue
        verify_seal(body)
        index[str(body["receipt_sha"])] = (path, dict(body))
    return index


def candidates(ledger: Path, cells: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_ledger(ledger)
    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: str(r.get("sealed_at", ""))):
        latest[str(row["cell_id"])] = dict(row)
    index = index_receipts(cells)
    found: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for cell_id, row in sorted(latest.items()):
        cell = CellId.parse(cell_id)
        if cell.kind != "cell":
            continue
        entry = index.get(str(row["receipt_sha"]))
        if entry is None:
            unresolved.append(cell_id)
            continue
        path, receipt = entry
        candidate = candidate_from_receipt(receipt)
        candidate["receipt_path"] = str(path.relative_to(cells).as_posix())
        found.append(candidate)
    audit = {
        "ledger_rows": len(rows),
        "cells_latest": len(latest),
        "single_component_cells": len(found),
        "receipts_indexed": len(index),
        "unresolved_cell_ids": unresolved,
        "ledger_sha256": _file_sha256(ledger),
    }
    if unresolved:
        # A ledger row whose receipt is not on disk cannot be a winner: the
        # numbers would come from the row, which is a projection, and the
        # storage counts are exactly what the projection drops.
        raise SystemExit(
            f"{len(unresolved)} ledger rows name a receipt digest that is not "
            f"under {cells}: {unresolved[:5]}"
        )
    return found, audit


def _table(entries: list[dict[str, Any]]) -> str:
    lines = [
        f"{'component':5} {'cell':22} {'free-run':>9} {'stored':>9} "
        f"{'grade':10} {'status':6} {'elig':5}",
    ]
    for entry in sorted(
        entries,
        key=lambda c: (c["component_id"], -(c["free_run"] or 0.0)),
    ):
        ratio = entry["stored_ratio"]
        lines.append(
            f"{entry['component_id']:5} {entry['cell_id']:22} "
            f"{(entry['free_run'] or 0.0):9.4f} "
            f"{('--' if ratio is None else f'{ratio:.4f}'):>9} "
            f"{entry['grade']:10} {entry['status']:6} "
            f"{'yes' if eligible(entry) else 'no':5}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    found, audit = candidates(args.ledger, args.cells)
    print(json.dumps(audit, indent=1))
    print()
    print(_table(found))
    print()
    out = Path(args.out_directory)
    out.mkdir(parents=True, exist_ok=True)
    for variant, (ranking, description) in VARIANTS.items():
        if ranking not in RANKINGS:  # pragma: no cover - guarded by VARIANTS
            raise SystemExit(f"unknown ranking {ranking}")
        recipe = build_recipe(
            found,
            variant=variant,
            ranking=ranking,
            ledger_sha256=audit["ledger_sha256"],
            description=description,
        )
        recipe["derivation_audit"] = audit
        validate_recipe(recipe)
        path = out / f"composition-recipe-{variant}.json"
        path.write_bytes(canonical_json_bytes(recipe))
        print(f"=== {variant} ({ranking}) -> {path}")
        for entry in recipe["components"]:
            ratio = entry["isolation"]["stored_ratio"]
            shown = "--" if ratio is None else f"{float(ratio):.4f}"
            print(
                f"  fold {entry['component_id']:5} {entry['cell_id']:22} "
                f"free-run {(entry['isolation']['free_run'] or 0.0):.4f} "
                f"stored {shown}"
            )
        for entry in recipe["passed_through"]:
            print(f"  pass {entry['component_id']:5} {entry['reason'][:96]}")
        print()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out-directory", type=Path, required=True)
    return parser


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
