#!/usr/bin/env python3
"""Run one canonical LFM2 component-method cell from a sealed JSON request."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.dynamic_circuit_ir import canonical_json_bytes  # noqa: E402
from neural_forge.hf_lfm2_matrix_cell import (  # noqa: E402
    MatrixCellRequest,
    load_executor,
    run_matrix_cell,
    wave_one_cell_ids,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--executor")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--print-wave-one",
        action="store_true",
        help="Print the predeclared first-wave cell identifiers and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_wave_one:
        print(canonical_json_bytes({"cell_ids": list(wave_one_cell_ids())}).decode())
        return 0
    if args.request is None or args.executor is None or args.out is None:
        raise SystemExit("--request, --executor, and --out are required for a run")
    request = MatrixCellRequest.from_dict(
        json.loads(args.request.read_text(encoding="utf-8"))
    )
    receipt = run_matrix_cell(
        request,
        load_executor(args.executor),
        args.out,
        ledger_path=args.ledger,
    )
    print(
        canonical_json_bytes(
            {
                "status": receipt["status"],
                "cell_id": receipt["request"]["cell_id"],
                "evidence_class": receipt["request"]["evidence_class"],
                "manual_intelligence_status": receipt[
                    "manual_intelligence_review"
                ]["status"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
        ).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
