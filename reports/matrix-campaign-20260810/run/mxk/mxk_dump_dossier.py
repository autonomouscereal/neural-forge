#!/usr/bin/env python3
"""Print one frozen-review dossier as reviewable text, child and parent paired.

The dossier is canonical JSON with sixteen verbatim responses in each lane; the
protocol says to read the *whole* response and to grade the child absolutely
before viewing the parent, so this prints the child's sixteen first, then the
parent's, and never interleaves them case by case.  It computes nothing and
grades nothing: `seal_matrix_behavior_review.py` seals the verdict and
`behavior_verdict_for` does the band arithmetic.

Usage::

    python mxi_dump_dossier.py --dossier <path> [--lane child|parent|both]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _lane(records, label: str) -> None:
    print(f"########## {label} ({len(records)} cases) ##########")
    for index, row in enumerate(records, start=1):
        print()
        print(f"===== [{label} {index}/{len(records)}] {row['id']} "
              f"({row['category']}) :: {row['termination']}, "
              f"{row['generated_tokens']} tokens =====")
        print(row["response_verbatim"])
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, required=True)
    parser.add_argument("--lane", choices=("child", "parent", "both"), default="both")
    arguments = parser.parse_args()
    body = json.loads(arguments.dossier.read_text(encoding="utf-8"))
    print(f"cell        {body['cell_id']}  ({body.get('cell_kind')})")
    print(f"component   {body.get('component')} / {body.get('technique')} "
          f"/ {body.get('level')} @ {body.get('site')}")
    print(f"stored      {body.get('stored_parameter_count')} of "
          f"{body.get('parent_parameter_count')} parent  "
          f"(bound {body.get('storage_bound_ratio')}, "
          f"satisfied {body.get('storage_bound_satisfied')})")
    print(f"token match {body.get('token_match_vs_control_diagnostic')}  [DIAGNOSTIC ONLY]")
    print(f"protocol    {body['protocol']['protocol_sha256'][:16]}  "
          f"pack {body['protocol']['prompt_pack_sha256'][:16]}")
    print(f"dossier     {body['dossier_sha256']}")
    print(f"receipt     {body['receipt_sha256'][:16]}")
    print()
    if arguments.lane in ("child", "both"):
        _lane(body["child_responses"], "CHILD")
    if arguments.lane in ("parent", "both"):
        _lane(body["parent_responses"], "PARENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
