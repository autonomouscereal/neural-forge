#!/usr/bin/env python3
"""Dump every wave-3 cell's verbatim decode-smoke text for the frozen review.

The review rubric reads the verbatim receipt text and nothing else, so this
prints exactly what the receipt sealed: no truncation, no normalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/ai_data/neural-forge-matrix/cells")
WAVE = int(sys.argv[2]) if len(sys.argv) > 2 else 3

for directory in sorted(ROOT.iterdir()):
    if not directory.is_dir():
        continue
    for candidate in (
        directory / "cell-receipt.json",
        directory / "wave2" / "cell-receipt.json",
        directory / "wave3" / "cell-receipt.json",
    ):
        if not candidate.is_file():
            continue
        body = json.loads(candidate.read_text(encoding="utf-8"))
        if int(body.get("campaign_wave", 0)) != WAVE:
            continue
        smoke = body["behavior"]["decode_smoke"]
        if not smoke:
            continue
        free_run = body["behavior"]["free_run"]["token_match_vs_control"]
        print("=" * 78)
        print(
            "CELL {cell}   status={status}   free-run={free}   path={path}".format(
                cell=body["cell_id"],
                status=body["outcome"]["status"],
                free=free_run,
                path=candidate,
            )
        )
        for index, entry in enumerate(smoke):
            print("--- smoke[{i}] terminated={t}".format(i=index, t=entry["terminated"]))
            print(entry["output_verbatim"])
        print()
