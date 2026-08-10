#!/usr/bin/env python3
"""Seal the frozen decode-smoke review of every wave-2 cell, serialized.

`experiments/seal_matrix_cell_review.py` is the sealer; this drives it once per
receipt from ONE process so the ledger appends stay serialized -- the same
reason the two GPU batches wrote separate shards.  Each review is a new sealed
instance; the unreviewed row stays in the append-only ledger.

Takes explicit receipt *paths* rather than cell ids, because five wave-2 cells
re-run a cell id wave 1 already sealed and their receipts live in a `wave2`
subdirectory beside wave 1's untouched artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952")
LEDGER = Path("/mnt/ai_data/neural-forge-matrix/ledger.jsonl")
PYTHON = "/home/cereal/model-surgery-lab/venv/bin/python"

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
reviewer = spec["reviewer"]

before = len([line for line in LEDGER.read_bytes().split(b"\n") if line.strip()])
failures: list[str] = []

for entry in spec["reviews"]:
    receipt = Path(entry["receipt"])
    if not receipt.is_file():
        failures.append(str(receipt) + ": no receipt")
        continue
    body = json.loads(receipt.read_text(encoding="utf-8"))
    if not body["behavior"]["decode_smoke"]:
        failures.append(body["cell_id"] + ": no decode smoke to review")
        continue
    result = subprocess.run(
        [
            PYTHON,
            str(DEPLOY / "experiments" / "seal_matrix_cell_review.py"),
            "--receipt",
            str(receipt),
            "--verdict",
            entry["verdict"],
            "--reviewer",
            reviewer,
            "--notes",
            entry["notes"],
            "--ledger",
            str(LEDGER),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(DEPLOY / "src"), "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        failures.append(body["cell_id"] + ": " + result.stderr.strip()[:400])
        continue
    print(result.stdout.splitlines()[0])

after = len([line for line in LEDGER.read_bytes().split(b"\n") if line.strip()])
print("\nledger rows {before} -> {after} (+{delta})".format(
    before=before, after=after, delta=after - before))
if failures:
    print("\nFAILURES:")
    for line in failures:
        print("  " + line)
    raise SystemExit(1)
