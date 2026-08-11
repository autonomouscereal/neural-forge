#!/bin/bash
# Wave-4 (MX-H) finalize: merge the per-cell ledger shards into the authoritative
# campaign ledger, re-render, and re-verify every sealed receipt.
#
# One serialized appender, as waves 1-3 established: append_ledger issues two
# write() calls per row and O_APPEND makes each atomic, not the pair.
set -eu
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
PY=/home/cereal/model-surgery-lab/venv/bin/python
W=/home/cereal/nf-mxh
A=/mnt/ai_data/neural-forge-matrix/ledger.jsonl
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

cp -p $A /mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.before-mxh-merge-$STAMP
echo "ledger before merge: $(wc -l < $A) rows, sha $(sha256sum $A | cut -c1-16)"

PYTHONPATH=$D/src $PY $W/../nf-mxg/merge_shards.py $A \
  $W/ledger-COMP.MA.RBEH.jsonl \
  $W/ledger-COMP.MA.RSTO.jsonl \
  $W/ledger-COMP.MB.RSTO.jsonl \
  $W/ledger-COMP.MB.RBEH.jsonl \
  $W/ledger-reviews.jsonl

echo "ledger after merge:  $(wc -l < $A) rows, sha $(sha256sum $A | cut -c1-16)"

PYTHONPATH=$D/src $PY $D/experiments/render_matrix_ledger.py \
  --ledger $A --out /mnt/ai_data/neural-forge-matrix/LEDGER.md

echo "--- re-verifying every sealed receipt in the campaign store ---"
PYTHONPATH=$D/src $PY - << "PYEOF"
import json
from pathlib import Path
from neural_forge.matrix_cell_receipt import SCHEMA, validate_receipt, verify_seal

root = Path("/mnt/ai_data/neural-forge-matrix/cells")
ok = bad = 0
failures = []
for path in sorted(root.rglob("*.json")):
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(body, dict) or body.get("schema") != SCHEMA:
        continue
    try:
        verify_seal(body)
        validate_receipt(body)
        ok += 1
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        bad += 1
        failures.append(f"{path}: {type(exc).__name__}: {exc}")
print(f"receipt seals verified: {ok} ok, {bad} failed")
for line in failures:
    print("  FAIL", line)
PYEOF
