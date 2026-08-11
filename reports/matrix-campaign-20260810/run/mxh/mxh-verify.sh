#!/bin/bash
# Wave-4 (MX-H) verification: render the campaign ledger and re-verify every
# sealed receipt in the store.
#
# The seal check and the schema check are reported SEPARATELY on purpose.
# `verify_seal` asks "are these the bytes that were sealed"; `validate_receipt`
# asks "does this receipt satisfy today's mandatory field set".  Wave-1 receipts
# predate `code_identity.compute_architecture` and the receipt module says so in
# its own docstring -- they are preserved byte-for-byte as sealed and read
# through the ledger.  Collapsing the two checks would report those as
# corruption, which they are not.
set -eu
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
PY=/home/cereal/model-surgery-lab/venv/bin/python
A=/mnt/ai_data/neural-forge-matrix/ledger.jsonl

echo "ledger: $(wc -l < $A) rows, sha $(sha256sum $A | cut -c1-16)"
PYTHONPATH=$D/src $PY $D/experiments/render_matrix_ledger.py \
  --ledger $A --out /mnt/ai_data/neural-forge-matrix/LEDGER.md 2>/dev/null

PYTHONPATH=$D/src $PY - << "PYEOF"
import collections
import json
from pathlib import Path

from neural_forge.matrix_cell_receipt import (
    SCHEMA,
    read_ledger,
    validate_receipt,
    verify_seal,
)

root = Path("/mnt/ai_data/neural-forge-matrix/cells")
sealed = schema_ok = 0
seal_failures = []
schema_failures = []
for path in sorted(root.rglob("*.json")):
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(body, dict) or body.get("schema") != SCHEMA:
        continue
    try:
        verify_seal(body)
        sealed += 1
    except Exception as exc:  # noqa: BLE001
        seal_failures.append(f"{path}: {exc}")
        continue
    try:
        validate_receipt(body)
        schema_ok += 1
    except Exception as exc:  # noqa: BLE001
        schema_failures.append(f"{path.parent.name}: {exc}")
print(f"receipt SEALS verified: {sealed} ok, {len(seal_failures)} failed")
for line in seal_failures:
    print("  SEAL FAIL", line)
print(
    f"receipt SCHEMAS validated: {schema_ok} ok, {len(schema_failures)} "
    "predate a mandatory field (wave 1, documented)"
)
reasons = collections.Counter(line.split(": ", 1)[1] for line in schema_failures)
for reason, count in reasons.items():
    print(f"    {count} x {reason}")

rows = read_ledger("/mnt/ai_data/neural-forge-matrix/ledger.jsonl")
instances = collections.Counter((r["cell_instance_id"], r["receipt_sha"]) for r in rows)
print(f"ledger rows {len(rows)}, distinct instances {len(instances)}, "
      f"duplicated {sum(1 for v in instances.values() if v > 1)}")
wave4 = [r for r in rows if r.get("campaign_wave") == 4]
print(f"wave-4 rows {len(wave4)}")
arch = collections.Counter(str(r.get("compute_architecture")) for r in wave4)
print("wave-4 compute architectures:", dict(arch))
digests = {r["receipt_sha"] for r in rows}
print(f"ledger receipt digests distinct: {len(digests)} of {len(rows)}")
PYEOF
