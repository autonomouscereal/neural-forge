#!/bin/bash
# Wave 42: the DUAL-COLUMN DEEP-BAND ASSESSMENT.  Zero GPU, zero build, zero
# solve, and it never writes into a sealed artifact -- it reads the ledger, the
# sealed per-case verdicts and the frozen dossiers of every graded ARCH child at
# or below a stated ratio, and publishes both readings of the copy-back clause
# side by side.
#
# WHY IT RUNS TWICE.  The first invocation is MACHINE ONLY: it publishes the
# deterministic copy-back measure over every lane of every child in the band, and
# that artifact is what the reviewer reads FROM when adjudicating the held lanes.
# The second invocation takes `--hand-reading` and republishes with the reviewer's
# per-lane calls merged in, plus an explicit machine-vs-hand disagreement list.
# Running the hand pass first would let the reviewer's expectation choose the
# threshold; running the machine pass first and never revisiting it would let a
# threshold stand in for a reading.  Both are published.
#
#   dualbandq.sh <max ratio> <out json> [hand-reading json]
#
# The ledger read is READ-ONLY and the artifact is written under $W, never under
# a wave root that is already sealed.
set -u
. /home/cereal/nf-mx42/env.sh
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/logs

RATIO="${1:?max ratio}"
OUT="${2:?out json}"
HAND="${3:-}"
ARGS=""
[ -n "$HAND" ] && ARGS="--hand-reading $HAND"

$PY $MX42/mx42_dualband.py \
  --ledger $M/ledger.jsonl \
  --root $M \
  --max-ratio "$RATIO" \
  --out "$OUT" $ARGS 2>&1 | tail -40
RC=$?

# Parse what was written, in the invocation that wrote it.  NOTES-MX-AO
# section 9 -- and the generator asserts it too, on purpose, because this is the
# artifact a ruling will be made from.
$PY -c "import json,sys; a=json.load(open(sys.argv[1])); \
print('DUALBAND OK', a['schema'], a['children_assessed'], 'children'); \
print('  A floor:', a['floor_under_copy_back_counts']); \
print('  B floor:', a['floor_under_copy_back_refused'])" "$OUT"
echo "DUALBANDQ DONE rc=$RC"
