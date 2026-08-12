#!/bin/bash
# Seal one ADDENDUM 6 behaviour verdict with its ADDENDUM 9 per-lane reading.
# The verdict is NEVER passed: behavior_verdict_for computes it from the counts.
#
#   $1 cell  $2 useful  $3 partial  $4 miss  $5 smoke  $6 safety  $7 notes
#   $8..     zero or more case ids the degeneracy clause charges
set -u
CELL="${1:?cell}"; U="${2:?useful}"; P="${3:?partial}"; M="${4:?miss}"
SMOKE="${5:?smoke}"; SAFETY="${6:?safety}"; NOTES="${7:-}"
shift 7
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8
W=/home/cereal/nf-mxq
DIR=/mnt/ai_data/neural-forge-matrix/wave13/cells/$CELL
LANES=""
DEGEN=""
for lane in "$@"; do LANES="$LANES --degenerate-lane $lane"; DEGEN="--degenerate"; done
mkdir -p $W/cases
PYTHONPATH=$D/src /home/cereal/model-surgery-lab/venv/bin/python \
  $D/experiments/seal_matrix_behavior_review.py \
  --receipt $DIR/cell-receipt.json --dossier $DIR/frozen-review-dossier.json \
  --reviewer claude-opus5-mxq \
  --useful $U --partial $P --observed-miss $M --parent-useful-or-partial 12 \
  --smoke-verdict $SMOKE --safety "$SAFETY" $DEGEN $LANES \
  --per-case-json $W/cases/$CELL.json \
  --ledger $W/ledger-review-$CELL.jsonl \
  --notes "$NOTES"
