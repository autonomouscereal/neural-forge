#!/bin/bash
# Seal one ADDENDUM 6 behaviour verdict.  The verdict itself is NEVER passed:
# behavior_verdict_for computes it from the counts, and this wrapper only
# carries the readings.
set -u
CELL="${1:?cell}"; U="${2:?useful}"; P="${3:?partial}"; M="${4:?miss}"
SMOKE="${5:?smoke}"; SAFETY="${6:?safety}"; DEGEN="${7:-no}"; NOTES="${8:-}"
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16
W=/home/cereal/nf-mxl
DIR=/mnt/ai_data/neural-forge-matrix/wave8/cells/$CELL
FLAG=""
[ "$DEGEN" = "yes" ] && FLAG="--degenerate"
mkdir -p $W/cases
PYTHONPATH=$D/src /home/cereal/model-surgery-lab/venv/bin/python \
  $D/experiments/seal_matrix_behavior_review.py \
  --receipt $DIR/cell-receipt.json --dossier $DIR/frozen-review-dossier.json \
  --reviewer claude-opus5-mxl \
  --useful $U --partial $P --observed-miss $M --parent-useful-or-partial 12 \
  --smoke-verdict $SMOKE --safety $SAFETY $FLAG \
  --per-case-json $W/cases/$CELL.json \
  --ledger $W/ledger-review-$CELL.jsonl \
  --notes "$NOTES"
