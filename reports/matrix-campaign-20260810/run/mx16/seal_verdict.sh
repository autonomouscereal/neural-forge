#!/bin/bash
# Seal one ADDENDUM 6 behaviour verdict for a wave-16 standard-shape child.
# `experiments/seal_matrix_behavior_review.py` is untouched by ADDENDUM 12 on
# purpose: a shape-changing child is graded on the same surface as every other.
#
#   NF_DEPLOY=<dir> ./seal_verdict.sh <tag> <useful> <partial> <miss> <safety> <notes> [degenerate lane ids...]
set -u
TAG="${1:?tag}"; U="${2}"; P="${3}"; M="${4}"; SAFETY="${5}"; NOTES="${6}"
shift 6
D="${NF_DEPLOY:?set NF_DEPLOY to this wave's own deployment directory}"
W=/mnt/ai_data/neural-forge-matrix/wave16
PY=/home/cereal/model-surgery-lab/venv/bin/python
ARGS=()
for lane in "$@"; do ARGS+=(--degenerate-lane "$lane"); done
if [ ${#ARGS[@]} -gt 0 ]; then ARGS+=(--degenerate); fi
PYTHONPATH="$D/src:$D/experiments" $PY $D/experiments/seal_matrix_behavior_review.py \
  --receipt $W/cells/ARCH.$TAG/cell-receipt.json \
  --dossier $W/cells/ARCH.$TAG/frozen-review-dossier.json \
  --reviewer claude-opus5-mx16 \
  --useful $U --partial $P --observed-miss $M \
  --parent-useful-or-partial 12 \
  --smoke-verdict coherent \
  --safety "$SAFETY" \
  --per-case-json $W/cells/ARCH.$TAG/per-case.json \
  --lanes-total 16 \
  --notes "$NOTES" \
  --ledger $W/shards/verdict-$TAG.jsonl \
  --out $W/cells/ARCH.$TAG/cell-receipt-behavior-reviewed.json \
  "${ARGS[@]}"
