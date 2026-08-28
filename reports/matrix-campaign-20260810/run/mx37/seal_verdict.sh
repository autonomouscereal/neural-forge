#!/bin/bash
# Seal one ADDENDUM 6 behaviour verdict for a wave-29 standard-shape child.
# `experiments/seal_matrix_behavior_review.py` is untouched by ADDENDUM 12 on
# purpose: a shape-changing child is graded on the same surface as every other.
#
# ADDENDUM 14: the safety lane is chargeable (its id comes from the dossier), and
# a review that charges NOTHING says so with the word `none` in place of the lane
# list, which seals the `none` reading instead of `unstated`.
#
#   NF_DEPLOY=<dir> ./seal_verdict.sh <tag> <useful> <partial> <miss> <safety> <notes> none
#   NF_DEPLOY=<dir> ./seal_verdict.sh <tag> <useful> <partial> <miss> <safety> <notes> <lane id>...
set -u
TAG="${1:?tag}"; U="${2}"; P="${3}"; M="${4}"; SAFETY="${5}"; NOTES="${6}"
shift 6
D="${NF_DEPLOY:?set NF_DEPLOY to the deployment cut from THIS wave-29 commit}"
W=/mnt/ai_data/neural-forge-matrix/wave37
PY=/home/cereal/model-surgery-lab/venv/bin/python
ARGS=()
if [ "${1:-}" = "none" ]; then
  ARGS+=(--no-charged-lanes)
else
  for lane in "$@"; do ARGS+=(--degenerate-lane "$lane"); done
  if [ ${#ARGS[@]} -gt 0 ]; then ARGS+=(--degenerate); fi
fi
PYTHONPATH="$D/src:$D/experiments" $PY $D/experiments/seal_matrix_behavior_review.py \
  --receipt $W/cells/ARCH.$TAG/cell-receipt.json \
  --dossier $W/cells/ARCH.$TAG/frozen-review-dossier.json \
  --reviewer claude-opus5-mx37 \
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
