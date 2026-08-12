#!/bin/bash
# Solve + review one wave-8 child, in one lane, because a solved child with no
# dossier is not a result under ADDENDUM 6.
set -u
CELL="${1:?cell}"; MODE="${2:?mode}"; RECIPE="${3:?recipe}"; GPU="${4:?gpu}"
M=/mnt/ai_data/neural-forge-matrix
cd /home/cereal/nf-mxo
./run-mxo.sh "$CELL" "$MODE" "$RECIPE" "$GPU"
DIR=$M/wave11/cells/$CELL
if [ -f "$DIR/composed-child-tensors.pt" ]; then
  ./run-mxo-review.sh "$CELL" "$DIR" "$GPU"
else
  echo "NO REVIEW for $CELL :: no composed tensors were persisted"
fi
