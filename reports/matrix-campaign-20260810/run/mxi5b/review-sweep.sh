#!/bin/bash
# Run the ADDENDUM 6 frozen review for every wave-5B cell of one lane that
# actually produced a child to review.
#
# A cell sealed `infra` has no installed-tensors.pt because the technique never
# returned weights, so there is nothing to reconstitute and nothing to grade.
# Skipping it here is not a behavioural judgement -- the receipt already records
# that the technique could not run the cell, and ADDENDUM 6 6.2 keeps that
# distinct from anything measured about the model.
set -u
GPU="${1:?gpu}"; shift
cd /home/cereal/nf-mxi
for CELL in "$@"; do
  DIR=/mnt/ai_data/neural-forge-matrix/cells/$CELL
  if [ ! -f "$DIR/installed-tensors.pt" ]; then
    echo "skip $CELL :: no installed-tensors.pt (technique returned no weights)"
    continue
  fi
  if [ -f "$DIR/frozen-review-dossier.json" ]; then
    echo "skip $CELL :: dossier already sealed"
    continue
  fi
  ./run-mxi-review.sh "$CELL" "$DIR" "$GPU"
done
echo "REVIEW LANE $GPU COMPLETE"
