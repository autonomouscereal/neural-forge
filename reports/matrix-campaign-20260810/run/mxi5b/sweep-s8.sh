#!/bin/bash
# Wave 5B step 2: the S8 ladder rung above every S4 cell that came back
# alive-parent-band.
#
# ADDENDUM 6 6.3: "where S4 is alive-parent-band under some technique, probe
# S8 at that technique and site".  Six cells qualify after step 1 -- C11 under
# T1, T2 and T8, and C13 under T4, T5 and T8 -- so all six are probed.  C12 is
# absent on purpose: its best S4 result is alive-degraded at 9/15, which is not
# the rung 6.3 says to climb from.
#
# Solve then review in one lane, because the review is the grade and a solved
# cell with no dossier is not a result.
set -u
GPU="${1:?gpu}"
cd /home/cereal/nf-mxi
solve_and_review() {
  local CELL="$1" CACHE="$2" WIDTH="$3"
  ./run-mxi-cell.sh "$CELL" "$CACHE" "$GPU" "$WIDTH"
  local DIR=/mnt/ai_data/neural-forge-matrix/cells/$CELL
  if [ -f "$DIR/installed-tensors.pt" ]; then
    ./run-mxi-review.sh "$CELL" "$DIR" "$GPU"
  else
    echo "no review for $CELL :: technique returned no weights"
  fi
}
solve_and_review C11.T1.S8.L00 L00-C11 91
solve_and_review C11.T2.S8.L00 L00-C11 91
solve_and_review C11.T8.S8.L00 L00-C11 91
solve_and_review C13.T4.S8.L05 L05-C13 91
solve_and_review C13.T5.S8.L05 L05-C13 319
solve_and_review C13.T8.S8.L05 L05-C13 319
echo "S8 LANE COMPLETE"
