#!/bin/bash
# GPU 4 tail of wave 5B: wait out the S8 probe lane, then review the short-conv
# output projection and attention-value cells solved by lane B.
set -u
cd /home/cereal/nf-mxi
until grep -q "S8 LANE COMPLETE" logs/s8-lane.log; do sleep 20; done
for CELL in C05.T1.S4.L05 C05.T2.S4.L05 C05.T3.S4.L05 C05.T4.S4.L05 C05.T5.S4.L05 C05.T8.S4.L05 \
            C08.T1.S4.L02 C08.T2.S4.L02 C08.T5.S4.L02 C08.T8.S4.L02; do
  DIR=/mnt/ai_data/neural-forge-matrix/cells/$CELL
  if [ -f "$DIR/installed-tensors.pt" ] && [ ! -f "$DIR/frozen-review-dossier.json" ]; then
    ./run-mxi-review.sh "$CELL" "$DIR" 4
  else
    echo "skip $CELL :: no tensors or dossier already sealed"
  fi
done
echo "GPU4 FOLLOWUP COMPLETE"
