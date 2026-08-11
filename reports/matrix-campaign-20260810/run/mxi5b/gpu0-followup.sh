#!/bin/bash
# GPU 0 tail of wave 5B: wait out the lane-A review batch, then re-run C10 on
# the evidence it is actually specified against and review the C09 cells.
#
# C10's first two attempts refused on the evidence, not on storage:
# `matrix_technique_registry` says in as many words that C10's endpoints moved
# to C09's in wave 3 -- attention.context and attention.o -- "so a C10 cell
# re-uses C09's authenticated site evidence rather than building a second copy
# of it".  The L02-C10 cache is the pre-move capture and binds
# operator.input/operator.output, which is why the harness refused it.  The fix
# is the cache tag, and nothing about the cell changes.
set -u
cd /home/cereal/nf-mxi
until grep -q "REVIEW LANE 0 COMPLETE" logs/review5b-a.log; do sleep 20; done

# C10.T1.S4.L02 is expected to MISS the storage bound and is run to record that
# refusal from the kernel's own count rather than from the planner's.  The
# head-drop kernel counts k_proj and v_proj in full -- 1,048,576 parameters the
# child still stores after every query head is gone -- against a 0.25x budget of
# 786,432, so no head count can satisfy S4.  S2's budget is 1,572,864 and four
# surviving heads store exactly that: a 4x drop in HEAD COUNT that is only a 2x
# drop in stored parameters, which is the honest way to state it.
./run-mxi-cell.sh C10.T1.S4.L02 L02-C09 0 384
./run-mxi-cell.sh C10.T1.S2.L02 L02-C09 0 256
for CELL in C10.T1.S2.L02 C09.T1.S4.L02 C09.T2.S4.L02 C09.T3.S4.L02 C09.T4.S4.L02 C09.T5.S4.L02 C09.T8.S4.L02; do
  DIR=/mnt/ai_data/neural-forge-matrix/cells/$CELL
  if [ -f "$DIR/installed-tensors.pt" ] && [ ! -f "$DIR/frozen-review-dossier.json" ]; then
    ./run-mxi-review.sh "$CELL" "$DIR" 0
  else
    echo "skip $CELL :: no tensors or dossier already sealed"
  fi
done
echo "GPU0 FOLLOWUP COMPLETE"
