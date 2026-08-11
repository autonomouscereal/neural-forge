#!/bin/bash
# The C10 S2 cell landed in its own wave5 subdirectory, because the earlier
# evidence-refused attempt had already written a cell-receipt.json at the top
# level and run-mxi-cell.sh will not open a sealed receipt for writing.  The
# GPU0 follow-up looked at the top level and skipped it, so the review is run
# here against the directory that actually holds the child.
set -u
cd /home/cereal/nf-mxi
until grep -q "GPU0 FOLLOWUP COMPLETE" logs/gpu0-followup.log; do sleep 20; done
./run-mxi-review.sh C10.T1.S2.L02-wave5 /mnt/ai_data/neural-forge-matrix/cells/C10.T1.S2.L02/wave5 0
echo "C10 TAIL COMPLETE"
