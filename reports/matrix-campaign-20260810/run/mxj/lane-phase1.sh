#!/bin/bash
# The greedy path, run optimistically: the recipe at step N is a pure function
# of the component ORDER, which is fixed before the wave starts, so the solves
# and the reviews do not depend on any grade.  Only the ESCALATION depends on a
# grade, so running the path first and grading it in order costs nothing and
# spends the GPU while the reviewer reads.
set -u
cd /home/cereal/nf-mxj
./drive.sh COMP.MA.MXJ2 M-A recipes/mxj-s2.json 0
./drive.sh COMP.MA.MXJ3 M-A recipes/mxj-s3.json 0
./drive.sh COMP.MA.MXJ4 M-A recipes/mxj-s4.json 0
./drive.sh COMP.MA.MXJ5 M-A recipes/mxj-s5.json 0
echo "PHASE1 LANE COMPLETE"
