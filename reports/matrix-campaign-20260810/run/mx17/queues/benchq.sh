#!/bin/bash
# ADDENDUM 8 benchmark column, parent first so the lane is calibrated before a
# child is quoted.  Only children that do not loop are benchmarked.
set -u
. /home/cereal/nf-mx17/env.sh
GPU="${1:?gpu}"
bash $MX17/run-bench.sh PARENT /home/cereal/model-surgery-lab/models/LFM2.5-230M $GPU | tail -3
for TAG in KV6JQ P1301 P1307 X06 X10; do
  bash $MX17/run-bench.sh ARCH.$TAG $W/children/MX17-$TAG $GPU | tail -2
done
echo BENCHQ DONE
