#!/bin/bash
# The ADDENDUM 8 benchmark column, one model per spec, parent FIRST.  The
# parent's row is the campaign's third calibration and it is re-run every wave
# for a reason: a benchmark number is only comparable to the sealed reference if
# the invocation has not moved, and the cheapest way to know that is to
# re-measure the thing whose answer is already known.
#
#   benchq.sh <gpu> <LABEL>:<model dir> ...
set -u
. /home/cereal/nf-mx36/env.sh
GPU="$1"; shift
for spec in "$@"; do
  LABEL=${spec%%:*}; DIR=${spec#*:}
  bash $MX36/run-bench.sh $LABEL $DIR $GPU
done
echo "BENCHQ $GPU DONE"
