#!/bin/bash
# Every excision the harness bug touched, rebuilt against a stream that reads
# only the past.  `mx18_mask_probe.py` proved the defect: an excision shifts
# every later attention block into a conv's slot in `config.layer_types`, the
# live model hands it the padding mask instead of the causal one, and it attends
# to the future.  The EXPORTED children were never wrong -- their own config is
# written correctly and every graded response came from a stock reload of it --
# but every solve conditioned on the live stream was, at and after the first
# shifted attention block.
#
# So this rebuilds, one axis at a time and nothing else moved, the readings the
# census rests on:
#
#   X13C        the CONTROL.  Block 13 is last and shifts no attention block, so
#               the fix must change nothing: X13C has to come back bit-identical
#               to wave 16's ARCH.X13A or the fix is not the fix.
#   X00C        the census's most expensive block, 0/15 with all sixteen lanes
#               charged -- and the excision that shifts EVERY attention site
#   X07C        the campaign's best per-child alive reading, 11/15
#   P1301C      the best pair, 7/15, and the child whose gate refused
#   T010713C    this wave's best triple, 3/15
#   X07KV6JQC   the merge on the best single, rebuilt (step 3)
set -u
. /home/cereal/nf-mx18/env.sh
GPU="${1:?gpu}"
mkdir -p $W/children $W/logs
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX18/mx18_build.py \
    --mode build --arm conditioned-local --child $W/children/MX18-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run X13C --excise 13
run X00C --excise 0
run X07C --excise 7
run P1301C --excise 13 --excise 1
run T010713C --excise 1 --excise 7 --excise 13
run X07KV6JQC --excise 7 --kv-groups 6 --kv-solve joint --kv-als 0
echo CAUSALQ DONE
