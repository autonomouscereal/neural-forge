#!/bin/bash
# The lambda fix, and the merge step that died without it.
#
# Wave 17's 2x2 says the joint relation solve is worth three cases (5/15 -> 8/15)
# and the value/out_proj alternation it ships with costs five (8/15 -> 3/15),
# while improving its own fit objective by 16-29% at every site and losing
# held-out cosine.  That is an overfit, and `NOTES-MX-U` §7 named the fix: choose
# the alternation's lambda on an internal split of the FIT rows.  `KV6JS` is
# `KV6J` rebuilt with exactly that flag and nothing else moved, so the pair
# (3/15 against whatever this reads) prices the fix alone.  `KV4JS` is the 16 -> 8
# step, which read 0/15 with nine looping lanes, rebuilt with the full fixed
# solve -- the axis's second step has never been tried with a solve that
# generalises.
set -u
. /home/cereal/nf-mx18/env.sh
mkdir -p $W/children $W/logs
run () {
  TAG="$1"; GPU="$2"; shift 2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX18/mx18_build.py \
    --mode build --arm conditioned-local --child $W/children/MX18-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run KV6JS 0 --kv-groups 6 --kv-solve joint --kv-lambda-split
run KV4JS 0 --kv-groups 4 --kv-solve joint --kv-lambda-split
echo LAMQ DONE
