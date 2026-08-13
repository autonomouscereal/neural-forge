#!/bin/bash
# The frontier attempts, built while the review queues run: the best pair so far
# (13+1) stacked with the joint merge and with the one width wave 16 got a
# non-zero score at, plus block 13 with the joint merge as the direct control on
# wave 16 ARCH.X13KV6 (6/15, average-in-gauge).
set -u
. /home/cereal/nf-mx17/env.sh
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY $MX17/mx17_build.py \
    --mode build --arm conditioned-local --child $W/children/MX17-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run X13KV6J --excise 13 --kv-groups 6 --kv-solve joint
run F1301KV6 --excise 13 --excise 1 --kv-groups 6 --kv-solve joint
run F1301W1536 --excise 13 --excise 1 --ff 1536
echo FRONTIERQ DONE
