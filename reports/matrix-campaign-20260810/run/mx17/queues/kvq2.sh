#!/bin/bash
set -u
. /home/cereal/nf-mx17/env.sh
run () {
  TAG="$1"; GPU="$2"; shift 2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX17/mx17_build.py \
    --mode build --arm conditioned-local --child $W/children/MX17-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run KV6J 4 --kv-groups 6 --kv-solve joint
run KV6A 4 --kv-groups 6 --kv-solve average
echo KVQ2 DONE
