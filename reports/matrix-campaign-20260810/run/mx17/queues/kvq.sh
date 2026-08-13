#!/bin/bash
# ADDENDUM 14 14.3: the identity gate first, then the joint solve, then the
# average-in-gauge rebuild that reproduces wave 16 through wave 17 code.
set -u
. /home/cereal/nf-mx17/env.sh
mkdir -p $W/children $W/logs $W/probes
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY $MX17/mx17_build.py \
    --mode build --arm conditioned-local --child $W/children/MX17-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run KVID --kv-groups 8 --kv-identity-gate --kv-solve joint
grep -E "identity|A0[0-9]|A1[0-9]" $W/logs/build-KVID.log | head -20
run KV6J --kv-groups 6 --kv-solve joint
echo KVQ DONE
