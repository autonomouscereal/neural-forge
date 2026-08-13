#!/bin/bash
# The 2x2 says the relation solve is worth three cases (5/15 -> 8/15) and the
# value alternation costs five (8/15 -> 3/15), so everything below is joint q/k
# with the value path OFF: the deeper merge, the merge on the best single
# excision, and the merge on the best pair.
set -u
. /home/cereal/nf-mx17/env.sh
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY $MX17/mx17_build.py \
    --mode build --arm conditioned-local --child $W/children/MX17-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run KV4JQ --kv-groups 4 --kv-solve joint --kv-als 0
run X13KV6JQ --excise 13 --kv-groups 6 --kv-solve joint --kv-als 0
run F1301KV6Q --excise 13 --excise 1 --kv-groups 6 --kv-solve joint --kv-als 0
echo KVQ4 DONE
