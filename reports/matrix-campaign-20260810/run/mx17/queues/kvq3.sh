#!/bin/bash
# The fourth cell of the 2x2: joint q/k with the value path OFF.  With wave 16s
# ARCH.KV6 (average, no value solve) as the fourth corner, the factorial reads
# which half of ADDENDUM 14 14.3 is doing the work.
set -u
. /home/cereal/nf-mx17/env.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY $MX17/mx17_build.py \
  --mode build --arm conditioned-local --child $W/children/MX17-KV6JQ --device cuda:0 \
  --kv-groups 6 --kv-solve joint --kv-als 0 > $W/logs/build-KV6JQ.log 2>&1
echo "KV6JQ rc=$? $(tail -1 $W/logs/build-KV6JQ.log)"
echo KVQ3 DONE
