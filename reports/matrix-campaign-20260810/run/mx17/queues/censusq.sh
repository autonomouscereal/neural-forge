#!/bin/bash
# The six single-block excisions wave 16 did not build: L00 (the last conv) and
# L02/L04/L06/L08/L10 (five of the six attention blocks).  ARCH.X12 -- the
# smallest bite in wave 16 and the only dead single -- is what makes the
# attention blocks the interesting ones.
set -u
. /home/cereal/nf-mx17/env.sh
mkdir -p $W/children $W/logs
for L in 00 02 04 06 08 10; do
  n=$((10#$L))
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY $MX17/mx17_build.py \
    --mode build --arm conditioned-local --excise $n --child $W/children/MX17-X$L --device cuda:0 \
    > $W/logs/build-X$L.log 2>&1
  echo "X$L rc=$? $(tail -1 $W/logs/build-X$L.log)"
done
echo CENSUSQ DONE
