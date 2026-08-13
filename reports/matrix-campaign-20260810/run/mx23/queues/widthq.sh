#!/bin/bash
# The wave-23 rungs.  Each is one build against a SEALED basis; the only things
# that move between them are the basis artifact, the arm and the writers' lambda
# rule, which is the whole point of the wave.
set -u
. /home/cereal/nf-mx23/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/children $W/logs
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX23/mx23_build.py \
    --mode build --child $W/children/MX23-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
for spec in "$@"; do
  TAG=${spec%%:*}; ARGS=${spec#*:}
  run $TAG $ARGS
done
echo "WIDTHQ $GPU DONE"
