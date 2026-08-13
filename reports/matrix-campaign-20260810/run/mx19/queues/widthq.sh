#!/bin/bash
# The width ladder.  Each rung is one build against the sealed basis for its own
# width; nothing else moves between them.
set -u
. /home/cereal/nf-mx19/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/children $W/logs
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX19/mx19_build.py \
    --mode build --arm conditioned-local --child $W/children/MX19-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
for spec in "$@"; do
  TAG=${spec%%:*}; ARGS=${spec#*:}
  run $TAG $ARGS
done
echo "WIDTHQ $GPU DONE"
