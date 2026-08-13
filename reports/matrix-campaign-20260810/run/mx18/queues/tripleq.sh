#!/bin/bash
# Wave 17's distance law, at three bodies.  Eight pairs, sorted by the distance
# between the two blocks, scored 2, 2, 4 | 6, 6 | 6 | 7 | 7 of fifteen -- monotone,
# no inversion, two anchors, and NOT a function of the two blocks' solo scores.
# The law's direct prediction is that the cheapest triple is the one whose three
# pairwise distances are all large, so they are built widest-spread first:
#
#   1+7+13  distances 6, 6, 12   (the two anchors of the pair map, plus its best)
#   1+9+13  distances 8, 4, 12
#   3+7+13  distances 4, 6, 10
#   3+9+13  distances 6, 4, 10
#
# Every one is 0.842437x, three conv blocks at the identical parameter count, so
# the only thing that moves between them is the spread.  Wave 16's 13+11+9 --
# distances 2, 2, 4, the tightest triple there is -- read 0/15, which is the
# fourth corner of this and the reason the order runs widest first.
set -u
. /home/cereal/nf-mx18/env.sh
mkdir -p $W/children $W/logs
build () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY $MX18/mx18_build.py \
    --mode build --arm conditioned-local --child $W/children/MX18-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
build T010713 --excise 1 --excise 7 --excise 13
build T010913 --excise 1 --excise 9 --excise 13
build T030713 --excise 3 --excise 7 --excise 13
build T030913 --excise 3 --excise 9 --excise 13
echo TRIPLEQ DONE
