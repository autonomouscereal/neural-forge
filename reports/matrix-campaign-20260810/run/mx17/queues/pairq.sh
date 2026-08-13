#!/bin/bash
# The pair map, ordered by the behavioural single-block census wave 16 built and
# this wave completed.  Wave 16 has 13+11 (2/15) and 13+9 (6/15) already, so the
# anchor-13 family here is a DISTANCE ladder against those two: 13+7 at six
# blocks, 13+3 at ten, 13+1 at twelve.  The 7-anchored family asks whether the
# anchor matters or only the distance.
set -u
. /home/cereal/nf-mx17/env.sh
mkdir -p $W/children $W/logs
build () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY $MX17/mx17_build.py \
    --mode build --arm conditioned-local --child $W/children/MX17-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
build P1307 --excise 13 --excise 7
build P0709 --excise 7 --excise 9
build P1301 --excise 13 --excise 1
build P0711 --excise 7 --excise 11
build P0911 --excise 9 --excise 11
build P1303 --excise 13 --excise 3
echo PAIRQ DONE
