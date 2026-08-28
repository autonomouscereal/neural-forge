#!/bin/bash
# Wave 29: COMPLEMENT-MERGE.  One CPU pass per blend, no GPU, no solve.
#
#   blendq.sh <TAG>:<source dir>:<other dir>:alpha=<a>|part=<k> ...
set -u
. /home/cereal/nf-mx29/env.sh
mkdir -p $W/children $W/logs
for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  SRC=$(echo "$spec" | cut -d: -f2)
  OTH=$(echo "$spec" | cut -d: -f3)
  MODE=$(echo "$spec" | cut -d: -f4)
  case "$MODE" in
    alpha=*) FLAG="--alpha ${MODE#alpha=}" ;;
    part=*)  FLAG="--partition ${MODE#part=}" ;;
    *) echo "$TAG SKIPPED: mode '$MODE' is neither alpha= nor part="; continue ;;
  esac
  if [ ! -d "$SRC" ] || [ ! -d "$OTH" ]; then echo "$TAG SKIPPED: missing source"; continue; fi
  rm -rf $W/children/MX29-$TAG
  $PY $MX29/mx29_blend.py --source "$SRC" --other "$OTH" \
    --child $W/children/MX29-$TAG --tag $TAG $FLAG \
    > $W/logs/blend-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h '^blend ' $W/logs/blend-$TAG.log | tail -1)"
  [ -f $W/children/MX29-$TAG/mx29-build-receipt.json ] || tail -6 $W/logs/blend-$TAG.log
done
echo "BLENDQ DONE"
