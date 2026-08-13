#!/bin/bash
# Wave 26: the V20 stack.  Wave 25 found one rung of the vocabulary axis that
# reads THIRTEEN of fifteen at 0.799135x -- one case above the parent, zero cap
# hits, zero solve -- and immediately wrote down why nobody should believe it as
# a claim about intelligence.  This queue takes the only position that can settle
# what the rung is worth: STACK OFF IT.  A base that is a measurement artefact
# collapses under a second fold; a base that is genuinely intact carries one.
#
# Wave 25's own composed row is the prior.  K = 32,768 on the block-13 excision
# cost exactly ONE case (11 -> 10) while paying 5.3 points of ratio, and it did
# that by converting six Useful lanes into nine Partial ones.  If the same
# excision costs the 20,480 rung the same one case, V20X13 reads about twelve at
# 0.7466x, which would be the deepest reading anywhere near the parent's band.
#
# Every rung here is a POST-PROCESS on one sealed child's own bytes -- one CPU
# pass, no capture, no forward call -- so the whole queue is cheap and only the
# reviews are rationed.  The source is NAMED rather than rebuilt and its axes are
# carried out of its own build receipt by `mx26_vocab.py`: a composed child that
# re-describes its parent's provenance in a driver is a child whose provenance is
# that driver's opinion.
#
#   vocabq.sh <TAG>:<K>:<source dir>[:<source cell receipt>] ...
set -u
. /home/cereal/nf-mx26/env.sh
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  K=$(echo "$spec" | cut -d: -f2)
  SRC=$(echo "$spec" | cut -d: -f3)
  CELL=$(echo "$spec" | cut -d: -f4)
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; continue; fi
  rm -rf $W/children/MX26-$TAG
  $PY $MX26/mx26_vocab.py --mode build --k $K --tag $TAG \
    --source "$SRC" --source-cell-receipt "${CELL:-}" \
    --child $W/children/MX26-$TAG \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX26-$TAG/mx26-build-receipt.json ] || tail -8 $W/logs/vocabbuild-$TAG.log
done
echo "VOCABQ DONE"
