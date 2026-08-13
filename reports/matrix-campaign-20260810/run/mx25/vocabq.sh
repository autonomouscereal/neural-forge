#!/bin/bash
# Wave 25: the vocabulary coreset carried onto the QUALITY base, and the one
# point that decides whether the composition sign-change is a cliff or a crossing.
#
# Wave 24 measured the axis solo at three K and composed it with the block-13
# excision at two:
#
#     K        solo        composed with X13     delta
#     32,768   11 / 15     -- not built --       ?
#     16,384    9 / 15      5 / 15               -4
#      8,192    4 / 15      6 / 15               +2
#
# Two things are missing from that table and this queue builds both.  The top
# row is the arm the dose-response says should be the best child the axis can
# carry -- 0.801397x on the campaign's highest-scoring body -- and nobody has
# seen it.  The blank in the middle is where the delta changes sign, and one
# rung at K=12,288 is the cheapest thing that can say whether the crossing is
# smooth (a curve the campaign can model) or a cliff (a coincidence of two
# unrelated failures).  Both rungs are built solo AND composed, because a
# composed point whose solo partner does not exist cannot be differenced.
#
# Every rung is a POST-PROCESS on one sealed child's own bytes, so the whole
# queue is one CPU pass and only the reviews are rationed.  The source is NAMED
# rather than rebuilt and its axes are carried out of its own build receipt by
# `mx25_vocab.py` -- a composed child that re-describes its parent's provenance
# in a driver is a child whose provenance is that driver's opinion.
#
#   vocabq.sh <TAG>:<K>:<source dir>[:<source cell receipt>] ...
set -u
. /home/cereal/nf-mx25/env.sh
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  K=$(echo "$spec" | cut -d: -f2)
  SRC=$(echo "$spec" | cut -d: -f3)
  CELL=$(echo "$spec" | cut -d: -f4)
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; continue; fi
  rm -rf $W/children/MX25-$TAG
  $PY $MX25/mx25_vocab.py --mode build --k $K --tag $TAG \
    --source "$SRC" --source-cell-receipt "${CELL:-}" \
    --child $W/children/MX25-$TAG \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX25-$TAG/mx25-build-receipt.json ] || tail -8 $W/logs/vocabbuild-$TAG.log
done
echo "VOCABQ DONE"
