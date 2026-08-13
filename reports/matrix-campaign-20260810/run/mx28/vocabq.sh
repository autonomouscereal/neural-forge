#!/bin/bash
# Wave 28: the vocabulary fold carried onto this wave's bases.
#
# Wave 27 found that the 13/15 "shelf" is a PEAK: K = 19,456 (ARCH.V19) reads
# 14/15 at 0.794570x with zero cap hits, two cases above the parent, and a
# 1,024-row step in either direction costs a case (11 / 13 / 14 / 13 / 13).  It
# also found that the cost of composing an excision onto a vocabulary base is a
# property of the BLOCK and not of K -- block 13 costs three on any
# shelf-or-better base, block 7 costs four, block 9 costs three to four.
#
# So this queue asks the one question those two results pose together: does the
# SUMMIT base carry the folds the shoulder bases carry, and does it carry them
# for the same price?  Every V19* rung below is the identical post-process the
# V18 and V20 rungs got, on the identical bodies, so the K column is controlled
# to the byte and a delta is the base's.
#
# It also rebuilds V18X13X01 -- wave 27 graded it 3/15 and did not retain it --
# because the lambda ladder in `gainq.sh` needs the zero-solve control it is
# differenced against to exist.  That rebuild is CHECKED against wave 27's
# sealed digest by `driftq.sh --check-vocab`, which is a free reproduction test
# of the vocabulary path across a wave boundary.
#
# Every rung here is a POST-PROCESS on one sealed child's own bytes -- one CPU
# pass, no capture, no forward call -- so the whole queue is cheap and only the
# reviews are rationed.  The source is NAMED rather than rebuilt and its axes are
# carried out of its own build receipt by `mx28_vocab.py`: a composed child that
# re-describes its parent's provenance in a driver is a child whose provenance is
# that driver's opinion.
#
#   vocabq.sh <TAG>:<K>:<source dir>[:<source cell receipt>] ...
set -u
. /home/cereal/nf-mx28/env.sh
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  K=$(echo "$spec" | cut -d: -f2)
  SRC=$(echo "$spec" | cut -d: -f3)
  CELL=$(echo "$spec" | cut -d: -f4)
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; continue; fi
  rm -rf $W/children/MX28-$TAG
  $PY $MX28/mx28_vocab.py --mode build --k $K --tag $TAG \
    --source "$SRC" --source-cell-receipt "${CELL:-}" \
    --child $W/children/MX28-$TAG \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX28-$TAG/mx28-build-receipt.json ] || tail -8 $W/logs/vocabbuild-$TAG.log
done
echo "VOCABQ DONE"
