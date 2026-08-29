#!/bin/bash
# Wave 29: the vocabulary fold carried onto this wave's bases.
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
# carried out of its own build receipt by `mx40_vocab.py`: a composed child that
# re-describes its parent's provenance in a driver is a child whose provenance is
# that driver's opinion.
#
#   vocabq.sh <TAG>:<K>:<source dir>[:<source cell receipt>] ...
#
# WAVE 32 ADDS ONE ENVIRONMENT SWITCH, `BELOW_FLOOR=1`, and puts it in the
# environment rather than in the colon-delimited spec on purpose: it releases the
# coverage half of wave 25's census gate for EVERY rung in the invocation, and a
# flag that dangerous should be visible on the command line that launched the
# batch rather than buried in the fourth field of the third argument.  The
# merge-closure assertion is not releasable and is not affected by it.
set -u
. /home/cereal/nf-mx40/env.sh
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs
FLOOR=""
[ "${BELOW_FLOOR:-0}" = "1" ] && FLOOR="--below-floor"

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  K=$(echo "$spec" | cut -d: -f2)
  SRC=$(echo "$spec" | cut -d: -f3)
  CELL=$(echo "$spec" | cut -d: -f4)
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; continue; fi
  rm -rf $W/children/MX40-$TAG
  $PY $MX40/mx40_vocab.py --mode build --k $K --tag $TAG $FLOOR \
    --source "$SRC" --source-cell-receipt "${CELL:-}" \
    --child $W/children/MX40-$TAG \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX40-$TAG/mx40-build-receipt.json ] || tail -8 $W/logs/vocabbuild-$TAG.log
done
echo "VOCABQ DONE"
