#!/bin/bash
# Wave 24: the vocabulary DOSE-RESPONSE.  Wave 23 built one rung (K=16,384) and
# read 9/15 at 0.780874x; a single point on an axis is a datum, not a curve, and
# the campaign cannot say whether the coreset's cost is linear in the rows it
# drops, flat until a cliff, or already past one.  This builds the two rungs
# either side of it -- K=8,192 (0.744354x) and K=32,768 (0.853916x) -- from the
# SAME census, the same closure and the same zero-solve gather, so the three
# points differ in exactly one integer.
#
# The composed rung is K=8,192 on ARCH.X13A, the campaign's highest-scoring
# child: 0.691827x, which is materially below the re-review bar in a way no
# solo rung reaches.
#
# Every rung is a POST-PROCESS on one sealed child's own bytes, so the whole
# ladder is one CPU pass and only the reviews are rationed.  The source is NAMED
# rather than rebuilt and its axes are carried out of its own build receipt by
# `mx24_vocab.py` -- a composed child that re-describes its parent's provenance
# in a driver is a child whose provenance is that driver's opinion.
#
#   vocabq.sh <K> <TAGPREFIX> [rung ...]     (no rungs = the whole ladder)
set -u
. /home/cereal/nf-mx24/env.sh
K="${1:?K}"; PREFIX="${2:?tag prefix, e.g. V8}"; shift 2
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs

run () {
  TAG="$1"; SRC="$2"; CELL="$3"
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; return; fi
  rm -rf $W/children/MX24-$TAG
  $PY $MX24/mx24_vocab.py --mode build --k $K --tag $TAG \
    --source "$SRC" --source-cell-receipt "$CELL" \
    --child $W/children/MX24-$TAG \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
}

for RUNG in ${@:-SOLO X13 CL896}; do
case $RUNG in
  # the pure axis at this K -- what makes every rung below it attributable
  SOLO)  run $PREFIX        /home/cereal/model-surgery-lab/models/LFM2.5-230M "" ;;
  # the campaign's highest-scoring child (13/15) carrying this K on top
  X13)   run ${PREFIX}X13   $M/wave16/children/MX16-X13A  $M/wave16/cells/ARCH.X13A/cell-receipt.json ;;
  # the frontier body; at K=8,192 this is 0.437x, and wave 23's K=16,384 arm of
  # it held exactly one lane, so this rung is built for the curve and reviewed
  # only if the GPU is otherwise idle
  CL896) run ${PREFIX}CL896 $M/wave15/children/MX15-CL896 $M/wave15/cells/ARCH.CL896/cell-receipt.json ;;
  *) echo "$RUNG: no such rung" ;;
esac
done
echo "VOCABQ DONE"
