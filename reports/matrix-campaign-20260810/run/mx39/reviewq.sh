#!/bin/bash
# Seal + frozen review, one child per invocation.
#
# The refusal at the top is wave 19's operational mistake, fixed and kept: that
# wave queued one child's review on both GPUs, sealed it twice, and left two
# ARCH rows for one measurement in the shard.  A tag whose cell receipt already
# exists is refused HERE rather than resolved later by a merge guard that would
# have to choose between two receipts for one child.  FORCE=1 is what a genuine
# re-run is.
set -u
. /home/cereal/nf-mx39/env.sh
GPU="${1:?gpu}"; shift
for tag in "$@"; do
  if [ -f "$W/cells/ARCH.$tag/cell-receipt.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "$tag REFUSED: $W/cells/ARCH.$tag/cell-receipt.json already exists (FORCE=1 to re-seal)"
    continue
  fi
  bash $MX39/seal_and_review.sh $tag MX39-$tag $GPU > $W/logs/review-$tag.log 2>&1
  rc=$?
  echo "$tag rc=$rc :: $(grep -h "sealed ARCH" $W/logs/review-$tag.log | tail -1) :: $(grep -h "sealed review surface" $W/logs/review-$tag.log | tail -1)"
  [ $rc -ne 0 ] && tail -12 $W/logs/review-$tag.log
done
echo "REVIEWQ $GPU DONE"
