#!/bin/bash
set -u
. /home/cereal/nf-mx18/env.sh
GPU="${1:?gpu}"; shift
for tag in "$@"; do
  bash $MX18/seal_and_review.sh $tag MX18-$tag $GPU > $W/logs/review-$tag.log 2>&1
  rc=$?
  echo "$tag rc=$rc :: $(grep -h "sealed ARCH" $W/logs/review-$tag.log | tail -1) :: $(grep -h "sealed review surface" $W/logs/review-$tag.log | tail -1)"
  [ $rc -ne 0 ] && tail -12 $W/logs/review-$tag.log
done
echo "REVIEWQ $GPU DONE"
