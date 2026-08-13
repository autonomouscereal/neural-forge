#!/bin/bash
# One rung: build the child, seal it under ADDENDUM 12, generate its dossier.
#   $1 tag  $2 gpu  $3.. builder axis flags
set -u
. /home/cereal/nf-mx17/env.sh
TAG="${1:?tag}"; GPU="${2:?gpu}"; shift 2
mkdir -p $W/children $W/logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX17/mx17_build.py \
  --mode build --arm conditioned-local --child $W/children/MX17-$TAG --device cuda:0 "$@" \
  > $W/logs/build-$TAG.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "BUILD FAILED $TAG rc=$rc"; tail -20 $W/logs/build-$TAG.log; exit $rc; fi
tail -2 $W/logs/build-$TAG.log
bash $MX17/seal_and_review.sh $TAG MX17-$TAG $GPU > $W/logs/review-$TAG.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "SEAL/REVIEW FAILED $TAG rc=$rc"; tail -30 $W/logs/review-$TAG.log; exit $rc; fi
tail -3 $W/logs/review-$TAG.log
echo "RUNG DONE $TAG"
