#!/bin/bash
# Generate the frozen review dossier for a child whose ARCH receipt is ALREADY
# sealed, without going near the sealer.
#
# This exists because of a real failure and the right response to it.  Wave 21's
# deployment was cut as a hardlink copy of wave 20's plus a verified delta --
# cheap, and byte-true against the commit -- and the frozen review then refused
# every child with `manual prompt artifact must be one ordinary file`.  The
# guard is `st_nlink != 1` in `lfm2_carrier_review_cli._ordinary_file`, and it
# is RIGHT: a prompt pack reachable under a second name is a prompt pack
# something else can rewrite, and the review's whole authority is that the pack
# it read is the pack the seal names.  The deployment's links were broken and
# re-verified against the commit manifest; nothing about the review moved.
#
# By then the four ARCH receipts were sealed and their shard rows written -- the
# seal is step one and it succeeded.  Re-running `seal_and_review.sh` with
# FORCE=1 would seal a SECOND receipt for each child and leave the merge to
# choose between two rows for one measurement, which is exactly wave 19's
# double-seal.  So this runs step two only, against the receipt that already
# exists, and refuses a tag whose receipt does not.
#
#   reviewonly.sh <gpu> <tag>...
set -u
. /home/cereal/nf-mx22/env.sh
GPU="${1:?gpu}"; shift
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$NF_DEPLOY/src:$NF_DEPLOY/experiments"

for TAG in "$@"; do
  R=$W/cells/ARCH.$TAG/cell-receipt.json
  if [ ! -f "$R" ]; then
    echo "$TAG REFUSED: no sealed receipt at $R"; continue
  fi
  if [ -f "$W/cells/ARCH.$TAG/frozen-review-dossier.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "$TAG REFUSED: dossier already exists (FORCE=1 to regenerate)"; continue
  fi
  $PY $NF_DEPLOY/experiments/run_lfm2_composition_frozen_review.py \
    --receipt "$R" \
    --child-model $W/children/MX22-$TAG \
    --parent-model $PARENT \
    --out $W/cells/ARCH.$TAG/frozen-review-dossier.json \
    --device cuda:0 --require-compute-architecture sm86 \
    --dtype float32 --maximum-new-tokens 1024 \
    > $W/logs/review-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'sealed review surface' $W/logs/review-$TAG.log | tail -1)"
done
echo "REVIEWONLY $GPU DONE"
