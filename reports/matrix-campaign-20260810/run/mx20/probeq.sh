#!/bin/bash
# The two never-built slate classes, built and read at their named sites.
#
# Neither is a ledger ARCH row and both say why in their own receipt: CBT's
# installed tensor is dense, so the loader counts the parent's parameter count
# and its 4.266x is a STORAGE ratio; F-LANDMARK has no config that expresses it
# at all.  The frozen review is the same review -- same protocol digest, same
# prompt pack, same sixteen cases, same cap -- run through `mx20_probe_review.py`
# because `run_lfm2_composition_frozen_review.py` reads a sealed ARCH receipt and
# there is no honest way to give it one.
set -u
. /home/cereal/nf-mx20/env.sh
GPU="${1:?gpu}"; WHICH="${2:?cbt|landmark}"; LAYER="${3:-12}"
CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU -> $CAP" >&2; exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $W/probes $W/logs

case $WHICH in
  cbt)
    $PY $MX20/mx20_probe_cbt.py --layer $LAYER --device cuda:0 \
      --child $W/probes/CBT-L$LAYER-child \
      --receipt $W/probes/CBT-L$LAYER-receipt.json > $W/logs/probe-cbt-L$LAYER.log 2>&1 || {
        echo "CBT build rc=$?"; tail -15 $W/logs/probe-cbt-L$LAYER.log; exit 4; }
    $PY $MX20/mx20_probe_review.py --probe cbt --device cuda:0 \
      --require-compute-architecture sm86 \
      --child $W/probes/CBT-L$LAYER-child \
      --receipt $W/probes/CBT-L$LAYER-receipt.json \
      --out $W/probes/CBT-L$LAYER-dossier.json >> $W/logs/probe-cbt-L$LAYER.log 2>&1
    echo "cbt rc=$? :: $(grep -h 'held-out cos' $W/logs/probe-cbt-L$LAYER.log | tail -1)"
    ;;
  landmark)
    $PY $MX20/mx20_probe_landmark.py --layer $LAYER --device cuda:0 \
      --state $W/probes/LANDMARK-L$LAYER-state \
      --receipt $W/probes/LANDMARK-L$LAYER-receipt.json > $W/logs/probe-landmark-L$LAYER.log 2>&1 || {
        echo "LANDMARK build rc=$?"; tail -15 $W/logs/probe-landmark-L$LAYER.log; exit 4; }
    $PY $MX20/mx20_probe_review.py --probe landmark --device cuda:0 \
      --require-compute-architecture sm86 \
      --landmark-state $W/probes/LANDMARK-L$LAYER-state \
      --receipt $W/probes/LANDMARK-L$LAYER-receipt.json \
      --out $W/probes/LANDMARK-L$LAYER-dossier.json >> $W/logs/probe-landmark-L$LAYER.log 2>&1
    echo "landmark rc=$? :: $(grep -h 'out_proj re-solved' $W/logs/probe-landmark-L$LAYER.log | tail -1)"
    ;;
  *) echo "unknown probe $WHICH" >&2; exit 2 ;;
esac
echo "PROBEQ $WHICH DONE"
