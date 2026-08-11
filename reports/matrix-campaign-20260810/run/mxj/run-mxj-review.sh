#!/bin/bash
# The ADDENDUM 6 frozen review surface for one sealed wave-6 composed child.
#
#   $1  cell id        (for the log name)
#   $2  cell directory (holds cell-receipt.json and composed-child-tensors.pt)
#   $3  gpu index
#
# Pinned before a token is generated, by the review script itself and not by
# this wrapper: the protocol re-seals to e5e37768..., the prompt pack digests to
# the seal's own byte-form 0ee6bf14..., and the child is reconstituted from the
# parent plus the sealed composed tensors with every folded tensor's digest
# checked against the composition receipt.  The parent is decoded on the same
# sixteen prompts IN THE SAME PROCESS, because ADDENDUM 6 scores against the
# parent's own 12/15 and a control from another process is not the same control.
set -u
CELL="${1:?cell id}"; DIR="${2:?cell directory}"; GPU="${3:?gpu}"
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-eb32153e
W=/home/cereal/nf-mxj
PY=/home/cereal/model-surgery-lab/venv/bin/python
CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU -> $CAP" >&2; exit 3
fi
mkdir -p $W/logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
$PY $D/experiments/run_lfm2_composition_frozen_review.py \
  --receipt $DIR/cell-receipt.json \
  --composed-tensors $DIR/composed-child-tensors.pt \
  --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --protocol-seal $D/reports/r12-certified-20260804/PROTOCOL_SEAL.json \
  --prompts-json $D/research/lfm2_semantic_whole_response_prompts_v1.json \
  --out $DIR/frozen-review-dossier.json \
  --device cuda:0 --require-compute-architecture sm86 \
  --maximum-new-tokens 1024 \
  > $W/logs/review-$CELL.log 2>&1
rc=$?
echo "review $CELL rc=$rc :: $(tail -1 $W/logs/review-$CELL.log)"
exit $rc
