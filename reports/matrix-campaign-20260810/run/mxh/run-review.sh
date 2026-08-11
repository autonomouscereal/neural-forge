#!/bin/bash
# Frozen-protocol review surface for one sealed composed child.
#   $1 cell id   $2 cell directory   $3 gpu index
set -u
CELL="${1:?cell id}"; DIR="${2:?cell directory}"; GPU="${3:?gpu}"
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
W=/home/cereal/nf-mxh
PY=/home/cereal/model-surgery-lab/venv/bin/python
CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c "import torch; print(\"%d.%d\" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))" 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU -> $CAP" >&2; exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
$PY $D/experiments/run_lfm2_composition_frozen_review.py \
  --receipt $DIR/cell-receipt.json \
  --composed-tensors $DIR/composed-child-tensors.pt \
  --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --protocol-seal $D/reports/r12-certified-20260804/PROTOCOL_SEAL.json \
  --prompts-json $W/lfm2_semantic_whole_response_prompts_v1.json \
  --out $DIR/frozen-review-dossier.json \
  --device cuda:0 --require-compute-architecture sm86 \
  --maximum-new-tokens 1024 \
  > $W/logs/review-$CELL.log 2>&1
echo "review $CELL rc=$? :: $(tail -1 $W/logs/review-$CELL.log)"
