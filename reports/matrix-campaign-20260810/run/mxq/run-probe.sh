#!/bin/bash
# Wave 13 lane shopping.  $1 out-file basename, $2 gpu, then lane arguments.
set -u
NAME="${1:?name}"; GPU="${2:?gpu}"; shift 2
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
M=/mnt/ai_data/neural-forge-matrix
W=/home/cereal/nf-mxq
PY=/home/cereal/model-surgery-lab/venv/bin/python
CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c "import torch; print(\"%d.%d\" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))" 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU -> $CAP" >&2; exit 3
fi
mkdir -p $W/logs $M/wave13/probes
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
$PY $W/mxq_shop_probe.py \
  --out $M/wave13/probes/$NAME.json \
  --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
  --evidence-split-manifest $EV/evidence-manifest.json \
  --evidence-cache-root $M/wave13/evidence-cache \
  --knobs $W/mxq-knobs.json \
  --workspace $M/wave13/probes/workspace \
  --device cuda:0 --require-compute-architecture sm86 \
  "$@" > $W/logs/probe-$NAME.log 2>&1
rc=$?
echo "probe $NAME rc=$rc"
tail -30 $W/logs/probe-$NAME.log
exit $rc
