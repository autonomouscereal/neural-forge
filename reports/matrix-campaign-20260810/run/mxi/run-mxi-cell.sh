#!/bin/bash
# One wave-5 (MX-I) matrix cell: solve, install, seal, and persist the folded
# tensors the frozen review will reconstitute the child from.
#
#   $1  cell id            e.g. C13.T5.S4.L05
#   $2  evidence cache tag e.g. L05-C13
#   $3  gpu index          0 or 4 (both SM86 RTX 3090)
#   $4  stored width or -  the S-level width (see --plan-storage-width) or the
#                          exact-4x layout width for an X4 cell
#   $5  output subdir      e.g. wave5   (never overwrites a sealed receipt)
#
# The three disciplines waves 1-3 had to learn are unchanged and still enforced:
# the GPU is pinned by PCI bus id AND re-checked inside the process that seals
# the receipt (--require-compute-architecture sm86); each batch writes its own
# ledger shard because append_ledger issues two writes per row and only the
# individual write is atomic; and a cell id that is already sealed writes into
# its own subdirectory rather than opening a sealed receipt for writing.
#
# New in wave 5: --save-installed-tensors.  ADDENDUM 6 grades the WHOLE model
# with this one component folded, and the frozen review runs in a separate
# process, so the child has to be reconstitutable from the receipt -- which
# means the folded tensors and their digests are part of the cell's output, not
# a side effect that dies with the process.
set -u

CELL="${1:?cell id}"; CACHE="${2:?evidence cache tag}"; GPU="${3:?gpu}"
WIDTH="${4:--}"; SUBDIR="${5:-wave5}"

D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-eb32153e
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
W=/home/cereal/nf-mxi
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=eb32153e
WAVE=5

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi

OUT=/mnt/ai_data/neural-forge-matrix/cells/$CELL
[ -f "$OUT/cell-receipt.json" ] && OUT=$OUT/$SUBDIR
mkdir -p "$OUT" "$W/logs"

WARG=""
[ "$WIDTH" != "-" ] && WARG="--x4-width $WIDTH"

echo "=== $(date -u +%H:%M:%S) start $CELL (cache $CACHE, gpu $GPU, width $WIDTH) -> $OUT"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
$PY $D/experiments/run_lfm2_matrix_cell.py \
  --cell-id "$CELL" --mode cell --campaign-wave $WAVE --repo-head $REPO_HEAD \
  --out "$OUT" \
  --ledger $W/ledger-gpu$GPU.jsonl \
  --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --control-artifact /mnt/ai_data/neural-forge-matrix/cells/CTRL.copy_exact \
  --device cuda:0 --dtype float32 \
  --require-compute-architecture sm86 \
  --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
  --evidence-split-manifest $EV/evidence-manifest.json \
  --evidence-cache /home/cereal/nf-mxe/evidence-cache/$CACHE \
  --save-installed-tensors \
  $WARG \
  --knobs $W/mxi-knobs.json \
  > $W/logs/cell-$CELL.log 2>&1
rc=$?
echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/cell-$CELL.log)"
echo "$OUT"
exit $rc
