#!/bin/bash
# Wave-4 (MX-H) composition cells.  ADDENDUM 4 modes on one composed child each.
#
#   $1  cell id     COMP.MA.RBEH | COMP.MA.RSTO | COMP.MB.RBEH | ...
#   $2  mode        M-A | M-B
#   $3  recipe      RBEH | RSTO
#   $4  gpu index   0 | 4   (nvidia-smi index; both are SM86 RTX 3090s)
#
# The three disciplines waves 1-3 established, unchanged:
# * GPU.  CUDA default ordering is FASTEST_FIRST and maps index 4 onto a P40
#   on this host; the shell preflight AND --require-compute-architecture sm86
#   both stay, so the refusal lives inside the process that seals the receipt.
# * Ledger.  Each run writes its own shard; a single serialized appender merges.
# * Re-runs.  A cell id that already has a receipt writes into a subdirectory;
#   no sealed receipt is ever opened for writing.
set -u
CELL="${1:?cell id required}"; MODE="${2:?mode required}"
RECIPE="${3:?recipe variant required}"; GPU="${4:?gpu index required}"

D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
W=/home/cereal/nf-mxh
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=8d59c4d2
WAVE=4

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c "import torch; print(\"%d.%d\" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))" 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

OUT=/mnt/ai_data/neural-forge-matrix/cells/$CELL
[ -f "$OUT/cell-receipt.json" ] && OUT=$OUT/rerun-$(date -u +%H%M%S)
mkdir -p $W/logs

EXTRA=""
if [ "$MODE" = "M-B" ]; then
  EXTRA="--conditioned-capture-root /mnt/ai_data/neural-forge-matrix/evidence-conditioned/shared \
         --donor-calibration /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/mapped_donor_calibration.json"
fi

echo "=== $(date -u +%H:%M:%S) start $CELL mode=$MODE recipe=$RECIPE gpu=$GPU out=$OUT"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
$PY $D/experiments/run_lfm2_matrix_composition.py \
  --cell-id "$CELL" --mode "$MODE" --campaign-wave $WAVE --repo-head $REPO_HEAD \
  --recipe $W/composition-recipe-$RECIPE.json \
  --out $OUT \
  --ledger $W/ledger-$CELL.jsonl \
  --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --control-artifact /mnt/ai_data/neural-forge-matrix/cells/CTRL.copy_exact \
  --device cuda:0 --dtype float32 \
  --require-compute-architecture sm86 \
  --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
  --evidence-split-manifest $EV/evidence-manifest.json \
  --evidence-cache-root /home/cereal/nf-mxe/evidence-cache \
  --knobs $W/mxh-knobs.json --hard-cap-seconds 21600 --target-seconds 7200 \
  $EXTRA \
  > $W/logs/comp-$CELL.log 2>&1
rc=$?
echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/comp-$CELL.log)"
exit $rc
