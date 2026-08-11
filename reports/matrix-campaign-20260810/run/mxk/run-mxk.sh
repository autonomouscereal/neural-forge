#!/bin/bash
# One wave-7 (MX-K) composed child: solve every component in the recipe, install
# them in topological order, seal, and persist the composed tensors the frozen
# review reconstitutes the child from.
#
#   $1  cell id     e.g. COMP.MA.MXK1
#   $2  mode        M-A | M-B
#   $3  recipe file
#   $4  gpu index   0 | 4  (both SM86 RTX 3090)
#
# Unchanged from waves 4-6, on purpose:
# * GPU pinned by PCI bus id in the shell AND re-checked by
#   --require-compute-architecture inside the process that seals the receipt.
# * Each run writes its own ledger shard; one serialized appender merges.
# * A cell id that is already sealed writes into a subdirectory; no sealed
#   receipt is ever opened for writing.
# * Every heavy artifact lands under /mnt/ai_data; the root filesystem has 7 GB
#   free and a wave-7 site cache is 250-320 MB each.
#
# New in wave 7:
# * the deployment is THIS wave's own commit, 17377c16, because the wave changes
#   the receipt schema and a wave-6 deployment cannot seal a v2 site.  Its 605
#   python modules were verified byte-identical to the git blobs of that commit
#   server-side, and it differs from the wave-6 surface in exactly the two files
#   the commit touched.
# * the evidence cache root is the wave7 one, whose 21 already-built site caches
#   are symlinks to the wave-5/6 ones, so a reused site costs nothing and a NEW
#   site's payload is written to the 1.8 TB share.
set -u
CELL="${1:?cell id required}"; MODE="${2:?mode required}"
RECIPE="${3:?recipe file required}"; GPU="${4:?gpu index required}"

D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
M=/mnt/ai_data/neural-forge-matrix
W=/home/cereal/nf-mxk
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=17377c16
WAVE=7

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

OUT=$M/wave7/cells/$CELL
[ -f "$OUT/cell-receipt.json" ] && OUT=$OUT/rerun-$(date -u +%H%M%S)
mkdir -p "$OUT" $W/logs

EXTRA=""
if [ "$MODE" = "M-B" ]; then
  EXTRA="--conditioned-capture-root $M/wave7/evidence-conditioned \
         --donor-calibration /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/mapped_donor_calibration.json"
fi

echo "=== $(date -u +%H:%M:%S) start $CELL mode=$MODE recipe=$(basename $RECIPE) gpu=$GPU -> $OUT"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
$PY $D/experiments/run_lfm2_matrix_composition.py \
  --cell-id "$CELL" --mode "$MODE" --campaign-wave $WAVE --repo-head $REPO_HEAD \
  --recipe "$RECIPE" \
  --out "$OUT" \
  --ledger $W/ledger-$CELL.jsonl \
  --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --control-artifact $M/cells/CTRL.copy_exact \
  --device cuda:0 --dtype float32 \
  --require-compute-architecture sm86 \
  --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
  --evidence-split-manifest $EV/evidence-manifest.json \
  --evidence-cache-root $M/wave7/evidence-cache \
  --knobs $W/mxk-knobs.json --hard-cap-seconds 28800 --target-seconds 10800 \
  $EXTRA \
  > $W/logs/comp-$CELL.log 2>&1
rc=$?
echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/comp-$CELL.log)"
echo "$OUT"
exit $rc
