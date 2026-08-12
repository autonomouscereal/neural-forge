#!/bin/bash
# One wave-10 (MX-N) composed child: solve every component in the recipe, install
# them in topological order, seal, and persist the composed tensors the frozen
# review reconstitutes the child from.
#
#   $1  cell id     e.g. COMP.MA.MXN1
#   $2  mode        M-A | M-B
#   $3  recipe file
#   $4  gpu index   0 | 4  (both SM86 RTX 3090)
#
# Unchanged from waves 4-9, on purpose:
# * GPU pinned by PCI bus id in the shell AND re-checked by
#   --require-compute-architecture inside the process that seals the receipt.
# * Each run writes its own ledger shard; one serialized appender merges.
# * A cell id that is already sealed writes into a subdirectory; no sealed
#   receipt is ever opened for writing.
# * Every heavy artifact lands under /mnt/ai_data; the root filesystem has 6.6 GB
#   free and a wave-10 child persists a 268 MB embedding tensor.
# * The knob file is byte-identical to waves 7, 8 and 9's (e355e4e7bb8a3058...).
#   Wave 10 compares its token-interface rungs directly against wave-9 rows and
#   its body rungs against wave-8 rows; a knob change would turn every one of
#   those comparisons into a comparison between two different experiments.
#
# New in wave 10:
# * A NEW DEPLOYMENT, neural-forge-matrix-9acaf38f, because wave 10 changes
#   source: T5 at C01 grows the sparse-OMP tail that A9/A11 actually used, and
#   its storage counts the atom ids the A9-era accounting left out.  The archive
#   was cut with `git archive -c core.autocrlf=false`, so this deployment needs
#   no hand-restored assets at all -- all 815 files, python and not, are
#   byte-identical to the git blobs of its own commit, including the frozen
#   review protocol (e5e37768...), its seal (7b859172...) and the prompt pack
#   (0ee6bf14...).  Verified on extraction, before any child was generated.
# * campaign wave 10, and a wave10 evidence-cache root whose already-built site
#   caches are symlinks into wave 9's, so a reused site costs nothing.
# * The token route's heavy solve is pinned to the same sm86 card the receipt is
#   sealed on, by the recipe's `token_route_solve_device` knob.  The harness
#   hands C01 a CPU table; a 65,536-row float64 OMP on that CPU is roughly an
#   order of magnitude slower, and leaving the device implicit would make the
#   fold's numerics depend on what happened to be plugged in.
set -u
CELL="${1:?cell id required}"; MODE="${2:?mode required}"
RECIPE="${3:?recipe file required}"; GPU="${4:?gpu index required}"

D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-9acaf38f
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
M=/mnt/ai_data/neural-forge-matrix
W=/home/cereal/nf-mxn
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=9acaf38f
WAVE=10

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

OUT=$M/wave10/cells/$CELL
[ -f "$OUT/cell-receipt.json" ] && OUT=$OUT/rerun-$(date -u +%H%M%S)
mkdir -p "$OUT" $W/logs

EXTRA=""
if [ "$MODE" = "M-B" ]; then
  EXTRA="--conditioned-capture-root $M/wave10/evidence-conditioned \
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
  --evidence-cache-root $M/wave10/evidence-cache \
  --knobs $W/mxn-knobs.json --hard-cap-seconds 28800 --target-seconds 10800 \
  $EXTRA \
  > $W/logs/comp-$CELL.log 2>&1
rc=$?
echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/comp-$CELL.log)"
echo "$OUT"
exit $rc
