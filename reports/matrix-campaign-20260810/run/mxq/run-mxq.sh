#!/bin/bash
# One wave-12 (MX-P) composed child: solve every component in the recipe, install
# them in topological order, solve any ADDENDUM 10 corrective routes the recipe
# asks for, seal, and persist the tensors the frozen review reconstitutes the
# child from -- both the folded ones and, when there are any, the corrective
# routes, which are hooks and not state_dict entries.
#
#   $1  cell id     e.g. COMP.MA.MXP1
#   $2  mode        M-A | M-B | M-C
#   $3  recipe file
#   $4  gpu index   0 | 4  (both SM86 RTX 3090)
#
# Unchanged from waves 4-11, on purpose:
# * GPU pinned by PCI bus id in the shell AND re-checked by
#   --require-compute-architecture inside the process that seals the receipt.
# * Each run writes its own ledger shard; one serialized appender merges.
# * A cell id that is already sealed writes into a subdirectory; no sealed
#   receipt is ever opened for writing.
# * Every heavy artifact lands under /mnt/ai_data; the root filesystem has under
#   5 GB free and a child that folds C01 persists a 268 MB embedding tensor.
# * The knob file is byte-identical to waves 7-11's (e355e4e7bb8a3058...).  Wave
#   12 compares its rungs directly against wave 11's MXO3 and MXO4 rows; a knob
#   change would turn every one of those comparisons into a comparison between
#   two different experiments.
#
# New in wave 12:
# * A NEW DEPLOYMENT, neural-forge-matrix-63e95cd8, because wave 12 changes
#   source: ADDENDUM 10's corrective-component class (src/neural_forge/
#   matrix_corrective_routes.py), the composition runner's corrective pass, the
#   net-storage term that charges it, and the frozen review's refusal to
#   reconstitute a corrected child without its routes.  Cut with
#   `git archive -c core.autocrlf=false`; the review surface is verified on
#   extraction, before any child is generated -- protocol e5e37768..., seal
#   7b859172... in PROTOCOL_SEAL.json, prompt pack 0ee6bf14...
# * campaign wave 12, and a wave13 evidence-cache root whose already-built site
#   caches are symlinks into wave 11's, so a reused site costs nothing.
# * CORRECTIONS=1 turns on the ADDENDUM 10 post-pass for a recipe that carries a
#   `corrections` block.  It needs the conditioned-capture machinery in ANY mode,
#   because a route is solved against a capture taken through the composed child;
#   the captures land in tmpfs and are deleted as their rows are read, exactly as
#   wave 11's ALS captures were, and for the same reason (8 GB each against a
#   share that writes at ~20 MB/s).

set -u
CELL="${1:?cell id required}"; MODE="${2:?mode required}"
RECIPE="${3:?recipe file required}"; GPU="${4:?gpu index required}"

D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
M=/mnt/ai_data/neural-forge-matrix
W=/home/cereal/nf-mxq
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=63e95cd8
WAVE=13

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

OUT=$M/wave13/cells/$CELL
[ -f "$OUT/cell-receipt.json" ] && OUT=$OUT/rerun-$(date -u +%H%M%S)
mkdir -p "$OUT" $W/logs

EXTRA=""
DONOR=/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/mapped_donor_calibration.json
if [ "$MODE" = "M-B" ]; then
  EXTRA="--conditioned-capture-root $M/wave13/evidence-conditioned --donor-calibration $DONOR"
fi
if [ "${CORRECTIONS:-0}" = "1" ]; then
  # ADDENDUM 10 M2.  A corrective route is solved from a capture taken THROUGH
  # the composed child, so the capture machinery is required even in M-A, where
  # every fold was solved against parent evidence.  One capture per route, in
  # depth-ascending order, because installing a route at L moves every row every
  # deeper block reads.
  mkdir -p /dev/shm/nf-mxq-captures
  EXTRA="$EXTRA --conditioned-capture-root /dev/shm/nf-mxq-captures --donor-calibration $DONOR --correction-ridge-relative ${CORRECTION_RIDGE:-1e-10}"
fi
if [ "$MODE" = "M-C" ]; then
  # ADDENDUM 9's ALS.  Two deviations from the M-B wiring, both deliberate:
  #
  # * The conditioned captures land in /dev/shm, not on the share.  One capture
  #   is 8 GB and an ALS run takes one per rung per sweep; the share writes at
  #   ~20 MB/s, which would make the capture cost the run.  They are pure
  #   re-derivable evidence -- the receipt seals every capture's cache key,
  #   split manifest digest, collector implementation digest and child state
  #   digest -- and the runner deletes each one as soon as its rows are read,
  #   so nothing heavy accumulates in either place.  The host has 126 GB of
  #   tmpfs and 230 GB of free RAM.
  # * --als-verify-capture-licence is ON for the first child of the wave: it
  #   spends one extra capture to prove the reuse rule serves bit-identical
  #   rows.  ALS_VERIFY=0 turns it off once the wave has its proof.
  mkdir -p /dev/shm/nf-mxq-captures
  EXTRA="--conditioned-capture-root /dev/shm/nf-mxq-captures --donor-calibration $DONOR --als-max-sweeps ${ALS_SWEEPS:-4} --als-improvement-floor ${ALS_FLOOR:-0.05} --als-objective ${ALS_OBJECTIVE:-parent-target}"
  [ "${ALS_VERIFY:-1}" = "1" ] && EXTRA="$EXTRA --als-verify-capture-licence"
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
  --evidence-cache-root $M/wave13/evidence-cache \
  --knobs $W/mxq-knobs.json --hard-cap-seconds 28800 --target-seconds 10800 \
  $EXTRA \
  > $W/logs/comp-$CELL.log 2>&1
rc=$?
echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/comp-$CELL.log)"
echo "$OUT"
exit $rc
