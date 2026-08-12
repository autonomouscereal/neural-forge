#!/bin/bash
# One wave-11 (MX-O) composed child: solve every component in the recipe, install
# them in topological order, seal, and persist the composed tensors the frozen
# review reconstitutes the child from.
#
#   $1  cell id     e.g. COMP.MA.MXN1
#   $2  mode        M-A | M-B
#   $3  recipe file
#   $4  gpu index   0 | 4  (both SM86 RTX 3090)
#
# Unchanged from waves 4-10, on purpose:
# * GPU pinned by PCI bus id in the shell AND re-checked by
#   --require-compute-architecture inside the process that seals the receipt.
# * Each run writes its own ledger shard; one serialized appender merges.
# * A cell id that is already sealed writes into a subdirectory; no sealed
#   receipt is ever opened for writing.
# * Every heavy artifact lands under /mnt/ai_data; the root filesystem has under
#   6 GB free and a wave-11 child persists a 268 MB embedding tensor.
# * The knob file is byte-identical to waves 7-10's (e355e4e7bb8a3058...).  Wave
#   11 compares its M-C rungs directly against the wave-10 M-A rows they
#   re-solve; a knob change would turn every one of those comparisons into a
#   comparison between two different experiments.
#
# New in wave 11:
# * A NEW DEPLOYMENT, neural-forge-matrix-fcd69d17, because wave 11 changes
#   source: ADDENDUM 9's mode M-C (the alternating joint co-solve), T8's
#   relation_target_endpoint knob, and the ('T5','C01') double-registration fix.
#   Cut with `git archive -c core.autocrlf=false`; the review surface was
#   verified on extraction, before any child was generated -- protocol
#   e5e37768..., seal e5e37768... in PROTOCOL_SEAL.json, prompt pack 0ee6bf14...
# * campaign wave 11, and a wave11 evidence-cache root whose already-built site
#   caches are symlinks into wave 10's, so a reused site costs nothing.
# * MODE M-C, whose conditioned captures land in tmpfs and are deleted as their
#   rows are read (see the M-C branch below for why that is the honest choice).

set -u
CELL="${1:?cell id required}"; MODE="${2:?mode required}"
RECIPE="${3:?recipe file required}"; GPU="${4:?gpu index required}"

D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-fcd69d17
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
M=/mnt/ai_data/neural-forge-matrix
W=/home/cereal/nf-mxo
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=fcd69d17
WAVE=11

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

OUT=$M/wave11/cells/$CELL
[ -f "$OUT/cell-receipt.json" ] && OUT=$OUT/rerun-$(date -u +%H%M%S)
mkdir -p "$OUT" $W/logs

EXTRA=""
DONOR=/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/mapped_donor_calibration.json
if [ "$MODE" = "M-B" ]; then
  EXTRA="--conditioned-capture-root $M/wave11/evidence-conditioned --donor-calibration $DONOR"
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
  mkdir -p /dev/shm/nf-mxo-captures
  EXTRA="--conditioned-capture-root /dev/shm/nf-mxo-captures --donor-calibration $DONOR --als-max-sweeps ${ALS_SWEEPS:-4} --als-improvement-floor ${ALS_FLOOR:-0.05} --als-objective ${ALS_OBJECTIVE:-parent-target}"
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
  --evidence-cache-root $M/wave11/evidence-cache \
  --knobs $W/mxo-knobs.json --hard-cap-seconds 28800 --target-seconds 10800 \
  $EXTRA \
  > $W/logs/comp-$CELL.log 2>&1
rc=$?
echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/comp-$CELL.log)"
echo "$OUT"
exit $rc
