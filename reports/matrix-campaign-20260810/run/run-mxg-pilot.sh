#!/bin/bash
# Wave-3 pilot: prove the three new kernels and the X4 path before the wave.
# Writes into its own directory with its own ledger; that ledger is NEVER
# merged into the campaign authority (same discipline as MX-F's 5-cell pilot).
set -u
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
W=/home/cereal/nf-mxg
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=0

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

# cell | evidence-cache | knobs | x4-width (- for none)
CELLS=(
  "C04.T1.M16.L00|L00-C04|c04|-"
  "C10.T1.M16.L02|L02-C09|base|-"
  "C05.T9.M8.L05|L05-C05|t9|-"
  "C06.T1.X4.L02|L02-C06|base|768"
)

mkdir -p $W/logs $W/pilot
for entry in "${CELLS[@]}"; do
  CELL="${entry%%|*}"; rest="${entry#*|}"
  CACHE="${rest%%|*}"; rest="${rest#*|}"
  KNOBSEL="${rest%%|*}"; X4="${rest##*|}"
  case "$KNOBSEL" in
    base) KNOBS=$W/mxg-knobs.json ;;
    t9)   KNOBS=$W/mxg-knobs-t9.json ;;
    c04)  KNOBS=$W/mxg-knobs-c04.json ;;
    *) echo "unknown knob selector $KNOBSEL" >&2; exit 2 ;;
  esac
  X4ARG=""
  [ "$X4" != "-" ] && X4ARG="--x4-width $X4"
  echo "=== $(date -u +%H:%M:%S) pilot start $CELL (cache $CACHE, knobs $KNOBSEL, x4 $X4)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $PY $D/experiments/run_lfm2_matrix_cell.py \
    --cell-id "$CELL" --mode cell --campaign-wave 3 --repo-head 9165e45c \
    --out $W/pilot/$CELL \
    --ledger $W/pilot/ledger-pilot.jsonl \
    --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --control-artifact /mnt/ai_data/neural-forge-matrix/cells/CTRL.copy_exact \
    --device cuda:0 --dtype float32 \
    --require-compute-architecture sm86 \
    --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
    --evidence-split-manifest $EV/evidence-manifest.json \
    --evidence-cache /home/cereal/nf-mxe/evidence-cache/$CACHE \
    $X4ARG \
    --knobs $KNOBS \
    > $W/logs/pilot-$CELL.log 2>&1
  rc=$?
  echo "=== $(date -u +%H:%M:%S) pilot done  $CELL rc=$rc :: $(tail -2 $W/logs/pilot-$CELL.log | tr '\n' ' ')"
done
echo "PILOT COMPLETE, $(wc -l < $W/pilot/ledger-pilot.jsonl 2>/dev/null || echo 0) rows"
