#!/bin/bash
# Wave-2 (MX-F) matrix cells, one batch per RTX 3090.
#
#   $1  batch letter (a|b) -- selects the cell list, the GPU and the LEDGER SHARD
#
# Same two disciplines wave 1 had to learn, both now enforced twice:
#
# * GPU.  CUDA's default device ordering is FASTEST_FIRST and maps nvidia-smi
#   index 4 onto a Tesla P40 (sm_61) this torch build cannot run.  The shell
#   preflight below is kept, and every cell additionally passes
#   --require-compute-architecture sm86, so the refusal now lives inside the
#   process that seals the receipt rather than only in this file.
# * Ledger.  append_ledger issues TWO write() calls per row, so O_APPEND makes
#   each atomic and not the pair.  Each batch writes its own shard and a single
#   serialized appender merges them afterwards.  Nothing here touches
#   /mnt/ai_data/neural-forge-matrix/ledger.jsonl.
set -u

BATCH="${1:?batch letter required}"
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
W=/home/cereal/nf-mxf
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=347e8d93
WAVE=2

case "$BATCH" in
  a)
    GPU=0
    # cell-id | evidence-cache key (site+component) | knob file
    CELLS=(
      "C05.T1.M8.L05|L05-C05|base"
      "C05.T5.M8.L05|L05-C05|base"
      "C05.T1.M2.L05|L05-C05|base"
      "C05.T2.M16.L05|L05-C05|base"
      "C05.T3.M16.L05|L05-C05|base"
      "C13.T2.M16.L05|L05-C13|base"
      "C13.T3.M16.L05|L05-C13|base"
      "C13.T5.M16.L05|L05-C13|t5k16"
      "C13.T1.M16.L13|L13-C13|base"
      "C09.T1.M8.L02|L02-C09|base"
      "C09.T1.M4.L02|L02-C09|base"
      "C09.T2.M16.L02|L02-C09|base"
      "C08.T1.M8.L02|L02-C08|base"
      "C08.T1.M4.L02|L02-C08|base"
      "C03.T5.M16.L00|L00-C03|base"
      "C04.T1.M16.L00|L00-C04|base"
      "C14.T1.NA.L00|L00-C14|base"
      "C10.T1.M16.L02|L02-C10|base"
    )
    ;;
  b)
    GPU=4
    CELLS=(
      "C06.T1.M8.L02|L02-C06|base"
      "C06.T1.M4.L02|L02-C06|base"
      "C06.T1.M2.L02|L02-C06|base"
      "C06.T7.M8.L02|L02-C06|base"
      "C06.T7.M4.L02|L02-C06|base"
      "C07.T1.M8.L02|L02-C07|base"
      "C07.T1.M4.L02|L02-C07|base"
      "C07.T1.M2.L02|L02-C07|base"
      "C06.T1.M16.L08|L08-C06|base"
      "C06.T1.M16.L12|L12-C06|base"
      "C06.T7.M16.L12|L12-C06|base"
      "C07.T1.M16.L08|L08-C07|base"
      "C07.T1.M16.L12|L12-C07|base"
      "C09.T1.M16.L08|L08-C09|base"
    )
    ;;
  *) echo "unknown batch $BATCH" >&2; exit 2 ;;
esac

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
echo "preflight ok: GPU $GPU -> $CAP"

SHARD=$W/ledger-$BATCH.jsonl
mkdir -p $W/logs
echo "batch $BATCH on GPU $GPU, ${#CELLS[@]} cells, shard $SHARD"

for entry in "${CELLS[@]}"; do
  CELL="${entry%%|*}"
  rest="${entry#*|}"
  CACHE="${rest%%|*}"
  KNOBSEL="${rest##*|}"
  case "$KNOBSEL" in
    base)   KNOBS=$W/mxf-knobs.json ;;
    t5k16)  KNOBS=$W/mxf-knobs-t5-k16.json ;;
    *) echo "unknown knob selector $KNOBSEL" >&2; exit 2 ;;
  esac
  # Five wave-2 cells re-run a cell id wave 1 already sealed (the four void
  # cells and the C13.T5 rescue probe).  The ledger keeps both instances by
  # design -- "a re-run of a cell gets a new cell instance id, never an
  # overwrite" -- but the *files* would collide, so a repeat writes into its
  # own wave-2 subdirectory and wave 1's sealed artifacts are never opened
  # for writing.
  OUT=/mnt/ai_data/neural-forge-matrix/cells/$CELL
  if [ -f "$OUT/cell-receipt.json" ]; then
    OUT=$OUT/wave2
    echo "    (re-run of a sealed cell; wave-1 artifacts untouched -> $OUT)"
  fi
  echo "=== $(date -u +%H:%M:%S) start $CELL (cache $CACHE, knobs $KNOBSEL)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $PY $D/experiments/run_lfm2_matrix_cell.py \
    --cell-id "$CELL" --mode cell --campaign-wave $WAVE --repo-head $REPO_HEAD \
    --out $OUT \
    --ledger $SHARD \
    --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --control-artifact /mnt/ai_data/neural-forge-matrix/cells/CTRL.copy_exact \
    --device cuda:0 --dtype float32 \
    --require-compute-architecture sm86 \
    --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
    --evidence-split-manifest $EV/evidence-manifest.json \
    --evidence-cache /home/cereal/nf-mxe/evidence-cache/$CACHE \
    --knobs $KNOBS \
    > $W/logs/cell-$CELL.log 2>&1
  rc=$?
  echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/cell-$CELL.log)"
done

echo "BATCH $BATCH COMPLETE, $(wc -l < $SHARD) shard rows"
