#!/bin/bash
# Wave-1 matrix cells 4-30, one batch per GPU.
#
#   $1  batch letter (a|b) -- selects the cell list, the GPU and the LEDGER SHARD
#
# Ledger races: matrix_cell_receipt.append_ledger opens the ledger "ab" and then
# issues TWO write() calls (the row, then the newline).  O_APPEND makes each of
# those atomic on its own, but not the pair, so two concurrent runners can
# interleave a newline into another runner's row.  Every batch therefore writes
# its own shard and the shards are merged into the authoritative ledger, in
# deterministic cell order, by a single serialized appender after both batches
# finish.  Nothing in this script touches /mnt/ai_data/neural-forge-matrix/ledger.jsonl.
#
# GPU selection: nvidia-smi index 4 is the second RTX 3090, but CUDA's default
# device ordering is FASTEST_FIRST and maps index 4 onto a Tesla P40 (sm_61)
# that this torch build cannot run.  CUDA_DEVICE_ORDER=PCI_BUS_ID makes the
# indices agree with nvidia-smi, and the preflight below refuses to start a
# batch on anything but the SM86 card wave-1 cells 1-3 were measured on.
set -u

BATCH="${1:?batch letter required}"
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
W=/home/cereal/nf-mxe
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=c59be064

case "$BATCH" in
  a)
    GPU=0
    # cell-id | evidence-cache key (site+component: the endpoint set is a
    # function of the component alone, and the runner's cache binding refuses a
    # payload whose endpoint set differs) | knob file
    CELLS=(
      "C05.T8.M16.L00|L00-C05|base"
      "C13.T5.M16.L00|L00-C13|base"
      "C13.T1.M16.L00|L00-C13|base"
      "C13.T8.M16.L00|L00-C13|base"
      "C11.T5.M16.L00|L00-C11|base"
      "C12.T5.M16.L00|L00-C12|base"
      "C03.T5.M16.L00|L00-C03|base"
      "C04.T1.M16.L00|L00-C04|base"
      "C14.T1.NA.L00|L00-C14|base"
      "C13.T5.M16.L05|L05-C13|base"
      "C13.T1.M16.L05|L05-C13|base"
      "C05.T1.M16.L05|L05-C05|base"
    )
    ;;
  b)
    GPU=4
    CELLS=(
      "C09.T5.M16.L02|L02-C09|base"
      "C09.T1.M16.L02|L02-C09|base"
      "C09.T8.M16.L02|L02-C09|base"
      "C09.T4.M16.L02|L02-C09|t4rank31"
      "C08.T5.M16.L02|L02-C08|base"
      "C08.T1.M16.L02|L02-C08|base"
      "C06.T1.M16.L02|L02-C06|base"
      "C06.T7.M16.L02|L02-C06|base"
      "C07.T1.M16.L02|L02-C07|base"
      "C10.T1.M16.L02|L02-C10|base"
      "C06.T7.M16.L08|L08-C06|base"
      "C05.T1.M16.L13|L13-C05|base"
      "C05.T4.M16.L13|L13-C05|t4rank31"
      "C13.T5.M16.L13|L13-C13|base"
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
mkdir -p $W/logs $W/evidence-cache
echo "batch $BATCH on GPU $GPU, ${#CELLS[@]} cells, shard $SHARD"

for entry in "${CELLS[@]}"; do
  CELL="${entry%%|*}"
  rest="${entry#*|}"
  CACHE="${rest%%|*}"
  KNOBSEL="${rest##*|}"
  case "$KNOBSEL" in
    base)      KNOBS=$W/mxe-knobs.json ;;
    t4rank31)  KNOBS=$W/mxe-knobs-t4-rank31.json ;;
    *) echo "unknown knob selector $KNOBSEL" >&2; exit 2 ;;
  esac
  echo "=== $(date -u +%H:%M:%S) start $CELL (cache $CACHE, knobs $KNOBSEL)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $PY $D/experiments/run_lfm2_matrix_cell.py \
    --cell-id "$CELL" --mode cell --campaign-wave 1 --repo-head $REPO_HEAD \
    --out /mnt/ai_data/neural-forge-matrix/cells/$CELL \
    --ledger $SHARD \
    --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --control-artifact /mnt/ai_data/neural-forge-matrix/cells/CTRL.copy_exact \
    --device cuda:0 --dtype float32 \
    --parent-trace-cache $EV/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff \
    --evidence-split-manifest $EV/evidence-manifest.json \
    --evidence-cache $W/evidence-cache/$CACHE \
    --knobs $KNOBS \
    > $W/logs/cell-$CELL.log 2>&1
  rc=$?
  echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/cell-$CELL.log)"
done

echo "BATCH $BATCH COMPLETE, $(wc -l < $SHARD) shard rows"
