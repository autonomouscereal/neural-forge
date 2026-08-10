#!/bin/bash
# Wave-3 (MX-G) matrix cells, one batch per RTX 3090.
#
#   $1  batch letter (a|b) -- selects the cell list, the GPU and the LEDGER SHARD
#
# The three disciplines waves 1-2 had to learn, all still enforced:
#
# * GPU.  CUDA's default device ordering is FASTEST_FIRST and maps nvidia-smi
#   index 4 onto a Tesla P40 (sm_61) this torch build cannot run.  The shell
#   preflight below is kept, and every cell additionally passes
#   --require-compute-architecture sm86, so the refusal lives inside the
#   process that seals the receipt as well as in this file.
# * Ledger.  append_ledger issues TWO write() calls per row, so O_APPEND makes
#   each atomic and not the pair.  Each batch writes its own shard and a single
#   serialized appender merges them afterwards.  Nothing here touches
#   /mnt/ai_data/neural-forge-matrix/ledger.jsonl.
# * Re-runs.  A cell id waves 1-2 already sealed writes into its own wave3
#   subdirectory; no sealed receipt is ever opened for writing.
#
# New in wave 3: a per-cell --x4-width.  X4 is "the width the canonical exact-4x
# layout assigns this component" (design 1.1) and the registry refuses to guess
# it.  The widths below are read off the sealed exact-4x materialization
# `reports/exact4x-heterogeneous-blank-20260729/physical-materialization.json`
# (57,420,144 parameters, THE canonical budget): attention geometry in causal
# order gives L2 and L12 both (12 heads, 6 kv, 64 dim), so a query projection
# keeps 12*64=768 and a key/value projection 6*64=384; conv internal widths in
# causal order give L5 = 704.
set -u

BATCH="${1:?batch letter required}"
D=/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952
EV=/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence
W=/home/cereal/nf-mxg
PY=/home/cereal/model-surgery-lab/venv/bin/python
REPO_HEAD=9165e45c
WAVE=3

case "$BATCH" in
  a)
    GPU=0
    # cell-id | evidence-cache (site+component) | knob file | x4 width or -
    CELLS=(
      "C05.T3.M16.L00|L00-C05|base|-"
      "C05.T3.M16.L13|L13-C05|base|-"
      "C05.T3.M8.L05|L05-C05|base|-"
      "C05.T3.M4.L05|L05-C05|base|-"
      "C05.T3.M2.L05|L05-C05|base|-"
      "C05.T3.X4.L05|L05-C05|base|704"
      "C05.T1.X4.L05|L05-C05|base|704"
      "C05.T9.M8.L05|L05-C05|t9|-"
      "C05.T9.M4.L05|L05-C05|t9t3|-"
      "C13.T3.M8.L05|L05-C13|base|-"
      "C13.T3.M4.L05|L05-C13|base|-"
      "C13.T3.M16.L13|L13-C13|base|-"
      "C04.T1.M16.L00|L00-C04|c04|-"
      "C04.T1.M2.L00|L00-C04|c04|-"
    )
    ;;
  b)
    GPU=4
    CELLS=(
      "C06.T1.X4.L02|L02-C06|base|768"
      "C06.T1.M8.L08|L08-C06|base|-"
      "C06.T1.M8.L12|L12-C06|base|-"
      "C06.T1.X4.L12|L12-C06|base|768"
      "C06.T5.M16.L12|L12-C06|base|-"
      "C07.T1.M8.L08|L08-C07|base|-"
      "C07.T1.M8.L12|L12-C07|base|-"
      "C07.T1.X4.L12|L12-C07|base|384"
      "C08.T3.M16.L02|L02-C08|base|-"
      "C09.T3.M16.L02|L02-C09|base|-"
      "C09.T3.M8.L02|L02-C09|base|-"
      "C09.T3.X4.L02|L02-C09|base|768"
      "C10.T1.M16.L02|L02-C09|base|-"
      "C10.T1.M8.L02|L02-C09|base|-"
    )
    ;;
  c)
    # Batch C is adaptive, not planned: three second sites the first 28 cells
    # made worth running.  C06/T1 and C07/T1 both cleared 0.90 at X4 at L12 and
    # 5 rule 2 needs a pass at ">=2 of 3 sites", so a single X4 site is an
    # anecdote; C06/T5 matched dense ridge at L12 while contracting storage,
    # which is the first time a route has done that at an attention component,
    # and it had one site.
    GPU=4
    CELLS=(
      "C06.T1.X4.L08|L08-C06|base|768"
      "C07.T1.X4.L08|L08-C07|base|384"
      "C06.T5.M16.L08|L08-C06|base|-"
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
  CELL="${entry%%|*}"; rest="${entry#*|}"
  CACHE="${rest%%|*}"; rest="${rest#*|}"
  KNOBSEL="${rest%%|*}"; X4="${rest##*|}"
  case "$KNOBSEL" in
    base) KNOBS=$W/mxg-knobs.json ;;
    t9)   KNOBS=$W/mxg-knobs-t9.json ;;
    t9t3) KNOBS=$W/mxg-knobs-t9-t3.json ;;
    c04)  KNOBS=$W/mxg-knobs-c04.json ;;
    *) echo "unknown knob selector $KNOBSEL" >&2; exit 2 ;;
  esac
  X4ARG=""
  [ "$X4" != "-" ] && X4ARG="--x4-width $X4"
  OUT=/mnt/ai_data/neural-forge-matrix/cells/$CELL
  if [ -f "$OUT/cell-receipt.json" ]; then
    OUT=$OUT/wave3
    echo "    (re-run of a sealed cell; waves 1-2 artifacts untouched -> $OUT)"
  fi
  echo "=== $(date -u +%H:%M:%S) start $CELL (cache $CACHE, knobs $KNOBSEL, x4 $X4)"
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
    $X4ARG \
    --knobs $KNOBS \
    > $W/logs/cell-$CELL.log 2>&1
  rc=$?
  echo "=== $(date -u +%H:%M:%S) done  $CELL rc=$rc :: $(tail -1 $W/logs/cell-$CELL.log)"
done

echo "BATCH $BATCH COMPLETE, $(wc -l < $SHARD) shard rows"
