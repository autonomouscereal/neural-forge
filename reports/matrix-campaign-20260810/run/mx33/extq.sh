#!/bin/bash
# Wave 29: the ADDENDUM 16.1 EXTENSION surface, one model per spec.
#
# This queue produces a SECOND coordinate and never a verdict.  Nothing it writes
# reaches `seal_matrix_behavior_review.py`, no ledger row is generated from it,
# and no frontier claim in this wave or any later one fires on it.  Its whole
# purpose is resolution: a 15-point instrument cannot separate a real +1 from a
# tick, and five of eight round-2 kills rode exactly that margin.
#
# The parent runs ONCE per bank, not once per child: the parent's responses do
# not depend on which child preceded them, and 30 prompts of parent decode per
# child is an hour of GPU spent proving that.
#
#   extq.sh <gpu> <LABEL>:<child dir>:<cell receipt> ...
#   PARENT_ONLY=1 extq.sh <gpu>
set -u
. /home/cereal/nf-mx33/env.sh
GPU="${1:?gpu}"; shift
BANK=${BANK:-$NF_DEPLOY/research/lfm2_extension_bank_prompts_v1.json}
SEAL=${SEAL:-$NF_DEPLOY/reports/matrix-campaign-20260810/extension-bank-seal.json}
WHICH=${WHICH:-extension}
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M
mkdir -p $W/extension $W/logs

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU

if [ "${PARENT_ONLY:-0}" = "1" ]; then
  $PY $MX33/mx33_extreview.py --deployment $NF_DEPLOY --bank $BANK --seal $SEAL \
    --which $WHICH --parent-model $PARENT --parent-only --label PARENT \
    --out $W/extension/$WHICH-PARENT.json > $W/logs/ext-PARENT-$WHICH.log 2>&1
  echo "PARENT rc=$? :: $(tail -1 $W/logs/ext-PARENT-$WHICH.log)"
  exit 0
fi

PV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $PARENT/config.json)
for spec in "$@"; do
  LABEL=$(echo "$spec" | cut -d: -f1)
  DIR=$(echo "$spec" | cut -d: -f2)
  CELL=$(echo "$spec" | cut -d: -f3)
  if [ ! -d "$DIR" ]; then echo "$LABEL SKIPPED: no child at $DIR"; continue; fi
  if [ ! -f "$CELL" ]; then echo "$LABEL SKIPPED: no cell receipt at $CELL"; continue; fi
  CV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $DIR/config.json)
  TOKFLAG=""
  if [ "$CV" != "$PV" ]; then TOKFLAG="--child-tokenizer $DIR"; fi
  $PY $MX33/mx33_extreview.py --deployment $NF_DEPLOY --bank $BANK --seal $SEAL \
    --which $WHICH --receipt "$CELL" --child-model "$DIR" $TOKFLAG \
    --parent-model $PARENT --skip-parent --label "$LABEL" \
    --out $W/extension/$WHICH-$LABEL.json > $W/logs/ext-$LABEL-$WHICH.log 2>&1
  echo "$LABEL rc=$? :: $(tail -1 $W/logs/ext-$LABEL-$WHICH.log)"
done
echo "EXTQ $GPU DONE"
