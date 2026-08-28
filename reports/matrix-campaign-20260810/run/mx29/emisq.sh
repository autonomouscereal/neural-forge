#!/bin/bash
# Wave 29: the free emission forensic (retention plan #5's pre-read, with
# ADDENDUM 16.6's sink and massive-activation channels).  Builds nothing.
#
#   emisq.sh <gpu> <LABEL>:<child dir> ...
#   PARENT_ONLY=1 emisq.sh <gpu>
set -u
. /home/cereal/nf-mx29/env.sh
GPU="${1:?gpu}"; shift
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M
PROBES=${PROBES:-$NF_DEPLOY/research/lfm2_semantic_whole_response_prompts_v1.json}
mkdir -p $W/prereads $W/logs
CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
PV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $PARENT/config.json)
for spec in "$@"; do
  LABEL=$(echo "$spec" | cut -d: -f1)
  DIR=$(echo "$spec" | cut -d: -f2)
  if [ ! -d "$DIR" ]; then echo "$LABEL SKIPPED: no child at $DIR"; continue; fi
  CV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $DIR/config.json)
  TOKFLAG=""
  if [ "$CV" != "$PV" ]; then TOKFLAG="--child-tokenizer $DIR"; fi
  $PY $MX29/mx29_emission.py --deployment $NF_DEPLOY --parent $PARENT \
    --child "$DIR" $TOKFLAG --label "$LABEL" --probes $PROBES \
    --out $W/prereads/emission-$LABEL.json > $W/logs/emission-$LABEL.log 2>&1
  echo "$LABEL rc=$? :: $(grep -h 'reading' -A6 $W/logs/emission-$LABEL.log | tail -8 | tr '\n' ' ')"
done
echo "EMISQ $GPU DONE"
