#!/bin/bash
# Generate the frozen review dossier for a child whose ARCH receipt is ALREADY
# sealed, without going near the sealer.  Wave 23's script with ONE repair: the
# child-tokenizer flag is derived and passed here too.
#
# Wave 23's copy predated ADDENDUM 15 15.3 and did not pass it, so it would have
# refused every child in that wave and every child in this one -- the whole
# reason it exists is the case where the seal succeeded and the review did not,
# and on the vocabulary axis that is precisely the case where a missing flag is
# fatal.  The flag is DERIVED from the two configs rather than typed, and both
# scripts refuse it when the vocabularies agree and refuse its absence when they
# do not, so this line cannot silently do the wrong thing in either direction.
#
#   reviewonly.sh <gpu> <tag>...
set -u
. /home/cereal/nf-mx41/env.sh
GPU="${1:?gpu}"; shift
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$NF_DEPLOY/src:$NF_DEPLOY/experiments"
PV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $PARENT/config.json)

for TAG in "$@"; do
  R=$W/cells/ARCH.$TAG/cell-receipt.json
  if [ ! -f "$R" ]; then
    echo "$TAG REFUSED: no sealed receipt at $R"; continue
  fi
  if [ -f "$W/cells/ARCH.$TAG/frozen-review-dossier.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "$TAG REFUSED: dossier already exists (FORCE=1 to regenerate)"; continue
  fi
  CV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $W/children/MX41-$TAG/config.json)
  CHILD_TOK=""
  if [ "$CV" != "$PV" ]; then CHILD_TOK="--child-tokenizer $W/children/MX41-$TAG"; fi
  $PY $NF_DEPLOY/experiments/run_lfm2_composition_frozen_review.py \
    --receipt "$R" \
    --child-model $W/children/MX41-$TAG \
    $CHILD_TOK \
    --parent-model $PARENT \
    --out $W/cells/ARCH.$TAG/frozen-review-dossier.json \
    --device cuda:0 --require-compute-architecture sm86 \
    --dtype float32 --maximum-new-tokens 1024 \
    > $W/logs/review-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'sealed review surface' $W/logs/review-$TAG.log | tail -1)"
done
echo "REVIEWONLY $GPU DONE"
