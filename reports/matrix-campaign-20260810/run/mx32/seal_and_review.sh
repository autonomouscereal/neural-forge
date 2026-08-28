#!/bin/bash
# Wave 30: seal one standard-shape child under ADDENDUM 12 and generate its
# frozen review dossier.  Wave 29's script with the wave root, the campaign
# wave number and the build receipt name moved -- a receipt filename that names
# last wave is how a driver silently seals last wave's build.
#
#   NF_DEPLOY=<deployment dir>  ./seal_and_review.sh <tag> <child dir name> <gpu>
#
# GPU is pinned by PCI bus id AND re-checked in-process: `NOTES-MX-E` §2 is the
# wave where CUDA's FASTEST_FIRST ordering put a batch on a P40 and nothing in
# the receipt said so.
set -u
TAG="${1:?tag}"; CHILD="${2:?child dir}"; GPU="${3:?gpu}"
D="${NF_DEPLOY:?set NF_DEPLOY to the deployment cut from THIS wave-29 commit}"
W=/mnt/ai_data/neural-forge-matrix/wave32
PY=/home/cereal/model-surgery-lab/venv/bin/python
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M
HEAD=$(basename "$D" | sed 's/neural-forge-matrix-//')

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$D/src:$D/experiments"
mkdir -p $W/cells/ARCH.$TAG $W/shards

# ADDENDUM 15 15.3.  A child whose vocab_size differs from the parent's is
# prompted and decoded through its OWN tokenizer; the flag is DERIVED from the
# two configs rather than typed, because a tokenizer named by hand could name
# one the child was not built with.  Both scripts refuse the flag when the
# vocabularies agree and refuse its absence when they do not, so this line
# cannot silently do the wrong thing in either direction.
CV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $W/children/$CHILD/config.json)
PV=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['vocab_size'])" $PARENT/config.json)
CHILD_TOK=""
if [ "$CV" != "$PV" ]; then CHILD_TOK="--child-tokenizer $W/children/$CHILD"; fi
echo "vocab: parent $PV child $CV ${CHILD_TOK:-(shared tokenizer)}"

$PY $D/experiments/seal_matrix_architecture_child.py \
  --cell-id ARCH.$TAG \
  --cell-instance-id mx32-$(echo $TAG | tr 'A-Z' 'a-z')-0001 \
  --child-model $W/children/$CHILD \
  $CHILD_TOK \
  --build-receipt $W/children/$CHILD/mx32-build-receipt.json \
  --parent-model $PARENT \
  --campaign-wave 32 \
  --repo-head $HEAD \
  --device cuda:0 --require-compute-architecture sm86 \
  --out $W/cells/ARCH.$TAG/cell-receipt.json \
  --ledger-shard $W/shards/arch-$TAG.jsonl || exit 4

$PY $D/experiments/run_lfm2_composition_frozen_review.py \
  --receipt $W/cells/ARCH.$TAG/cell-receipt.json \
  --child-model $W/children/$CHILD \
  $CHILD_TOK \
  --parent-model $PARENT \
  --out $W/cells/ARCH.$TAG/frozen-review-dossier.json \
  --device cuda:0 --require-compute-architecture sm86 \
  --dtype float32 --maximum-new-tokens 1024 || exit 5
echo "DONE $TAG"
