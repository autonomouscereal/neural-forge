#!/bin/bash
# The per-endpoint census, wave 22's copy, and it runs AFTER `basisq.sh` rather
# than before it.  Wave 21 ran this first because it was that wave's falsifier:
# the max-min hypothesis dies if the worst reader is already retained at the
# pooled rate, and that is knowable without a build.  Wave 22 adds no hypothesis
# -- every cell in it is a build against machinery that already exists -- so the
# same pass is run for the thing it is unambiguously good for: printing the
# retained READ energy of THIS WAVE'S OWN new bases per endpoint, which needs
# those bases to exist.
#
# It still builds nothing and it still decides nothing about a child that IS
# built; the review does that.
set -u
. /home/cereal/nf-mx22/env.sh
GPU="${1:?gpu}"
W19=/mnt/ai_data/neural-forge-matrix/wave19/basis
W21=/mnt/ai_data/neural-forge-matrix/wave21/basis
mkdir -p $W/logs $W/basis
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX22/mx22_build.py \
  --mode endpoints --device cuda:0 \
  --evaluate-basis $W19/E-reader-pca-896.safetensors \
  --evaluate-basis $W/basis/E-reader-maxmin-896.safetensors \
  --evaluate-basis $W/basis/E-reader-pca-720.safetensors \
  --evaluate-basis $W21/E-reader-pca-736.safetensors \
  --maxmin-width 896 --maxmin-width 720 \
  --selector-out $W/basis/endpoint-retention.json \
  > $W/logs/endpoints.log 2>&1
echo "endpoints rc=$? -> $W/basis/endpoint-retention.json"
tail -6 $W/logs/endpoints.log
