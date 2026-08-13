#!/bin/bash
# The free measurement wave 21 opens with, and the one that can falsify its own
# step-3 hypothesis before a build.  It builds nothing: one capture pass, then
# the retained READ energy of every sealed basis PER ENDPOINT -- the number the
# law is about, which no census has ever printed per reader -- and what the
# max-min iteration would achieve at the same widths.
#
# Wave 20's selectorq.sh is the model for this: a measurement that can kill the
# wave's own hypothesis is the measurement to take first.  Under the anti-proxy
# rule it decides nothing about a child that IS built; the review does that.
set -u
. /home/cereal/nf-mx21/env.sh
GPU="${1:?gpu}"
W19=/mnt/ai_data/neural-forge-matrix/wave19/basis
mkdir -p $W/logs $W/basis
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX21/mx21_build.py \
  --mode endpoints --device cuda:0 \
  --evaluate-basis $W19/E-reader-pca-896.safetensors \
  --evaluate-basis $W19/E-reader-pca-768.safetensors \
  --evaluate-basis $W19/E-reader-pca-640.safetensors \
  --evaluate-basis $W19/E-reader-pca-512.safetensors \
  --maxmin-width 896 --maxmin-width 768 --maxmin-width 736 \
  --selector-out $W/basis/endpoint-retention.json \
  > $W/logs/endpoints.log 2>&1
echo "endpoints rc=$? -> $W/basis/endpoint-retention.json"
tail -6 $W/logs/endpoints.log
