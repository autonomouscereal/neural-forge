#!/bin/bash
# The free measurement wave 20 opens with.  It builds nothing: one capture pass
# with the case-implicated row selectors on, then the retained fraction of the
# pooled stream and of each selector's rows for every sealed wave-19 basis, for
# a grid of attribution-weighted betas, and for the reader-generalised subspace.
#
# It decides how much GPU steps 1 and 3 are worth.  Under the anti-proxy rule it
# decides nothing about a child that IS built -- the review does that -- but a
# measurement that can falsify the wave's hypothesis before a build is exactly
# the measurement to take first.
set -u
. /home/cereal/nf-mx20/env.sh
GPU="${1:?gpu}"
W19=/mnt/ai_data/neural-forge-matrix/wave19/basis
mkdir -p $W/logs $W/basis
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX20/mx20_build.py \
  --mode selectors --device cuda:0 \
  --hidden-size "${2:-768}" \
  --beta-grid "${3:-0,0.125,0.25,0.5,0.75,1.0}" \
  --evaluate-basis $W19/E-reader-pca-896.safetensors \
  --evaluate-basis $W19/E-reader-pca-768.safetensors \
  --evaluate-basis $W19/E-reader-pca-640.safetensors \
  --evaluate-basis $W19/E-reader-pca-512.safetensors \
  --evaluate-basis $W19/E-evidence-pca-768.safetensors \
  --selector-out $W/basis/selector-retention-${2:-768}.json \
  > $W/logs/selectors-${2:-768}.log 2>&1
echo "selectors rc=$? -> $W/basis/selector-retention-${2:-768}.json"
tail -3 $W/logs/selectors-${2:-768}.log
