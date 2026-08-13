#!/bin/bash
# The free go/no-go census, and the sealed basis artifact for every rung of the
# ladder.  One pass over the corpus prices every candidate width in both second
# moments on both splits; the per-width runs then write the bases the children
# are built against, one digest each.
set -u
. /home/cereal/nf-mx19/env.sh
GPU="${1:?gpu}"
mkdir -p $W/basis $W/logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX19/mx19_build.py \
  --mode basis --device cuda:0 --basis reader-pca > $W/basis/census-reader-pca.json 2>$W/logs/census.log
echo "census rc=$?"
for W_ in 960 896 768 512; do
  for MODE in reader-pca evidence-pca; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX19/mx19_build.py \
      --mode basis --device cuda:0 --hidden-size $W_ --basis $MODE \
      --basis-file $W/basis/E-$MODE-$W_.safetensors > /dev/null 2>>$W/logs/census.log
    echo "basis $MODE $W_ rc=$?"
  done
done
echo "CENSUSQ DONE"
