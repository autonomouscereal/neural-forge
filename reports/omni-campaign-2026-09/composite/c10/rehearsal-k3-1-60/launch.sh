#!/bin/sh
# M4 C10 partial-K3 union-install rehearsal (2026-10-10, Claude), c10_union_install.py >= 1.3.0.
# STOP (env, default 63) = --stop-after-layer: the run persists layers 0..STOP under child.partial/ and exits 3;
# it never writes the child (root floor 60 GB: child 54.4 GB + partial 32.7 GB do not fit beside K3 / the lanes).
# Prefix rehearsal of 2026-10-10: STOP=2 sh launch.sh --overwrite   (trunk layers 0,1,2 = K3 [1],[2],[3,4])
OC=/home/cereal/omni-campaign
R=$OC/composite/c10/rehearsal-k3-1-60
export CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $OC
exec nice -n 19 /usr/bin/time -v /home/cereal/model-surgery-lab/venv/bin/python composite/tools/c10_union_install.py \
  --trunk /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native \
  --parent-trunk $OC/composite/s0/q38-27bu-chimera-bf16-sep04 \
  --k3-dense $R/k3_dense_1_60 --k3-form E --k3-layers 1-60 \
  --k3-config /mnt/lexar/models/moonshotai/Kimi-K3/config.json \
  --charts $OC/composite/c17/charts --chart-arm ridge --chart-bias-ceiling 0 \
  --alignment $R/alignment_override_k3_1_60.json \
  --evidence $OC/composite/corpus/omni-evidence-27b-v1.json \
  --k3-channels 12288 --parent-parameter-count 27781427952 \
  --device cuda:0 --selection-device cuda:0 --solve-device cuda:0 \
  --engine streamed --group-size 3 --chunk 8192 --gram-block 16384 \
  --out $R/child --receipt $R/union_receipt.json --stop-after-layer ${STOP:-63} "$@"
