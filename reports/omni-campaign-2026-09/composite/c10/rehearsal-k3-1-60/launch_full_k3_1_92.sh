#!/bin/sh
# TEMPLATE (not run on 2026-10-10): C10 union install of the FULL K3 (MoE layers 1-92) onto the S3 native trunk.
# Prerequisites (all read-only checks, see REHEARSAL.md section 5):
#   - K3 L61-L92 landed: k3-run-oct09-claude/fold/dense/L61..L92/{ffn.safetensors,done.json} + the 61-92 pass receipt
#   - c17 charts L00..L63 + alignment.json (92 MoE layers, dense L0 skipped) under $OC/composite/c17/charts
#   - $F/k3_dense_1_92/Lxx/ffn.safetensors symlinks (L01..L92) -> the K3 dense dir (built by the mkdir/ln loop below)
#   - nvidia-smi -i 4 < 2000 MiB; root avail >= 33 GB + 60 GB floor for the per-layer pass
# Per-layer pass (exit 3, no child):  STOP=63 sh launch_full_k3_1_92.sh --overwrite
# Child write (needs ~54.4 GB more beside the 32.7 GB partial on the --out filesystem):  sh launch_full_k3_1_92.sh --resume
OC=/home/cereal/omni-campaign
F=$OC/composite/c10/full-k3-1-92
K3D=/home/cereal/omni-campaign/k3-run-oct09-claude/fold/dense
if [ ! -d $F/k3_dense_1_92 ]; then
  for l in $(seq -w 1 92); do mkdir -p $F/k3_dense_1_92/L$l; ln -sfn $K3D/L$l/ffn.safetensors $F/k3_dense_1_92/L$l/ffn.safetensors; done
fi
mkdir -p $F/logs
export CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-4}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $OC
exec nice -n 19 /usr/bin/time -v /home/cereal/model-surgery-lab/venv/bin/python composite/tools/c10_union_install.py \
  --trunk /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native \
  --parent-trunk $OC/composite/s0/q38-27bu-chimera-bf16-sep04 \
  --k3-dense $F/k3_dense_1_92 --k3-form E --k3-layers 1-92 \
  --k3-config /mnt/lexar/models/moonshotai/Kimi-K3/config.json \
  --charts $OC/composite/c17/charts --chart-arm ridge --chart-bias-ceiling 0 \
  --evidence $OC/composite/corpus/omni-evidence-27b-v1.json \
  --k3-channels 12288 --parent-parameter-count 27781427952 \
  --device cuda:0 --selection-device cuda:0 --solve-device cuda:0 \
  --engine streamed --group-size 3 --chunk 8192 --gram-block 16384 \
  --out $F/child --receipt $OC/composite/s5/union_receipt.json ${STOP:+--stop-after-layer $STOP} "$@"
