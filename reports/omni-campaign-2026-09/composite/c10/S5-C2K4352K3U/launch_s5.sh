#!/bin/sh
# S5: the C10 union install of record (2026-10-11, Claude, S5 build lane).  c10_union_install.py >= 1.5.1,
# CLOSED LOOP, design of record from composite/c10/design_sweep/VERDICT.md (sha d5e1f60b...):
#   a003 = trunk/additive:0.03  (Y = 1.0 FFN_trunk + 0.03 TERM_k3); runner-up g01 = trunk/rms:0.1 (TARGET=...).
# K3 dense L1-92 (k3-run-oct09-claude, pass receipt 8948bab7...), c17 ridge charts L00-L63 + alignment.json
# (92 MoE layers -> 64 trunk layers; dense K3 L0 -> nowhere per that alignment), --chart-bias-ceiling 0
# (record only, owner policy 2026-10-06), --k3-channels 12288 (<= 12,538: the five media units stay budgeted).
# Parent targets: the moved S0 import (/mnt/ai_data, MOVED.json at composite/s0/); only read for scores here.
# Disk: --min-free-gib 60 stops (exit 3, resumable) before any partial layer or the child write would cross
# the 60 GiB root floor.  Usage (GPU = nvidia-smi index, check < 2000 MiB first):
#   GPU=4 STOP=63 setsid nohup sh launch_s5.sh > logs/s5_install.log 2>&1 < /dev/null &      (all layers, no child)
#   GPU=4 setsid nohup sh launch_s5.sh --resume > logs/s5_write.log 2>&1 < /dev/null &       (writes the child)
OC=/home/cereal/omni-campaign
S=$OC/composite/c10/S5-C2K4352K3U
EF=$OC/import/rq27b/tree_ef1ffb9
K3D=$OC/k3-run-oct09-claude/fold/dense
if [ ! -d $S/k3_dense_1_92 ]; then
  for l in $(seq -w 1 92); do mkdir -p $S/k3_dense_1_92/L$l; ln -sfn $K3D/L$l/ffn.safetensors $S/k3_dense_1_92/L$l/ffn.safetensors; done
fi
mkdir -p $S/logs $OC/composite/s5
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CHIMERA_SRC=$EF/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $OC
exec nice -n 15 /usr/bin/time -v /home/cereal/model-surgery-lab/venv/bin/python composite/tools/c10_union_install.py \
  --trunk /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native \
  --parent-trunk /mnt/ai_data/omni-campaign/s0-probe-sep04 \
  --k3-dense $S/k3_dense_1_92 --k3-form E --k3-layers 1-92 \
  --k3-config /mnt/lexar/models/moonshotai/Kimi-K3/config.json \
  --charts $OC/composite/c17/charts --chart-arm ridge --chart-bias-ceiling 0 \
  --evidence $OC/qwen-fold/evidence/omni-evidence-27bu-v1.json \
  --k3-channels 12288 --parent-parameter-count 27781427952 \
  --union-target "${TARGET:-a003=trunk/additive:0.03}" \
  --device cuda:0 --selection-device cuda:0 --solve-device cuda:0 \
  --engine streamed --group-size 3 --chunk 8192 --gram-block 16384 --min-free-gib 60 \
  --out $S/child --receipt $OC/composite/s5/union_receipt.json ${STOP:+--stop-after-layer $STOP} "$@"
