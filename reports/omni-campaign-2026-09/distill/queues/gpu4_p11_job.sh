#!/usr/bin/env bash
# gpu4_p11_job.sh -- lane gpu4 job 6: S0 P11 OMNI1 teacher-forced needle baseline on GPU4 (single RTX 3090).
# Run by gpu4.sh after the card gate (S8 gone, GPU4 < 2000 MiB) and only when gpu4.p11.ready carries this file's sha.
# Engine: E2 = distill/tools/p11_needle_engine.py 1.0.0 (pinned below). First the GPU parity arm at 4,096 tokens
# (stock layer vs E2 with the fla recurrence, bf16, chunk 1024 + a 256 MiB score budget so chunk boundaries and query
# sub-chunks are crossed) gates the read; then `read`: 32k (2 depths + no-needle control), 128k (2 depths),
# 256k (2 depths), one streamed pass per length, a receipt after each pass.
set -uo pipefail
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
T=$OC/distill/tools/p11_needle_engine.py
WANT_T=b47eeff31774ab082bdf8001c14181931b53f6b275a2481e173f0c963aa81934
HAVE_T=$(sha256sum $T | cut -c1-64)
[ "$HAVE_T" = "$WANT_T" ] || { echo "refuse: $T sha256 $HAVE_T != pinned $WANT_T"; exit 3; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4   # torch's default order differs from nvidia-smi here
export PYTHONPATH=$OC/distill/tools OC_CALIBRATION_DIR=$OC/calibration OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "p11 job start $(date -u +%FT%TZ) on $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i 4)"
M=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
COMMON=(--model $M --chimera-src $OC/import/rq27b/tree_ef1ffb9 --needles $OC/distill/p11/needles_v1
        --device cuda:0 --dtype bfloat16 --kda-kernel fla)
nice -n 15 $PY $T parity "${COMMON[@]}" --length 4096 --chunk 1024 --attn-budget-mib 256 --parity-min-top1 0.90 \
  --out $OC/distill/tps/p11_omni1_e2_parity_gpu_bf16_fla_L4096_prejob.json
rc=$?
[ $rc -eq 0 ] || { echo "p11 job: E2 parity arm rc=$rc, no needle read $(date -u +%FT%TZ)"; exit $rc; }
nice -n 15 $PY $T read "${COMMON[@]}" --lengths 32768,131072,262144 --chunk 8192 --attn-budget-mib 2048 \
  --parity-length 0 --out-dir $OC/distill/tps --label p11_omni1
rc=$?
echo "p11 job end rc=$rc $(date -u +%FT%TZ)"
exit $rc
