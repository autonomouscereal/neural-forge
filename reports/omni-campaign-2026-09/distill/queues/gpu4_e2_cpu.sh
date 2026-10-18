#!/usr/bin/env bash
# lane gpu4: E2 build-needles + CPU fp32 parity arm on OMNI1 at 4k (CPU only, nice 15, 16 threads, no GPU visible)
set -uo pipefail
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export PYTHONPATH=$OC/distill/tools OC_CALIBRATION_DIR=$OC/calibration CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
T=$OC/distill/tools/p11_needle_engine.py
OMNI1=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
TOK=/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07
N=$OC/distill/p11/needles_v1
echo "start $(date -u +%FT%TZ) tool sha $(sha256sum $T | cut -c1-64)"
if [ ! -f $N/manifest.json ]; then
  nice -n 15 $PY $T build-needles --tokenizer-dir $TOK --out $N; echo "build-needles rc=$? $(date -u +%FT%TZ)"
fi
nice -n 15 $PY $T parity --model $OMNI1 --chimera-src $OC/import/rq27b/tree_ef1ffb9 --needles $N --length 4096 \
  --device cpu --dtype float32 --kda-kernel loop --chunk 1024 --attn-budget-mib 256 --label p11_omni1 \
  --out $OC/distill/tps/p11_omni1_e2_parity_cpu_fp32_loop_L4096.json
echo "parity rc=$? $(date -u +%FT%TZ)"
