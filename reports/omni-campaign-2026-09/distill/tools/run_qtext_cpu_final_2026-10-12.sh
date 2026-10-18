#!/usr/bin/env bash
# qtext lane, final tool (1.0.2) arms on the CPU, 2026-10-12.  GPU1-3 were held at 17.5-18.1 GiB
# by other lanes for the whole round-4 wait, so the real-checkpoint (Qwen3.5-0.8B) parity arms
# for the final tool run here: fp32, torch GDN kernels (fla has no CPU path), 16 threads, nice 15.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
M08=/home/cereal/models/Qwen3.5-0.8B
W1=$OC/distill/parity/qtext-2026-10-12
W=$OC/distill/parity/qtext-2026-10-12-cpu-v102
R=$OC/distill/tps
mkdir -p "$W"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
RUN="nice -n 15 $PY $T/qtext_teacher_cache.py"
COMMON="--model-dir $M08 --device cpu --dtype float32 --linear-attn-kernel torch"
sha256sum $T/qtext_teacher_cache.py $OC/distill/tps_probe.py
echo "===== 1 parity-text $(date -u +%FT%TZ)"
$RUN parity-text $COMMON --micro-batch 4 --parity-rows 8 --parity-length 1024 --work-dir $W/text \
  --receipt $R/qtext-parity-text-08b-fp32-cpu-v102.json; echo "exit $?"
echo "===== 2 parity-vl two images $(date -u +%FT%TZ)"
$RUN parity-vl $COMMON --micro-batch 2 --work-dir $W/vl --receipt $R/qtext-parity-vl-08b-fp32-cpu-v102.json; echo "exit $?"
echo "===== 3 parity-vl 8-frame video $(date -u +%FT%TZ)"
$RUN parity-vl $COMMON --micro-batch 1 --corpus $W1/video-src/video.jsonl --work-dir $W/video \
  --receipt $R/qtext-parity-video-08b-fp32-cpu-v102.json; echo "exit $?"
echo "===== 4 verify $(date -u +%FT%TZ)"
for d in $W/*/cache/*/*/; do $RUN verify --cache-dir "$d" | tail -1; done
echo "===== done $(date -u +%FT%TZ)"
