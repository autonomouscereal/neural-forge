#!/usr/bin/env bash
# qtext lane, tool 1.0.4 (wave-2 lows: L1 processor files locked, L2 target_mask shifted in the
# tool, L3 fp32 head; plus length-planned batching, corpus-lane VL records, video files via
# ffmpeg, --max-new-shards, smoke-vl), 2026-10-13.  CPU only (GPU0 is step 1's until its S8
# review ends; the P40s are lane p40's).  Steps:
#   0  negative control: the 1.0.4 test file against the 1.0.3 tool (must fail)
#   1-3  the Qwen3.5-0.8B parity arms exactly as 1.0.3 ran them (text / two-image VL /
#        8-frame video), fp32, torch GDN -- must stay exact
#   4  diagnostic arm: text at bf16 with the fp32 head (the production head path)
#   5  verify every cache written here
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
M08=/home/cereal/models/Qwen3.5-0.8B
W1=$OC/distill/parity/qtext-2026-10-12
W=$OC/distill/parity/qtext-2026-10-13-cpu-v104
R=$OC/distill/tps
mkdir -p "$W"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
RUN="nice -n 15 $PY $T/qtext_teacher_cache.py"
COMMON="--model-dir $M08 --device cpu --dtype float32 --linear-attn-kernel torch"
sha256sum $T/qtext_teacher_cache.py $T/test_qtext_teacher_cache.py $OC/distill/tps_probe.py $T/_superseded/qtext_teacher_cache_1.0.3.py
echo "===== 0 negative control: 1.0.4 tests vs the 1.0.3 tool $(date -u +%FT%TZ)"
N=$OC/distill/scratch/qtext-v104-negctl
rm -rf "$N"; mkdir -p "$N"
cp $T/_superseded/qtext_teacher_cache_1.0.3.py $N/qtext_teacher_cache.py
cp $T/test_qtext_teacher_cache.py $N/
(cd $N && QTEXT_TPS_PROBE=$OC/distill/tps_probe.py PYTHONPATH=$N nice -n 15 $PY -m pytest -o addopts="" -p no:cacheprovider \
   -p no:warnings test_qtext_teacher_cache.py -q 2>&1 | tail -4)
rm -rf "$N"
echo "===== 1 parity-text $(date -u +%FT%TZ)"
$RUN parity-text $COMMON --micro-batch 4 --parity-rows 8 --parity-length 1024 --work-dir $W/text \
  --receipt $R/qtext-parity-text-08b-fp32-cpu-v104.json; echo "exit $?"
echo "===== 2 parity-vl two images $(date -u +%FT%TZ)"
$RUN parity-vl $COMMON --micro-batch 2 --work-dir $W/vl --receipt $R/qtext-parity-vl-08b-fp32-cpu-v104.json; echo "exit $?"
echo "===== 3 parity-vl 8-frame video $(date -u +%FT%TZ)"
$RUN parity-vl $COMMON --micro-batch 1 --corpus $W1/video-src/video.jsonl --work-dir $W/video \
  --receipt $R/qtext-parity-video-08b-fp32-cpu-v104.json; echo "exit $?"
echo "===== 4 diagnostic: parity-text bf16, fp32 head $(date -u +%FT%TZ)"
$RUN parity-text --model-dir $M08 --device cpu --dtype bfloat16 --linear-attn-kernel torch --head-precision fp32 \
  --micro-batch 4 --parity-rows 8 --parity-length 1024 --work-dir $W/text-bf16-fp32head \
  --receipt $R/qtext-parity-text-08b-bf16-fp32head-cpu-v104.json; echo "exit $?"
echo "===== 5 verify $(date -u +%FT%TZ)"
for d in $W/*/cache/*/*/; do $RUN verify --cache-dir "$d" | tail -1; done
echo "===== done $(date -u +%FT%TZ)"
