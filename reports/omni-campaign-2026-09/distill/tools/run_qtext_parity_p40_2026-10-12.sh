#!/usr/bin/env bash
# qtext lane, 2026-10-12: parity arms + real-checkpoint smokes for qtext_teacher_cache.py
# on ONE P40 (GPU3, sm61), Pascal venv (torch 2.6 cu118; the main venv's torch 2.11
# cu128 has no sm_61 kernels).  Sequential; every step logs and is independent.
#   setsid nohup bash run_qtext_parity_p40_2026-10-12.sh > $OC/distill/logs/qtext_p40.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
G3=GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
W=$OC/distill/parity/qtext-2026-10-12
ALLOC=$OC/composite/receipts/allocator
mkdir -p "$W" "$OC/distill/tps"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G3
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
RUN="nice -n 15 $PY $T/qtext_teacher_cache.py"
step() { echo; echo "===== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 4p; }

step "1 parity-text 0.8B fp32 (fla kernels on sm61)"
$RUN parity-text --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 4 --parity-rows 8 \
  --parity-length 1024 --work-dir $W/text-fp32 --receipt $OC/distill/tps/qtext-parity-text-08b-fp32-p40.json \
  --allocator-receipt $ALLOC/qtext-parity-text-08b-p40-gpu3-claude.json --purpose "qtext parity-text 0.8B fp32"
echo "exit $?"

step "2 parity-vl 0.8B fp32, two drawn images (one row single-image, one row both images)"
$RUN parity-vl --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 2 \
  --work-dir $W/vl-fp32 --receipt $OC/distill/tps/qtext-parity-vl-08b-fp32-p40.json \
  --allocator-receipt $ALLOC/qtext-parity-vl-08b-p40-gpu3-claude.json --purpose "T(i) parity-vl 0.8B fp32"
echo "exit $?"

step "3 parity-vl 0.8B fp32, one 4-frame video row (video-in path)"
mkdir -p $W/video-src/frames
$PY - "$W/video-src" <<'EOF'
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(sys.argv[1]); names = []
for i in range(8):
    im = Image.new("RGB", (320, 224), (30, 30, 30)); d = ImageDraw.Draw(im)
    d.ellipse([20 + 30 * i, 80, 80 + 30 * i, 140], fill=(220, 200, 40)); im.save(root / "frames" / f"f{i}.png")
    names.append(f"frames/f{i}.png")
row = {"row_id": "video-parity-0", "messages": [
    {"role": "user", "content": [{"type": "video", "frames": names, "fps": 2.0},
                                 {"type": "text", "text": "Which way does the yellow ball move?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "It moves from left to right across the frame."}]}]}
(root / "video.jsonl").write_text(json.dumps(row) + "\n")
EOF
$RUN parity-vl --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 1 --corpus $W/video-src/video.jsonl \
  --work-dir $W/video-fp32 --receipt $OC/distill/tps/qtext-parity-video-08b-fp32-p40.json \
  --allocator-receipt $ALLOC/qtext-parity-video-08b-p40-gpu3-claude.json --purpose "T(i) parity video 0.8B fp32"
echo "exit $?"

step "4 parity-text 0.8B bf16 on sm61 (extra arm; the stage-C dtype, emulated on Pascal)"
$RUN parity-text --model-dir $M08 --device cuda:0 --dtype bfloat16 --micro-batch 4 --parity-rows 8 \
  --parity-length 1024 --work-dir $W/text-bf16 --receipt $OC/distill/tps/qtext-parity-text-08b-bf16-p40.json \
  --allocator-receipt $ALLOC/qtext-parity-text-08b-bf16-p40-gpu3-claude.json --purpose "qtext parity-text 0.8B bf16"
echo "exit $?"

step "5 real teacher smoke: build-text on Qwen3.8-27B-U, fp32 on one P40, 4 rows x 512, 2 shards"
$PY - "$W/q27-text-corpus.safetensors" "$M27" <<'EOF'
import sys
sys.path.insert(0, "/home/cereal/omni-campaign/distill/tools")
import qtext_teacher_cache as qc
from pathlib import Path
print(qc.make_parity_text_corpus(Path(sys.argv[1]), Path(sys.argv[2]), rows=4, length=512, seed=27))
EOF
$RUN build-text --model-dir $M27 --teacher q38-27bu-smoke --corpus $W/q27-text-corpus.safetensors \
  --out-root $W/q27-cache --device cuda:0 --dtype float32 --micro-batch 2 --rows-per-shard 2 --digest-threads 8 \
  --allocator-receipt $ALLOC/qtext-q27-text-smoke-p40-gpu3-claude.json --purpose "qtext 27B build-text smoke fp32"
echo "exit $?"

step "6 real teacher smoke: build-vl (T(i)) on Qwen3.8-27B-U, fp32, the two drawn images"
$RUN build-vl --model-dir $M27 --teacher q38-27bu-vl-smoke --corpus $W/vl-fp32/parity_vl_corpus.jsonl \
  --out-root $W/q27-cache --device cuda:0 --dtype float32 --micro-batch 2 --rows-per-shard 2 --digest-threads 8 \
  --allocator-receipt $ALLOC/qtext-q27-vl-smoke-p40-gpu3-claude.json --purpose "T(i) 27B build-vl smoke fp32"
echo "exit $?"

step "7 verify the 27B smoke caches"
for d in $W/q27-cache/*/*/; do $RUN verify --cache-dir "$d"; echo "exit $?"; done
step "done"
