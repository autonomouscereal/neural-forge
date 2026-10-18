#!/bin/bash
# run_teacher_score_parity.sh -- prove gguf_teacher_cache + teacher_score against qtext_teacher_cache (the production HF
# reference, exact fp32 head) on Qwen3.5-0.8B: one corpus (real prose + random-id rows, variable lengths, a store mask),
# three caches -- HF fp32 on CPU (reference), GGUF f32 through libllama on CPU, the same GGUF on a P40 -- then compare.
#   setsid nohup bash run_teacher_score_parity.sh > $W/parity.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
LLAMA=$OC/export/llamacpp
HF=/home/cereal/models/Qwen3.5-0.8B
W=${W:-/mnt/lexar/omni-campaign/distill/fullkd/scorer-parity}
GPU=${GPU:-1}
G=$W/gguf/Qwen3.5-0.8B-f32.gguf
export PYTHONPATH=$T:$LLAMA/llama.cpp/gguf-py CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src OMP_NUM_THREADS=16
mkdir -p "$W/gguf" "$W/caches"
utc() { date -u +%FT%TZ; }
echo "$(utc) parity start"
if [ ! -s "$G" ]; then
  CUDA_VISIBLE_DEVICES= "$PY" "$LLAMA/llama.cpp/convert_hf_to_gguf.py" "$HF" --outtype f32 --outfile "$G.tmp" > "$W/convert.log" 2>&1 \
    || { echo "convert failed"; tail -20 "$W/convert.log"; exit 1; }
  mv "$G.tmp" "$G"
fi
ls -la "$W/gguf"
CORPUS=$W/parity_corpus.safetensors
if [ ! -s "$CORPUS" ]; then
  "$PY" -c "
import sys; sys.path.insert(0, '$T')
import qtext_teacher_cache as q
from pathlib import Path
print(q.make_parity_text_corpus(Path('$CORPUS'), Path('$HF'), 8, 512, 20261018))"
fi
echo "$(utc) reference: qtext HF fp32 on CPU"
CUDA_VISIBLE_DEVICES= "$PY" "$T/qtext_teacher_cache.py" build-text --model-dir "$HF" --device cpu --dtype float32 \
  --linear-attn-kernel torch --teacher q35-08b-hf-fp32 --corpus "$CORPUS" --out-root "$W/caches" --rows-per-shard 4 \
  --micro-batch 2 --teacher-digest stat --purpose "teacher_score parity reference" > "$W/ref.log" 2>&1 || { echo "ref failed"; tail -20 "$W/ref.log"; }
echo "$(utc) gguf f32 on CPU"
CUDA_VISIBLE_DEVICES= "$PY" "$T/gguf_teacher_cache.py" build --gguf "$G" --teacher q35-08b-gguf-f32-cpu --corpus "$CORPUS" \
  --out-root "$W/caches" --rows-per-shard 4 --teacher-digest stat --scorer-arg=-ngl --scorer-arg=0 --scorer-arg=-c --scorer-arg=1024 \
  --scorer-arg=-b --scorer-arg=256 --scorer-arg=-ub --scorer-arg=128 > "$W/cpu.log" 2>&1 || { echo "cpu failed"; tail -20 "$W/cpu.log"; }
echo "$(utc) gguf f32 on GPU$GPU"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU flock -w 600 "$OC/distill/queues/gpu$GPU.lock" "$PY" "$T/gguf_teacher_cache.py" build --gguf "$G" --teacher q35-08b-gguf-f32-gpu$GPU \
  --corpus "$CORPUS" --out-root "$W/caches" --rows-per-shard 4 --teacher-digest stat --scorer-arg=-ngl --scorer-arg=999 \
  --scorer-arg=-c --scorer-arg=1024 --scorer-arg=-b --scorer-arg=256 --scorer-arg=-ub --scorer-arg=128 > "$W/gpu.log" 2>&1 \
  || { echo "gpu failed"; tail -20 "$W/gpu.log"; }
SHA=$(sha256sum "$CORPUS" | cut -d" " -f1)
for arm in q35-08b-gguf-f32-cpu q35-08b-gguf-f32-gpu$GPU; do
  echo "$(utc) compare reference vs $arm"
  "$PY" "$T/gguf_teacher_cache.py" compare --cache "$W/caches/q35-08b-hf-fp32/$SHA" --cache "$W/caches/$arm/$SHA" \
    --receipt "$W/compare-$arm.json" 2>&1 | tail -20
done
echo "$(utc) parity end"
