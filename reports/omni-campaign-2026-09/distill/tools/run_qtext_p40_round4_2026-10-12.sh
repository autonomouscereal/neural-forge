#!/usr/bin/env bash
# qtext lane round 4, 2026-10-12: EVERY arm on the final tool (1.0.2: card check before any
# session record; v1.0.1 sealed a refused start as a teacher change and failed the video arm)
# into a fresh work dir, so every receipt names one tool sha.  Each step runs on whichever
# P40 (GPU1-3) holds <= 2000 MiB when it starts; a start the tool refuses waits and retries.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
P40S="GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4 GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8"
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
W1=$OC/distill/parity/qtext-2026-10-12
W=$OC/distill/parity/qtext-2026-10-12-r4
R=$OC/distill/tps
ALLOC=$OC/composite/receipts/allocator
STEPLOG=$OC/distill/logs/qtext_r4_steps
mkdir -p "$W" "$STEPLOG"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
TOOL="$T/qtext_teacher_cache.py"
mem_of() { nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u=$1 '$1==u{print $2}'; }
pick() { for u in $P40S; do [ "$(mem_of $u)" -le 2000 ] && { echo $u; return 0; }; done; return 1; }
sha256sum $TOOL $T/qtext_vl_divergence_diag.py $OC/distill/tps_probe.py
cat > $ALLOC/qtext-round4-p40-claude.json <<EOF
{"name": "qtext-round4-p40-claude", "owner": "claude", "server": "ai", "kind": "host-venv",
 "gpus": "per step, any P40 of GPU1-3 at <= 2000 MiB (see steps)", "python": "$PY", "pid": $$,
 "launcher": "$T/run_qtext_p40_round4_2026-10-12.sh", "tool_sha256": "$(sha256sum $TOOL | cut -c1-64)",
 "purpose": "qtext lane final arms on tool 1.0.2", "started_utc": "$(date -u +%FT%TZ)", "state": "running", "steps": []}
EOF
record() {  # name card exit
  $PY - $ALLOC/qtext-round4-p40-claude.json "$1" "$2" "$3" <<'EOF3'
import json, sys, time
p, name, card, code = sys.argv[1:5]; d = json.load(open(p))
d["steps"].append({"step": name, "gpu_uuid": card, "exit": int(code), "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
json.dump(d, open(p, "w"), indent=1, sort_keys=True)
EOF3
}
run_step() {  # name, then the command (CUDA_VISIBLE_DEVICES is set per attempt)
  local name=$1; shift
  for attempt in $(seq 1 240); do
    card=$(pick) || { sleep 60; continue; }
    echo; echo "===== $name $(date -u +%FT%TZ) card $card used $(mem_of $card) MiB attempt $attempt"
    CUDA_VISIBLE_DEVICES=$card "$@" > $STEPLOG/$name.log 2>&1
    code=$?
    grep -v "Triton version\|fast path is not\|Loading weights" $STEPLOG/$name.log | grep -E "^\[|PARITY|verify |\"equal\"|\"max_abs\"|inv_freq|not_bit|Error" | cut -c1-420
    if grep -q "refusing to start" $STEPLOG/$name.log; then echo "card taken before start; retry"; sleep 60; continue; fi
    echo "exit $code"; record "$name" "$card" "$code"; return $code
  done
  echo "$name: no P40 within 4 h"; record "$name" none 99; return 99
}
RUN="nice -n 15 $PY $TOOL"

run_step 1-parity-text-08b-fp32 $RUN parity-text --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 4   --parity-rows 8 --parity-length 1024 --work-dir $W/text-fp32 --receipt $R/qtext-parity-text-08b-fp32-p40-v102.json
run_step 2-parity-vl-08b-fp32 $RUN parity-vl --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 2   --work-dir $W/vl-fp32 --receipt $R/qtext-parity-vl-08b-fp32-p40-v102.json
run_step 3-parity-video-08b-fp32 $RUN parity-vl --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 1 \
  --corpus $W1/video-src/video.jsonl --work-dir $W/video-fp32 --receipt $R/qtext-parity-video-08b-fp32-p40-v102.json
run_step 4-parity-text-08b-bf16-torchgdn $RUN parity-text --model-dir $M08 --device cuda:0 --dtype bfloat16 \
  --linear-attn-kernel torch --micro-batch 4 --parity-rows 8 --parity-length 1024 --work-dir $W/text-bf16-torch \
  --receipt $R/qtext-parity-text-08b-bf16-torchgdn-p40-v102.json
run_step 5-parity-vl-08b-bf16-torchgdn $RUN parity-vl --model-dir $M08 --device cuda:0 --dtype bfloat16 \
  --linear-attn-kernel torch --micro-batch 2 --work-dir $W/vl-bf16-torch --receipt $R/qtext-parity-vl-08b-bf16-torchgdn-p40-v102.json
run_step 6-vl-divergence-diag-v102 nice -n 15 $PY $T/qtext_vl_divergence_diag.py --model-dir $M08 \
  --corpus $W1/vl-fp32/parity_vl_corpus.jsonl --device cuda:0 --dtype float32 --out $R/qtext-vl-divergence-diag-08b-fp32-p40-v102.json
run_step 7-q27-build-text-smoke $RUN build-text --model-dir $M27 --teacher q38-27bu-smoke \
  --corpus $W1/q27-text-corpus.safetensors --out-root $W/q27-cache --device cuda:0 --dtype float32 --micro-batch 2 \
  --rows-per-shard 2 --head-chunk-rows 1024 --digest-threads 8 --purpose "qtext 27B build-text smoke fp32 r4"
run_step 8-q27-build-vl-smoke $RUN build-vl --model-dir $M27 --teacher q38-27bu-vl-smoke \
  --corpus $W1/vl-fp32/parity_vl_corpus.jsonl --out-root $W/q27-cache --device cuda:0 --dtype float32 --micro-batch 2 \
  --rows-per-shard 2 --head-chunk-rows 1024 --digest-threads 8 --purpose "T(i) 27B build-vl smoke fp32 r4"

echo; echo "===== 9 verify every round-4 cache (CPU)"
for d in $W/*/cache/*/*/ $W/q27-cache/*/*/; do CUDA_VISIBLE_DEVICES="" nice -n 15 $PY $TOOL verify --cache-dir "$d" | tail -1; done

$PY - $ALLOC/qtext-round4-p40-claude.json <<'EOF4'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p))
d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released (process exited)"})
json.dump(d, open(p, "w"), indent=1, sort_keys=True)
EOF4
echo "===== done $(date -u +%FT%TZ)"
