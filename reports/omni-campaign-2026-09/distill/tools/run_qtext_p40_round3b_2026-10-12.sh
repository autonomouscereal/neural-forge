#!/usr/bin/env bash
# qtext lane round 3b, 2026-10-12: steps 3-9 of round 3 (round 3 ran steps 1-2 and stopped at
# step 3 because another lane took GPU1).  Waits for a P40 (GPU1-3) holding <= 2000 MiB, then
# runs on it.  Same tool (1.0.1), same work dir, same receipts names.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
P40S="GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4 GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8"
CARD=""
for attempt in $(seq 1 240); do
  for u in $P40S; do
    m=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u=$u '$1==u{print $2}')
    if [ "$m" -le 2000 ]; then CARD=$u; break 2; fi
  done
  sleep 60
done
[ -z "$CARD" ] && { echo "no P40 at <= 2000 MiB within 4 h"; exit 3; }
echo "picked $CARD after $attempt checks"
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
W1=$OC/distill/parity/qtext-2026-10-12
W=$OC/distill/parity/qtext-2026-10-12-r3
R=$OC/distill/tps
ALLOC=$OC/composite/receipts/allocator
mkdir -p "$W"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
RUN="nice -n 15 $PY $T/qtext_teacher_cache.py"
used() { nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v u=$CARD '$1==u{print $2}'; }
step() { echo; echo "===== $1 $(date -u +%FT%TZ) card $CARD used $(used) MiB"; }
busy() { [ "$(used)" -gt 2000 ] && { echo "card holds $(used) MiB > 2000; stop"; exit 3; }; return 0; }
cat > $ALLOC/qtext-round3b-p40-claude.json <<EOF
{"name": "qtext-round3b-p40-claude", "owner": "claude", "server": "ai", "kind": "host-venv",
 "gpu_uuids": ["$CARD"], "python": "$PY", "pid": $$, "launcher": "$T/run_qtext_p40_round3b_2026-10-12.sh",
 "tool_sha256": "$(sha256sum $T/qtext_teacher_cache.py | cut -c1-64)",
 "purpose": "qtext lane final arms: parity-text/vl/video on the 0.8B, 27B build-text + build-vl smokes", "started_utc": "$(date -u +%FT%TZ)", "state": "running"}
EOF
sha256sum $T/qtext_teacher_cache.py $T/qtext_vl_divergence_diag.py $OC/distill/tps_probe.py
$PY - $ALLOC/qtext-round3-p40-claude.json <<'EOF2'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
if d.get("state") == "running":
    d.update({"ended_utc": "2026-10-13T01:39:15Z", "state": "released (launcher stopped itself at step 3: another lane put 9.7 GiB on the card)",
              "outcome": "steps 1-2 exit 0; steps 3-9 moved to qtext-round3b-p40-claude"})
    json.dump(d, open(p, "w"), indent=1, sort_keys=True)
EOF2

step "3 parity-vl 0.8B fp32, 8-frame video row"; busy
$RUN parity-vl --model-dir $M08 --device cuda:0 --dtype float32 --micro-batch 1 --corpus $W1/video-src/video.jsonl \
  --work-dir $W/video-fp32 --receipt $R/qtext-parity-video-08b-fp32-p40-v101.json --purpose "T(i) parity video 0.8B fp32"
echo "exit $?"

step "4 parity-text 0.8B bf16, torch GDN kernel (fla bf16 cannot compile for sm61)"; busy
$RUN parity-text --model-dir $M08 --device cuda:0 --dtype bfloat16 --linear-attn-kernel torch --micro-batch 4 \
  --parity-rows 8 --parity-length 1024 --work-dir $W/text-bf16-torch --receipt $R/qtext-parity-text-08b-bf16-torchgdn-p40-v101.json \
  --purpose "qtext parity-text 0.8B bf16 torch-GDN"
echo "exit $?"

step "5 parity-vl 0.8B bf16, torch GDN kernel"; busy
$RUN parity-vl --model-dir $M08 --device cuda:0 --dtype bfloat16 --linear-attn-kernel torch --micro-batch 2 \
  --work-dir $W/vl-bf16-torch --receipt $R/qtext-parity-vl-08b-bf16-torchgdn-p40-v101.json --purpose "T(i) parity-vl 0.8B bf16"
echo "exit $?"

step "6 VL divergence diagnostic on the final tool (expect every stage bit-equal)"; busy
nice -n 15 $PY $T/qtext_vl_divergence_diag.py --model-dir $M08 --corpus $W1/vl-fp32/parity_vl_corpus.jsonl \
  --device cuda:0 --dtype float32 --out $R/qtext-vl-divergence-diag-08b-fp32-p40-v101.json | grep -E '"equal"|vision_params_not|inv_freq' | head -30
echo "exit ${PIPESTATUS[0]}"

step "7 27B build-text smoke, fp32 on one P40, 4 rows x 512, 2 shards"; busy
$RUN build-text --model-dir $M27 --teacher q38-27bu-smoke --corpus $W1/q27-text-corpus.safetensors \
  --out-root $W/q27-cache --device cuda:0 --dtype float32 --micro-batch 2 --rows-per-shard 2 --head-chunk-rows 1024 \
  --digest-threads 8 --allocator-receipt $ALLOC/qtext-q27-text-smoke-r3-claude.json --purpose "qtext 27B build-text smoke fp32 r3"
echo "exit $?"

step "8 27B build-vl (T(i)) smoke, fp32, two drawn images"; busy
$RUN build-vl --model-dir $M27 --teacher q38-27bu-vl-smoke --corpus $W1/vl-fp32/parity_vl_corpus.jsonl \
  --out-root $W/q27-cache --device cuda:0 --dtype float32 --micro-batch 2 --rows-per-shard 2 --head-chunk-rows 1024 \
  --digest-threads 8 --allocator-receipt $ALLOC/qtext-q27-vl-smoke-r3-claude.json --purpose "T(i) 27B build-vl smoke fp32 r3"
echo "exit $?"

step "9 verify every round-3 cache"
for d in $W/*/cache/*/*/ $W/q27-cache/*/*/; do $RUN verify --cache-dir "$d" | tail -1; done

$PY - $ALLOC/qtext-round3b-p40-claude.json <<'EOF'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p))
d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released (process exited)"})
json.dump(d, open(p, "w"), indent=1, sort_keys=True)
EOF
step "done"
