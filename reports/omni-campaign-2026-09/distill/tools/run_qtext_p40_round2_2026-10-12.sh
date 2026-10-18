#!/usr/bin/env bash
# qtext lane round 2, 2026-10-12, on P40 GPU1 (GPU3 was taken mid-run by another lane's
# test: round 1 step 5 OOM'd beside an 11.16 GiB foreign process).  Steps:
#   a) VL divergence diagnostic (0.8B, fp32): where T(i) and the stock VL forward part bitwise;
#   b) the same diagnostic with cudnn.deterministic, to separate kernel choice from wiring;
#   c) the 27B build-text smoke rerun (fp32, head chunk 1024), then verify.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
G1=GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
W=$OC/distill/parity/qtext-2026-10-12
ALLOC=$OC/composite/receipts/allocator
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
step() { echo; echo "===== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 2p; }
busy() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); [ "$u" -gt 2000 ] && { echo "GPU1 holds $u MiB > 2000; stop"; exit 3; }; return 0; }
cat > $ALLOC/qtext-round2-p40-gpu1-claude.json <<EOF
{"name": "qtext-round2-p40-gpu1-claude", "owner": "claude", "server": "ai", "kind": "host-venv",
 "gpus": [1], "gpu_uuids": ["$G1"], "python": "$PY", "pid": $$, "launcher": "$T/run_qtext_p40_round2_2026-10-12.sh",
 "purpose": "qtext lane: T(i) divergence diagnostic + 27B build-text smoke rerun", "started_utc": "$(date -u +%FT%TZ)", "state": "running"}
EOF

step "a VL divergence diagnostic fp32"; busy
nice -n 15 $PY $T/qtext_vl_divergence_diag.py --model-dir $M08 --corpus $W/vl-fp32/parity_vl_corpus.jsonl \
  --device cuda:0 --dtype float32 --out $OC/distill/tps/qtext-vl-divergence-diag-08b-fp32-p40.json
echo "exit $?"

step "b VL divergence diagnostic fp32, cudnn.deterministic"; busy
nice -n 15 $PY $T/qtext_vl_divergence_diag.py --model-dir $M08 --corpus $W/vl-fp32/parity_vl_corpus.jsonl \
  --device cuda:0 --dtype float32 --deterministic --out $OC/distill/tps/qtext-vl-divergence-diag-08b-fp32-det-p40.json
echo "exit $?"

step "c 27B build-text smoke rerun fp32"; busy
# round 1 left cache_config.json + one session record (OOM, digests equal) and no shard: resume over it.
nice -n 15 $PY $T/qtext_teacher_cache.py build-text --model-dir $M27 --teacher q38-27bu-smoke \
  --corpus $W/q27-text-corpus.safetensors --out-root $W/q27-cache --device cuda:0 --dtype float32 \
  --micro-batch 2 --rows-per-shard 2 --head-chunk-rows 1024 --digest-threads 8 \
  --allocator-receipt $ALLOC/qtext-q27-text-smoke-r2-p40-gpu1-claude.json --purpose "qtext 27B build-text smoke fp32 r2"
echo "exit $?"
for d in $W/q27-cache/q38-27bu-smoke/*/; do nice -n 15 $PY $T/qtext_teacher_cache.py verify --cache-dir "$d"; echo "exit $?"; done

$PY - $ALLOC/qtext-round2-p40-gpu1-claude.json <<'EOF'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p))
d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released (process exited)"})
json.dump(d, open(p, "w"), indent=1, sort_keys=True)
EOF
step "done"
