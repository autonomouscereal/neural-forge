#!/bin/bash
# k3cache lane (k3_teacher_cache.py): the full 93-layer review parity pass + its compare, then the S0 P3
# probe, on ONE P40 (GPU3) in the Pascal venv (the lab venv's torch 2.11+cu128 has no sm_6x).
# GPU3 is shared by every lane's tests: each step waits until GPU3 holds <= 2000 MiB, and the pass is
# retried with --resume (layer checkpoints) if another tenant's allocation OOMs it mid-run.
# Run detached:  setsid nohup bash run_k3cache_p3_parity_2026-10-12.sh > $OC/distill/logs/k3cache/driver.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
OUT=$OC/distill/parity/k3cache-2026-10-12
P3=$OC/distill/tps/k3cache_p3_p40_2026-10-12
LOG=$OC/distill/logs/k3cache
ALLOC=$OC/composite/receipts/allocator/k3cache-p3-parity-gpu3-claude.json
mkdir -p "$OUT" "$P3" "$LOG"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
alloc() {  # $1 = started|ended|waited, $2 = note
  python3 - "$ALLOC" "$1" "$2" <<'PY'
import json, os, sys, time
path, what, note = sys.argv[1:4]
rec = json.load(open(path)) if os.path.exists(path) else {
    "lane": "k3cache (distill/tools/k3_teacher_cache.py): review parity pass + S0 P3 probe", "gpu": 3,
    "card": "Tesla P40", "by": "claude k3cache lane", "venv": "/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260",
    "log": "/home/cereal/omni-campaign/distill/logs/k3cache/"}
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rec[f"{what}_utc"] = stamp
rec.setdefault("notes", []).append(f"{stamp} {note}")
json.dump(rec, open(path, "w"), indent=1)
PY
}
wait_gpu3() {  # until GPU3 holds <= 2000 MiB on two reads 30 s apart (max 3 h)
  local quiet=0 waited=0
  while [ $quiet -lt 2 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$used" -le 2000 ]; then quiet=$((quiet+1)); else quiet=0; fi
    [ $quiet -ge 2 ] && break
    sleep 30; waited=$((waited+30))
    if [ $waited -ge 10800 ]; then echo "REFUSED: GPU3 busy for 3 h"; alloc ended "gave up waiting"; exit 9; fi
  done
  echo "GPU3 free ($used MiB) after ${waited}s $(date -u +%FT%TZ)"
}
echo "driver start $(date -u +%FT%TZ) tool_sha256=$(sha256sum $T/k3_teacher_cache.py | cut -c1-64)"
RC=1
for attempt in 1 2 3 4 5; do
  wait_gpu3
  alloc started "parity pass attempt $attempt"
  echo "== parity pass attempt $attempt $(date -u +%FT%TZ)"
  nice -n 15 "$PP" "$T/k3_teacher_cache.py" run \
    --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
    --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
    --pass-dir "$OUT/review-pass" --device cuda:0 --experts-device cuda:0 \
    --row-chunk-kda 8 --row-chunk-mla 1 --group-tokens 32768 --inline-digest --resume \
    >> "$LOG/parity_pass.log" 2>&1
  RC=$?
  echo "PASS_EXIT=$RC $(date -u +%FT%TZ)"
  [ $RC -eq 0 ] && break
  grep -q "OutOfMemoryError" <(tail -5 "$LOG/parity_pass.log") || break   # only a tenant OOM is retried
done

if [ $RC -eq 0 ]; then
  echo "== parity compare $(date -u +%FT%TZ)"
  nice -n 15 "$PP" "$T/k3_teacher_cache.py" parity --cache-dir "$OUT/review-pass" \
    --reference "$OC/k3-run-oct09-claude/fold/review/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$OC/k3-run-oct09-claude/fold/review/k3_review_native_input_ids_teacher_forced.safetensors" \
    --out "$OUT/parity_receipt.json" > "$LOG/parity_compare.log" 2>&1
  echo "PARITY_EXIT=$? $(date -u +%FT%TZ)"
fi

wait_gpu3
echo "== P3 $(date -u +%FT%TZ)"
alloc started "P3 probe"
# KDA torch-backend transients are ~0.4 MB per token per call (fp32 q/k/v/g [B,T,96,128]): with the
# 16.6 GB packed layer resident only ~4k tokens per KDA call fit a 24 GB card -> row chunk 2 at T=2048
CUDA_LAUNCH_BLOCKING=1 nice -n 15 "$PP" "$T/k3_teacher_cache.py" run \
  --input-ids "$OC/composite/corpus/k3_input_ids.safetensors" \
  --probe --layers 1-3 --synthetic-blocks 8 --no-head --row-limit 24 --truncate-seq 2048 \
  --group-tokens 16384 --row-chunk-kda 2 --row-chunk-mla 1 \
  --device cuda:0 --experts-device cuda:0 --no-prefetch-next --pass-dir "$P3" > "$LOG/p3.log" 2>&1
echo "P3_EXIT=$? $(date -u +%FT%TZ)"
alloc ended "driver finished pass_exit=$RC"
echo "driver end $(date -u +%FT%TZ)"
