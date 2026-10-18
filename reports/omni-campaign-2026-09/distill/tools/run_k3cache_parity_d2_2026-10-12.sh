#!/bin/bash
# k3cache lane, after the full review parity pass failed its pre-registered gates (parity_receipt.json):
#  D2  start from the fold engine's OWN teacher pair entering L61 (before_L61.safetensors of the 61-92 window,
#      the input state of the reference review dump), run L61-92 + the head and compare with
#      fold/review/*_teacher_forced.safetensors -- separates "this tool's window + head on a P40" from
#      "drift accumulated through L00-60";
#  P3  the S0 pure-forward probe (read / dequant / compute split, RSS with 8 AttnRes blocks resident).
# Both on GPU3 (P40, Pascal venv).  GPU3 is shared by every lane's tests: each attempt waits for <= 2000 MiB
# on two reads 30 s apart and an attempt that OOMs on another tenant's allocation is retried (max 6).
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
OUT=$OC/distill/parity/k3cache-2026-10-12
P3=$OC/distill/tps/k3cache_p3_p40_2026-10-12
LOG=$OC/distill/logs/k3cache
SNAP=/mnt/lexar/omni-campaign/k3-continuation-oct08-claude/fold/snapshots/before_L61.safetensors
ALLOC=$OC/composite/receipts/allocator/k3cache-parity-d2-p3-gpu3-claude.json
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() {
  python3 - "$ALLOC" "$1" <<'PY'
import json, os, sys, time
path, note = sys.argv[1:3]
rec = json.load(open(path)) if os.path.exists(path) else {
    "lane": "k3cache parity diagnosis D2 (L61-92 + head from the reference snapshot) + S0 P3 probe", "gpu": 3,
    "card": "Tesla P40", "by": "claude k3cache lane", "venv": "/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260",
    "log": "/home/cereal/omni-campaign/distill/logs/k3cache/"}
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rec.setdefault("started_utc", stamp)
rec["last_utc"] = stamp
rec.setdefault("notes", []).append(f"{stamp} {note}")
json.dump(rec, open(path, "w"), indent=1)
PY
}
wait_gpu3() {
  local quiet=0
  while [ $quiet -lt 2 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$used" -le 2000 ]; then quiet=$((quiet+1)); else quiet=0; fi
    [ $quiet -ge 2 ] || sleep 30
  done
}
attempts() {  # $1 = label, $2 = log, rest = command
  local label=$1 log=$2; shift 2
  local rc=1
  for attempt in 1 2 3 4 5 6; do
    wait_gpu3
    note "$label attempt $attempt (GPU3 ${used} MiB)"
    echo "== $label attempt $attempt $(date -u +%FT%TZ)" >> "$log"
    "$@" >> "$log" 2>&1
    rc=$?
    echo "$label attempt $attempt exit=$rc $(date -u +%FT%TZ)"
    [ $rc -eq 0 ] && break
    tail -3 "$log" | grep -q "OutOfMemoryError" || break
    sleep 60
  done
  return $rc
}
echo "driver start $(date -u +%FT%TZ) tool_sha256=$(sha256sum $T/k3_teacher_cache.py | cut -c1-64)"
attempts D2 "$LOG/parity_d2.log" nice -n 15 "$PP" "$T/k3_teacher_cache.py" run \
  --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
  --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
  --probe --layers 61-92 --start-snapshot "$SNAP" \
  --snapshot-layout "k3_review_input_ids=98304:512,k3_review_native_input_ids=122880:512" \
  --pass-dir "$OUT/d2-from-L61" --device cuda:0 --experts-device cuda:0 \
  --row-chunk-kda 8 --row-chunk-mla 1 --group-tokens 32768
RC=$?
if [ $RC -eq 0 ]; then
  nice -n 15 "$PP" "$T/k3_teacher_cache.py" parity --cache-dir "$OUT/d2-from-L61" \
    --reference "$OC/k3-run-oct09-claude/fold/review/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$OC/k3-run-oct09-claude/fold/review/k3_review_native_input_ids_teacher_forced.safetensors" \
    --out "$OUT/parity_d2_receipt.json" > "$LOG/parity_d2_compare.log" 2>&1
  echo "D2_PARITY_EXIT=$? $(date -u +%FT%TZ)"
fi
# KDA torch-backend transients are ~0.4 MB per token per call (fp32 q/k/v/g [B,T,96,128]): with the
# 16.6 GB packed layer resident only ~4k tokens per KDA call fit a 24 GB card -> row chunk 2 at T=2048
attempts P3 "$LOG/p3.log" env CUDA_LAUNCH_BLOCKING=1 nice -n 15 "$PP" "$T/k3_teacher_cache.py" run \
  --input-ids "$OC/composite/corpus/k3_input_ids.safetensors" \
  --probe --layers 1-3 --synthetic-blocks 8 --no-head --row-limit 24 --truncate-seq 2048 \
  --group-tokens 16384 --row-chunk-kda 2 --row-chunk-mla 1 \
  --device cuda:0 --experts-device cuda:0 --no-prefetch-next --pass-dir "$P3"
echo "P3 exit=$? $(date -u +%FT%TZ)"
note "driver finished d2_exit=$RC"
python3 -c "import json,time;p='$ALLOC';r=json.load(open(p));r['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());json.dump(r,open(p,'w'),indent=1)"
echo "driver end $(date -u +%FT%TZ)"
