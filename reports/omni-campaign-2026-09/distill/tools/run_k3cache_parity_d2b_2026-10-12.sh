#!/bin/bash
# k3cache lane, parity diagnosis D2b: D2 again with --row-chunk-kda 1 (the reference's own row chunking) on the
# same P40.  D2b vs reference says what the KDA row batching contributes; D2b vs D2 is this tool's own noise
# floor on one card when only the batching changes.  GPU3 shared: wait for <= 2000 MiB, retry tenant OOMs.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
OUT=$OC/distill/parity/k3cache-2026-10-12
LOG=$OC/distill/logs/k3cache
SNAP=/mnt/lexar/omni-campaign/k3-continuation-oct08-claude/fold/snapshots/before_L61.safetensors
ALLOC=$OC/composite/receipts/allocator/k3cache-parity-d2b-gpu3-claude.json
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RC=1
for attempt in 1 2 3 4 5 6; do
  quiet=0
  while [ $quiet -lt 2 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ "$used" -le 2000 ]; then quiet=$((quiet+1)); else quiet=0; fi
    [ $quiet -ge 2 ] || sleep 30
  done
  python3 -c "
import json,os,time; p='$ALLOC'; r=json.load(open(p)) if os.path.exists(p) else {'lane':'k3cache parity diagnosis D2b (row chunk 1)','gpu':3,'card':'Tesla P40','by':'claude k3cache lane','log':'$LOG/parity_d2b.log'}
s=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()); r.setdefault('started_utc',s); r.setdefault('notes',[]).append(s+' attempt $attempt GPU3 $used MiB'); json.dump(r,open(p,'w'),indent=1)"
  echo "== D2b attempt $attempt $(date -u +%FT%TZ)" >> "$LOG/parity_d2b.log"
  nice -n 15 "$PP" "$T/k3_teacher_cache.py" run \
    --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
    --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
    --probe --layers 61-92 --start-snapshot "$SNAP" \
    --snapshot-layout "k3_review_input_ids=98304:512,k3_review_native_input_ids=122880:512" \
    --pass-dir "$OUT/d2b-from-L61-rowchunk1" --device cuda:0 --experts-device cuda:0 \
    --row-chunk-kda 1 --row-chunk-mla 1 --group-tokens 32768 >> "$LOG/parity_d2b.log" 2>&1
  RC=$?
  echo "D2b attempt $attempt exit=$RC $(date -u +%FT%TZ)"
  [ $RC -eq 0 ] && break
  tail -3 "$LOG/parity_d2b.log" | grep -q "OutOfMemoryError" || break
  sleep 60
done
if [ $RC -eq 0 ]; then
  nice -n 15 "$PP" "$T/k3_teacher_cache.py" parity --cache-dir "$OUT/d2b-from-L61-rowchunk1" \
    --reference "$OC/k3-run-oct09-claude/fold/review/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$OC/k3-run-oct09-claude/fold/review/k3_review_native_input_ids_teacher_forced.safetensors" \
    --out "$OUT/parity_d2b_receipt.json" > "$LOG/parity_d2b_compare.log" 2>&1
  echo "D2b_PARITY_EXIT=$? $(date -u +%FT%TZ)"
fi
python3 -c "import json,time;p='$ALLOC';r=json.load(open(p));r['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());r['exit']=$RC;json.dump(r,open(p,'w'),indent=1)"
echo "D2b end $(date -u +%FT%TZ)"
