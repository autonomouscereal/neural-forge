#!/bin/bash
# run_b2_geometry_diag_gpu6_2026-10-14.sh -- lane gpu6, plan Step 2a: k3_b2_geometry_diag over the fold's three files,
# L0-L2, five arms (see the tool's docstring), on GPU6, detached:
#   setsid nohup bash $OC/distill/tools/run_b2_geometry_diag_gpu6_2026-10-14.sh \
#     > $OC/distill/queues/logs/gpu6/b2_geometry_diag.driver.log 2>&1 < /dev/null &
# Waits for GPU6 < 2000 MiB (two reads 30 s apart) and MemAvailable >= 70 GiB (inside the tool); never competes.
# Thermal: telemetry every 60 s; SIGTERM only at shutdown - 2 C (the abort line), never a gate.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
Q=$OC/distill/queues
QL=$Q/logs/gpu6
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=6
C=$OC/composite/corpus
OUT=/mnt/lexar/omni-campaign/distill/k3-b2-geometry-diag-2026-10-14
RCPT=$OC/distill/parity/k3cache-2026-10-14-b2-geometry-diag
LOG=$QL/b2_geometry_diag.log
TH=$QL/b2_geometry_diag.thermal.log
ALLOC=$OC/composite/receipts/allocator/gpu6-b2diag-claude.json
mkdir -p "$QL" "$OUT" "$RCPT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
utc() { date -u +%FT%TZ; }
echo "driver start $(utc) tool $(sha256sum $T/k3_b2_geometry_diag.py | cut -c1-16) frozen $(sha256sum $Q/frozen/k3_teacher_cache-gpu6.py | cut -c1-16)"
q=0; while [ $q -lt 2 ]; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  if [ "$u" -lt 2000 ]; then q=$((q + 1)); else q=0; echo "$(utc) GPU$GPU holds $u MiB; waiting"; fi
  [ $q -ge 2 ] || sleep 30
done
SHUT=$(nvidia-smi -q -d TEMPERATURE -i $GPU | awk -F: '/GPU Shutdown Temp/ {gsub(/[^0-9]/, "", $2); print $2; exit}')
HARD=$(( ${SHUT:-96} - 2 ))
echo "{\"job\":\"b2-geometry-diag\",\"gpu\":$GPU,\"by\":\"claude gpu6 lane\",\"started_utc\":\"$(utc)\",\"log\":\"$LOG\"}" > "$ALLOC"
echo "$(utc) b2-diag START k3_b2_geometry_diag $(sha256sum $T/k3_b2_geometry_diag.py | cut -c1-16) log $LOG out $OUT" >> $Q/gpu6.status
CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$T/k3_b2_geometry_diag.py" \
  --input-ids "$C/k3_input_ids.safetensors" --input-ids "$C/k3_review_input_ids.safetensors" \
  --input-ids "$C/k3_review_native_input_ids.safetensors" --out "$OUT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$(utc) diag pid $PID; thermal abort line ${HARD} C (shutdown ${SHUT} C)" | tee -a "$TH"
peak=0
while [ -d /proc/$PID ]; do
  line=$(nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active,utilization.gpu,memory.used,power.draw \
         --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
  t=${line%%,*}
  case "$t" in ''|*[!0-9]*) ;; *)
    [ "$t" -gt "$peak" ] && peak=$t
    echo "$(utc) $line MemAvailable=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)GiB" >> "$TH"
    if [ "$t" -ge "$HARD" ]; then echo "$(utc) GPU$GPU ${t} C >= ${HARD} C: SIGTERM $PID" | tee -a "$TH" >> "$LOG"; kill -TERM $PID; fi;;
  esac
  sleep 60
done
wait $PID; RC=$?
echo "$(utc) diag exit=$RC peak ${peak} C"
cp -p "$OUT/b2_geometry_diag.json" "$RCPT/" 2>/dev/null && sha256sum "$RCPT/b2_geometry_diag.json"
echo "{\"job\":\"b2-geometry-diag\",\"gpu\":$GPU,\"by\":\"claude gpu6 lane\",\"ended_utc\":\"$(utc)\",\"exit\":$RC,\"log\":\"$LOG\"}" > "$ALLOC"
echo "$(utc) b2-diag END exit=$RC $(grep -o 'VERDICT .*' "$LOG" | tail -1 | head -c 400)" >> $Q/gpu6.status
echo "driver end $(utc)"
