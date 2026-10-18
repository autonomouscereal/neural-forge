#!/bin/bash
# Lane E3 (2026-10-09): c17 charts for trunk layers 0-41 (every aligned K3 layer <= 60 under 92:64, dense L0 skipped).
# CPU fp64, 3 detached workers over disjoint layer sets into ONE --out; afterwards run with MODE=merge to rebuild
# receipt.json from every L*/layer.json (all reused, verified by sha256).
set -u
OC=/home/cereal/omni-campaign
C17=$OC/composite/c17
PY=/home/cereal/model-surgery-lab/venv/bin/python
TAPS=$C17/trunk_taps/s3native-c2k4352-held-all
OUT=$C17/charts
MODE=${1:-fit}
THREADS=${THREADS:-24}
if [ ! -f "$TAPS/taps_receipt.json" ]; then echo "REFUSE: no taps receipt at $TAPS"; exit 3; fi
if [ ! -f "$C17/k3_rows/farm_manifest.json" ]; then echo "REFUSE: no K3 rows farm manifest"; exit 3; fi
free_g=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_g" -lt 110 ]; then echo "REFUSE: root has ${free_g}G free (charts need ~12.3G; floor 80G + K3 run growth)"; exit 4; fi
COMMON=(--k3-resid $C17/k3_rows
  --k3-pass-receipt /mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-03603908-570d-4c90-96cb-ee3a63a9a139/model/receipts/k3_dense_pass_receipt.json
  --trunk-taps $TAPS --trunk-rows-key normed
  --evidence-manifest $OC/composite/corpus/omni-evidence-27b-v1.json --tap-micro-batch 1
  --boundary-map $OC/composite/corpus/boundary_map.json
  --alignment 92:64 --dense-layer0 skip --skip-control-charts --allow-root-disk
  --threads $THREADS --out $OUT)
mkdir -p $C17/logs
if [ "$MODE" = merge ]; then
  setsid nohup /usr/bin/time -v $PY -u $OC/composite/tools/c17_chart_fit.py "${COMMON[@]}" --layers 0-41 > $C17/logs/c17_merge_0_41.log 2>&1 < /dev/null &
  echo "merge pid $!"; exit 0
fi
W0=0,3,6,9,12,15,18,21,24,27,30,33,36,39
W1=1,4,7,10,13,16,19,22,25,28,31,34,37,40
W2=2,5,8,11,14,17,20,23,26,29,32,35,38,41
i=0
for W in $W0 $W1 $W2; do
  setsid nohup /usr/bin/time -v $PY -u $OC/composite/tools/c17_chart_fit.py "${COMMON[@]}" --layers $W > $C17/logs/c17_w$i.log 2>&1 < /dev/null &
  echo "worker $i pid $! layers $W"
  i=$((i+1))
done
