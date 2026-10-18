#!/bin/bash
# Lane E3 (2026-10-10): c17 charts for trunk layers 42-63 (K3 MoE layers 61-92 under 92:64, dense L0 skipped),
# now that the K3 dense fold run k3-run-oct09-claude covers layers 0-92.  Same arguments as launch_c17_0_41.sh except
# --k3-pass-receipt (the full run's receipt, prefix carried forward unchanged) and the CPU caps: 3 detached workers x
# THREADS (default 10, total <= 32), nice 10, OMP/MKL/OPENBLAS threads pinned -- on 2026-10-10 three uncapped
# 24-thread workers took 66 of 88 CPUs and starved another lane.  Logs go to c17_w{i}_42_63.log so the L00-41
# worker logs stay as evidence.  Afterwards MODE=merge re-verifies every layer 0-63 by sha256 against the farm
# (all reused, nothing refitted) and rebuilds receipt.json over L00-63; the L00-41 receipt is kept beside it as
# receipt_L00-41.json.
set -u
OC=/home/cereal/omni-campaign
C17=$OC/composite/c17
PY=/home/cereal/model-surgery-lab/venv/bin/python
TAPS=$C17/trunk_taps/s3native-c2k4352-held-all
OUT=$C17/charts
K3_RECEIPT=$OC/k3-run-oct09-claude/receipts/k3_dense_pass_receipt.json
MODE=${1:-fit}
THREADS=${THREADS:-10}
WORKERS=3
if [ $((THREADS * WORKERS)) -gt 32 ]; then echo "REFUSE: $WORKERS x $THREADS threads > 32"; exit 2; fi
if [ ! -f "$TAPS/taps_receipt.json" ]; then echo "REFUSE: no taps receipt at $TAPS"; exit 3; fi
if [ ! -f "$C17/k3_rows/farm_manifest.json" ]; then echo "REFUSE: no K3 rows farm manifest"; exit 3; fi
if [ ! -f "$OUT/receipt_L00-41.json" ]; then echo "REFUSE: keep the L00-41 receipt as $OUT/receipt_L00-41.json first"; exit 3; fi
want=$(sha256sum "$K3_RECEIPT" | cut -d' ' -f1)
pinned=$($PY -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["pinned_receipt"]["sha256"], len(m["refused_layers"]), len(m["layers"]))' "$C17/k3_rows/farm_manifest.json")
if [ "$pinned" != "$want 0 92" ]; then echo "REFUSE: farm manifest is '$pinned', want '$want 0 92' (run scripts/extend_k3_rows_farm.py)"; exit 3; fi
free_g=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_g" -lt 70 ]; then echo "REFUSE: root has ${free_g}G free (22 charts need ~6.5G; floor 60G)"; exit 4; fi
COMMON=(--k3-resid $C17/k3_rows
  --k3-pass-receipt $K3_RECEIPT
  --trunk-taps $TAPS --trunk-rows-key normed
  --evidence-manifest $OC/composite/corpus/omni-evidence-27b-v1.json --tap-micro-batch 1
  --boundary-map $OC/composite/corpus/boundary_map.json
  --alignment 92:64 --dense-layer0 skip --skip-control-charts --allow-root-disk
  --threads $THREADS --out $OUT)
CAPS=(env OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS nice -n 10)
mkdir -p $C17/logs
if [ "$MODE" = merge ]; then
  setsid nohup "${CAPS[@]}" /usr/bin/time -v $PY -u $OC/composite/tools/c17_chart_fit.py "${COMMON[@]}" --layers 0-63 > $C17/logs/c17_merge_0_63.log 2>&1 < /dev/null &
  echo "merge pid $!"; exit 0
fi
# balanced on pool size: two-K3-layer pools (43 45 47 50 52 54 57 59 61 63) cost ~1.5x a one-layer pool
W0=42,45,48,51,54,57,60,62
W1=43,46,49,52,55,58,61
W2=44,47,50,53,56,59,63
i=0
for W in $W0 $W1 $W2; do
  setsid nohup "${CAPS[@]}" /usr/bin/time -v $PY -u $OC/composite/tools/c17_chart_fit.py "${COMMON[@]}" --layers $W > $C17/logs/c17_w${i}_42_63.log 2>&1 < /dev/null &
  echo "worker $i pid $! layers $W threads $THREADS"
  i=$((i+1))
done
echo "root free ${free_g}G; K3 receipt $want"
