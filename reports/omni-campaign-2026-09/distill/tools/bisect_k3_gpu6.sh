#!/bin/bash
# bisect_k3_gpu6.sh -- lane gpu6, ONLY after a B2 FAIL: where does k3_teacher_cache's teacher stream leave the fold
# engine's?  Runs the frozen K3 tool as a PROBE over the fold's OWN three input files in the fold's order and layout
# (24 x 4,096 evidence rows, then the 48 + 40 review rows at T = 512, --no-trim: token offsets 0 / 98,304 /
# 122,880 exactly as the fold), layers 0..LAST (default 60), with --prefix-digests, and compares every layer's
# prefix sha256 with the fold receipt's teacher prefix_out_sha256 (k3_dense_pass sha256_tensor over the same
# [143,360, 7,168] tensor).  Equal = bit-identical teacher state after that layer; the first unequal layer is where
# the two diverge.  Diagnostic only; nothing is written outside $OUT.
#   setsid nohup bash bisect_k3_gpu6.sh [LAST] > $OC/distill/queues/logs/gpu6/bisect.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
PY=/home/cereal/model-surgery-lab/venv/bin/python
K3TOOL=$Q/frozen/k3_teacher_cache-gpu6.py
GPU=6
LAST=${1:-60}
OUT=$OC/distill/parity/k3cache-2026-10-13-bisect-gpu6
FOLD=$OC/k3-run-oct09-claude/receipts/k3_dense_pass_receipt.json
C=$OC/composite/corpus
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONPATH=$OC/distill/tools
echo "bisect start $(date -u +%FT%TZ) layers 0-$LAST tool $(sha256sum $K3TOOL | cut -c1-16)"
while pgrep -f "distill/queues/gpu6.sh" >/dev/null && pgrep -f "k3_teacher_cache-gpu6.py run" >/dev/null; do sleep 60; done
q=0; while [ $q -lt 2 ]; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU); [ "$u" -lt 2000 ] && q=$((q+1)) || q=0; [ $q -ge 2 ] || sleep 30; done
echo "{\"job\":\"bisect\",\"gpu\":$GPU,\"by\":\"claude gpu6 lane\",\"started_utc\":\"$(date -u +%FT%TZ)\",\"log\":\"$Q/logs/gpu6/bisect.log\"}" \
  > $OC/composite/receipts/allocator/gpu6-bisect-claude.json
CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$K3TOOL" run --probe --layers 0-$LAST --no-head --prefix-digests --no-trim \
  --input-ids "$C/k3_input_ids.safetensors" --input-ids "$C/k3_review_input_ids.safetensors" \
  --input-ids "$C/k3_review_native_input_ids.safetensors" --pass-dir "$OUT/probe" \
  --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
  --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --group-tokens 32768
RC=$?
echo "probe exit=$RC $(date -u +%FT%TZ)"
python3 - "$OUT/probe/probe_receipt.json" "$FOLD" "$OUT/bisect_sha_compare.json" <<'PY'
import json, sys
mine, fold, out = sys.argv[1:4]
m = json.load(open(mine))["layers"]
f = json.load(open(fold))["layers"]
rows, first = [], None
for k in sorted(m, key=int):
    a = m[k].get("prefix_out_sha256")
    b = (f.get(k, {}).get("prefix_out_sha256") or {}).get("teacher")
    same = a is not None and a == b
    rows.append({"layer": int(k), "equal": same, "mine": a, "fold_teacher": b})
    if not same and first is None:
        first = int(k)
res = {"layers": rows, "first_unequal_layer": first, "all_equal": first is None,
       "note": "sha256_tensor of the teacher prefix after each layer over the fold's [143360, 7168] corpus"}
json.dump(res, open(out, "w"), indent=1)
print("BISECT", json.dumps({"layers_compared": len(rows), "first_unequal_layer": first,
                            "equal_layers": sum(r["equal"] for r in rows)}))
PY
echo "{\"job\":\"bisect\",\"gpu\":$GPU,\"by\":\"claude gpu6 lane\",\"ended_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$RC}" \
  > $OC/composite/receipts/allocator/gpu6-bisect-claude.json
echo "bisect end $(date -u +%FT%TZ)"
