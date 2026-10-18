#!/usr/bin/env bash
# Toolcall lane, step 2: native tool-use self-gen on the P40s (GPU1 + GPU2; GPU3 stays free).
#
#   1. wait for run_toolcall_export_2026-10-14.sh's export-check receipt (the parent's own Q4_K_M)
#   2. serve it on GPU1 / GPU2 (selfgen_sampler.py serve: card <= 2,000 MiB, launch + allocator receipts)
#   3. eog-probe both servers (the EOG table toolcall_selfgen relies on)
#   4. bank-build the held-out bank (frozen: never overwritten with different bytes), then the
#      parent-Q4 baseline: bank-run (greedy) + bank-grade
#   5. schemas (parent-synthesized, 48 domains x 3) -> plan (2,400 items)
#   6. smoke: 12k tokens, finalize --accept-short + export; gate = acceptance >= 20 %
#   7. production run to 1.5M rendered tokens -> finalize -> export
#   8. teardown ONLY after a sealed export: stop both servers through their receipts, rm the Q4
#      from declared scratch.  On any failure nothing is torn down (inspect / resume).
#
# Host-RAM gate before serving: MemAvailable >= 30 GiB (wait, never compete).
#
#   setsid nohup bash run_toolcall_selfgen_2026-10-14.sh > $OC/distill/toolcall/logs/driver.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
T=$OC/distill/tools
TC=$OC/distill/toolcall
SCRATCH=/mnt/lexar/omni-campaign/scratch/toolcall-q38-2026-10-14
CHECK=$TC/receipts/export_check_Q38-27BU-parent_toolcall.json
BANK=$OC/calibration/banks/omni_toolcall_bank_v1.json
SCHEMAS=$TC/schemas-2026-10-14
PLAN=$TC/plan-2026-10-14.jsonl
SMOKE=$TC/smoke-q38-q4km-2026-10-14
RUN=$TC/run-q38-q4km-2026-10-14
DEST=$TC/rescore/toolcall-q38-2026-10-14.safetensors
S1=http://127.0.0.1:18321
S2=http://127.0.0.1:18322
TARGET=1500000
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
mkdir -p "$TC/logs" "$TC/receipts" "$TC/bank-runs" "$TC/rescore"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
die() { stamp "STOP rc=$1: $2 (nothing torn down)"; exit "$1"; }

stamp "driver start; toolcall_selfgen $(sha256sum $T/toolcall_selfgen.py | cut -c1-16)"
while [ ! -s "$CHECK" ]; do
  if ! pgrep -f run_toolcall_export_2026-10-14.sh > /dev/null; then die 3 "export driver gone and no export-check receipt"; fi
  sleep 60
done
Q4=$("$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); assert r['parent_unchanged']; print(r['q4']['path'])" "$CHECK") || die 3 "export-check receipt unreadable or parent moved"
Q4SHA=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['q4']['sha256'])" "$CHECK")
stamp "Q4 $Q4 sha $Q4SHA"

while :; do
  avail_gib=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
  [ "$avail_gib" -ge 30 ] && break
  stamp "waiting: MemAvailable ${avail_gib} GiB < 30 GiB"; sleep 120
done

for g in 1 2; do
  port=$((18320 + g))
  if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then stamp "server on :$port already up"; continue; fi
  "$PY" "$T/selfgen_sampler.py" serve --gguf "$Q4" --gpu-index $g --allowed-gpus 1,2 --port $port \
    --name toolcall-q38-q4km-gpu$g --slots 4 --ctx-per-slot 8192 \
    --log-dir "$TC/logs" --receipt-dir "$TC/receipts" \
    --purpose "toolcall self-gen sampler: parent Q4_K_M llama-server (DISTILL_DESIGN 3 self-gen style, U7); claim in queues/p40.claims" \
    || die 4 "serve GPU$g failed"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

"$PY" "$T/selfgen_sampler.py" eog-probe --server $S1 --server $S2 --out "$TC/receipts/eog_probe_toolcall.json" || die 5 "eog-probe failed"

if [ ! -s "$BANK" ]; then
  "$PY" "$T/toolcall_selfgen.py" bank-build --out "$BANK" || die 6 "bank-build failed"
fi
sha256sum "$BANK"
if [ ! -s "$TC/bank-runs/parent-q4km.grade.json" ]; then
  "$PY" "$T/toolcall_selfgen.py" bank-run --bank "$BANK" --server $S1 --server $S2 --out "$TC/bank-runs/parent-q4km.responses.jsonl" || die 6 "bank-run failed"
  "$PY" "$T/toolcall_selfgen.py" bank-grade --bank "$BANK" --responses "$TC/bank-runs/parent-q4km.responses.jsonl" \
    --label "parent Qwen3.8-27B-U Q4_K_M (sampler proxy, llama-server greedy)" --out "$TC/bank-runs/parent-q4km.grade.json" || die 6 "bank-grade failed"
fi

if [ ! -s "$SCHEMAS/manifest.json" ]; then
  "$PY" "$T/toolcall_selfgen.py" schemas --server $S1 --server $S2 --out-dir "$SCHEMAS" --per-domain 3 --workers-per-server 4 || die 7 "schemas failed"
fi
if [ ! -s "$PLAN" ]; then
  "$PY" "$T/toolcall_selfgen.py" plan --schemas "$SCHEMAS/schemas.jsonl" --out "$PLAN" --items 2400 || die 7 "plan failed"
fi

if [ ! -s "$SMOKE/manifest.json" ]; then
  "$PY" "$T/toolcall_selfgen.py" run --plan "$PLAN" --server $S1 --server $S2 --out-dir "$SMOKE" --token-target 12000 \
    --gguf "$Q4" --gguf-sha256 "$Q4SHA" || die 8 "smoke run failed"
  "$PY" "$T/toolcall_selfgen.py" finalize --out-dir "$SMOKE" --bank "$BANK" --accept-short || die 8 "smoke finalize failed"
  "$PY" "$T/toolcall_selfgen.py" export --out-dir "$SMOKE" --dest "$SMOKE/smoke.safetensors" || die 8 "smoke export failed"
fi
"$PY" - "$SMOKE/journal.jsonl" <<'EOF' || die 9 "smoke acceptance below the 20 % gate"
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
ok = sum(r["status"] == "accepted" for r in rows)
reasons = collections.Counter(t["reason"][:70] for r in rows for t in r.get("rejections", []))
print(f"smoke: {ok}/{len(rows)} accepted; by category", dict(collections.Counter((r["category"], r["status"]) for r in rows)))
print("smoke rejection reasons:", reasons.most_common(12))
sys.exit(0 if rows and ok / len(rows) >= 0.2 else 1)
EOF

stamp "production run -> $TARGET tokens"
"$PY" "$T/toolcall_selfgen.py" run --plan "$PLAN" --server $S1 --server $S2 --out-dir "$RUN" --token-target $TARGET \
  --gguf "$Q4" --gguf-sha256 "$Q4SHA"
rc=$?
stamp "run rc=$rc"
[ "$rc" -eq 0 ] || die 10 "production run rc=$rc (resumable: re-run this driver)"
"$PY" "$T/toolcall_selfgen.py" finalize --out-dir "$RUN" --bank "$BANK" || die 11 "finalize failed"
"$PY" "$T/toolcall_selfgen.py" export --out-dir "$RUN" --dest "$DEST" || die 12 "export failed"
sha256sum "$RUN/manifest.json" "$DEST"

outcome="toolcall self-gen complete: $RUN/manifest.json sha $(sha256sum "$RUN/manifest.json" | cut -c1-16); export $DEST"
for g in 1 2; do
  "$PY" "$T/selfgen_sampler.py" stop --launch-receipt "$TC/receipts/serve-toolcall-q38-q4km-gpu$g.json" --outcome "$outcome"
done
echo "$(date -u +%FT%TZ) RELEASE GPU1,GPU2 owner=claude lane=toolcall ($outcome)" >> "$OC/distill/queues/p40.claims"
case "$Q4" in
  "$SCRATCH"/*.gguf) rm -f -- "$Q4" && stamp "removed transient $Q4" ;;
  *) stamp "refusing to remove $Q4: not in the declared scratch" ;;
esac
df -BG / /mnt/lexar
stamp "done"
