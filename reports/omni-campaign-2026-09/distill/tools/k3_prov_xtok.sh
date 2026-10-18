#!/bin/bash
# k3_prov_xtok.sh -- xtok targets for the PROVISIONAL bf16 K3 passes (ruling k3-b2-bf16-numerics-provisional-2026-10-17),
# one pass at a time as each seals, so the full-parameter KD run can take K3 rows while later passes still run.
#   setsid nohup bash k3_prov_xtok.sh > $Q/logs/k3_prov_xtok.log 2>&1 < /dev/null &
# Per pass k (0..3): wait for teacher_cache/k3/<sha16>-provisional-bf16/pass-k-of-4/manifest.json verdict "sealed" by a
# pinned K3 tool -> build xtok with the FROZEN 1.2.4 exactly as d2_plan 1.4.1 k3-seal does (same argv) -> verify the
# targets against the pass's Q38 cache with streamed_kd_trainer's own XtokTargets.verify_against -> write
# xtok/pass-k/READY.json (the data config's sealed_marker).  Nothing outside the -provisional-bf16 tree is written.
set -u
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
KC=$OC/distill/corpus/k3
K3TOOL=$Q/frozen/k3_teacher_cache-gpu6.py
OUT=${K3_PROV_OUT:-/mnt/lexar/omni-campaign/teacher_cache/k3/eb35851b2496a628-provisional-bf16}
POLL_S=${POLL_S:-300}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
utc() { date -u +%FT%TZ; }
sealed() {  # $1 pass dir: manifest verdict sealed
  python3 -c "import json,sys; m=json.load(open(sys.argv[1]+'/manifest.json')); sys.exit(0 if m.get('verdict')=='sealed' else 1)" "$1" 2> /dev/null
}
echo "$(utc) k3_prov_xtok start: out $OUT, frozen xtok tool $(sha256sum $K3TOOL | cut -c1-16)"
while :; do
  left=0
  for k in 0 1 2 3; do
    PD=$OUT/pass-$k-of-4
    XD=$OUT/xtok/pass-$k
    [ -s "$XD/READY.json" ] && continue
    left=$((left + 1))
    sealed "$PD" || continue
    SRC=$KC/pass-$k
    mkdir -p "$OUT/xtok"
    if [ ! -s "$XD/xtok_receipt.json" ]; then
      echo "$(utc) pass $k sealed: building xtok"
      nice -n 15 $PY $K3TOOL xtok --cache-dir "$PD" --boundary-map "$SRC/boundary_map.json" \
        --k3-ids "$SRC/k3_input_ids.safetensors" --qwen-ids "$SRC/qwen_input_ids.safetensors" --out "$XD" \
        > "$OUT/xtok/xtok-pass-$k.log" 2>&1 || { echo "$(utc) pass $k xtok FAILED rc=$? (see $OUT/xtok/xtok-pass-$k.log)"; continue; }
    fi
    nice -n 15 $PY - "$PD" "$XD" "$SRC" "$k" <<'PYEOF' || { echo "$(utc) pass $k verify FAILED"; continue; }
import hashlib, json, sys, time
from pathlib import Path
import streamed_kd_trainer as T
pd, xd, src, k = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
man_sha = sha(pd / "manifest.json")
receipt = json.loads((xd / "xtok_receipt.json").read_text())
named = [c.get("sha256") for c in receipt.get("cache") or []]
if named != [man_sha]:
    raise SystemExit(f"xtok receipt names K3 manifest(s) {named}, not pass {k}'s {man_sha[:16]}")
qsha = sha(src / "qwen_input_ids.safetensors")
cache_dir = Path("/mnt/lexar/omni-campaign/teacher_cache/qtext/q38-27bu") / qsha
cache = T.QTextCache(cache_dir)
xtok = T.XtokTargets(xd / "xtok_target.safetensors", "qwen_input_ids")
proof = xtok.verify_against(cache)
ready = {"schema": "omni-k3-provisional-xtok-ready-v1", "pass": k, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "ruling": "k3-b2-bf16-numerics-provisional-2026-10-17", "k3_manifest_sha256": man_sha,
         "k3_tool_sha256": json.loads((pd / "manifest.json").read_text()).get("tool_sha256"),
         "xtok": str(xd / "xtok_target.safetensors"), "xtok_sha256": xtok.sha256, "pairs": xtok.pairs,
         "xtok_receipt_sha256": sha(xd / "xtok_receipt.json"), "q38_cache": str(cache_dir),
         "q38_cache_manifest_sha256": cache.manifest_sha256, "verify": proof.get("method")}
tmp = xd / "READY.json.tmp"
tmp.write_text(json.dumps(ready, indent=1))
tmp.replace(xd / "READY.json")
print(f"pass {k} READY: {xtok.pairs} pairs, xtok {xtok.sha256[:16]}, q38 cache {cache.manifest_sha256[:16]}")
PYEOF
    echo "$(utc) pass $k READY ($XD/READY.json)"
  done
  [ "$left" = 0 ] && { echo "$(utc) every provisional pass has xtok + READY"; exit 0; }
  sleep "$POLL_S"
done
