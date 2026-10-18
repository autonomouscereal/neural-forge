#!/bin/bash
# queue_gpu6.sh -- Stage D wave 2, lane gpu6 (GPU6, Quadro RTX 8000 46 GB). One server-side queue, run detached:
#   setsid nohup bash $OC/distill/queues/gpu6.sh > $OC/distill/queues/logs/gpu6/queue.log 2>&1 < /dev/null &
#
# Jobs, in order (one status line per job state in $OC/distill/queues/gpu6.status):
#   b2            the deciding K3 parity run: full 93-layer review pass in the REFERENCE environment (GPU6, lab
#                 venv torch 2.11/cu128, kda torch, --row-chunk-kda 1, --row-chunk-mla 1, --token-chunk 1024,
#                 --no-trim = the fold's own T=512 review rows) + `parity` against the fold engine's review dump
#                 (PARITY_THRESHOLDS unchanged, the wave-1 gates) + a block-state compare against the fold's
#                 before_L61 teacher snapshot (blocks entering L00/12/24/36/48/60; diagnostic, never a gate)
#   k3-pass-0..3  ONLY IF b2 PASSES: the K3 2.0M-token cache, one pass per $OC/distill/corpus/k3/pass-k file
#                 (122 rows x 4096), into /mnt/lexar/omni-campaign/teacher_cache/k3/<corpus manifest sha16>/;
#                 checkpoints in /mnt/lexar/omni-campaign/scratch/k3cache-state/<pass_sha16>/ (declared scratch)
#   k3-probe      the K3 route bits-per-byte probe over the 4 sealed passes -- only with a tool that reads the
#                 distill corpus's boundary_map.safetensors (subcommand probe-distill); otherwise SKIP, recorded
#   h3-*          ONLY with a GPU6 H3 thermal ruling in $Q/gpu6_h3_thermal_ruling.txt (minimax-h3-video skill:
#                 GPU6 is denied to H3 after its native thermal abort until a separate thermal remediation
#                 decision exists): P4 bf16, P4 fp16, encode, text, then the guided-velocity cache (media-a 1.0.2)
#                 with 2,048 targets, or 1,024 when the chosen dtype's P4 is > 20 s/forward (ruling G5-4)
# If b2 FAILS the K3 cache does not start; the H3 jobs run (if ruled) and the lane bisects L00-60 by hand.
# Every GPU job first waits until step 1's S8 review (pid 917640) has exited AND GPU6 reads < 2000 MiB twice.
# START_AT=<job> skips the jobs before it (a relaunch after a fix); the frozen K3 tool is reused if present.
set -u
VERSION="queue_gpu6.sh 1.0.0"
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
Q=$OC/distill/queues
QL=$Q/logs/gpu6
STATUS=$Q/gpu6.status
ALLOCD=$OC/composite/receipts/allocator
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=6
S8_PID=917640
LEXAR=/mnt/lexar/omni-campaign
B2=$OC/distill/parity/k3cache-2026-10-13-b2-gpu6
REF=$OC/k3-run-oct09-claude/fold/review
SNAP=$LEXAR/k3-continuation-oct08-claude/fold/snapshots/before_L61.safetensors
KC=$OC/distill/corpus/k3
K3_STATE=$LEXAR/scratch/k3cache-state
CACHE_ROOT=$LEXAR/teacher_cache
TPS=$OC/distill/tps
START_AT=${START_AT:-b2}
mkdir -p "$QL" "$ALLOCD" "$B2" "$TPS" "$Q/frozen"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src

utc() { date -u +%FT%TZ; }
status() {  # $1 job $2 state $3 note
  echo "$(utc) $1 $2 $3" | tee -a "$STATUS"
}
alloc() {  # $1 job $2 started|ended $3 note [$4 exit]
  python3 - "$ALLOCD/gpu6-$1-claude.json" "$1" "$2" "$3" "${4:-}" "$GPU" <<'PY'
import json, os, subprocess, sys, time
path, job, what, note, rc, gpu = sys.argv[1:7]
rec = json.load(open(path)) if os.path.exists(path) else {
    "lane": "gpu6 (Stage D wave 2)", "job": job, "gpu": int(gpu), "by": "claude gpu6 lane",
    "queue": "/home/cereal/omni-campaign/distill/queues/gpu6.sh", "notes": []}
q = lambda f: subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits", "-i", gpu],
                             capture_output=True, text=True).stdout.strip()
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rec.setdefault("card", q("name")); rec.setdefault("uuid", q("uuid"))
rec[f"{what}_utc"] = stamp
rec[f"memory_used_mib_at_{what}"] = q("memory.used")
rec["state"] = "running" if what == "started" else "ended"
if rc != "":
    rec["exit"] = int(rc)
rec["notes"].append(f"{stamp} {what}: {note}")
json.dump(rec, open(path, "w"), indent=1)
PY
}
wait_card() {  # step 1's S8 review gone, then GPU6 < 2000 MiB on two reads 30 s apart
  local n=0 quiet=0 used
  while [ -d /proc/$S8_PID ] && grep -q run_review_finite_guard /proc/$S8_PID/cmdline 2>/dev/null; do
    [ $((n % 10)) -eq 0 ] && echo "$(utc) waiting for step 1 S8 pid $S8_PID"
    n=$((n + 1)); sleep 60
  done
  while [ $quiet -lt 2 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    if [ "$used" -lt 2000 ]; then quiet=$((quiet + 1)); else quiet=0; echo "$(utc) GPU$GPU holds $used MiB"; fi
    [ $quiet -ge 2 ] || sleep 30
  done
  echo "$(utc) GPU$GPU free ($used MiB), S8 pid $S8_PID gone"
}
wait_lexar() {  # $1 = GiB needed free on /mnt/lexar (K3 checkpoints ~72 GB per pass + the 60 GiB floor)
  local n=0 free
  while :; do
    free=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
    [ "$free" -ge "$1" ] && break
    [ $((n % 10)) -eq 0 ] && echo "$(utc) /mnt/lexar has ${free} GB free < $1; waiting (omnicoder purge log: $(tail -1 $OC/composite/receipts/disk/oct13/purge_omnicoder_weights.log 2>/dev/null))"
    n=$((n + 1)); sleep 120
  done
  echo "$(utc) /mnt/lexar ${free} GB free"
}
wait_ram() {  # $1 = GiB MemAvailable (K3 pass 97-130 GB beside at most one 56 GB Qwen streamer)
  local n=0 avail
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge "$1" ] && break
    [ $((n % 10)) -eq 0 ] && echo "$(utc) MemAvailable ${avail} GiB < $1; waiting"
    n=$((n + 1)); sleep 60
  done
  echo "$(utc) MemAvailable ${avail} GiB"
}
reached() {  # true once START_AT has been passed
  case " $ORDER " in *" $START_AT "*) ;; *) echo "unknown START_AT $START_AT"; exit 2;; esac
  [ "$PASSED_START" = 1 ] && return 0
  [ "$1" = "$START_AT" ] && { PASSED_START=1; return 0; }
  status "$1" SKIP "before START_AT=$START_AT"
  return 1
}
ORDER="b2 k3-pass-0 k3-pass-1 k3-pass-2 k3-pass-3 k3-probe h3-p4-bf16 h3-p4-fp16 h3-encode h3-text h3-velocity"
PASSED_START=0

echo "$VERSION start $(utc) START_AT=$START_AT host $(hostname)"
# the K3 tool B2 proves is the tool the passes run: frozen once, by sha
K3TOOL=$Q/frozen/k3_teacher_cache-gpu6.py
if [ ! -e "$K3TOOL" ]; then cp -p "$T/k3_teacher_cache.py" "$K3TOOL"; chmod a-w "$K3TOOL"; fi
K3SHA=$(sha256sum "$K3TOOL" | cut -c1-64)
K3VER=$("$PY" "$K3TOOL" --version 2>/dev/null)
echo "frozen K3 tool $K3TOOL $K3VER sha256 $K3SHA"
status queue START "$VERSION k3tool=$K3VER sha256=${K3SHA:0:16} start_at=$START_AT"

# ---------------------------------------------------------------- b2: the deciding parity run
B2_OK=0
if reached b2; then
  status b2 WAIT "S8 pid $S8_PID + GPU$GPU < 2000 MiB"
  wait_card
  alloc b2 started "K3 full-pass review parity, reference env (lab venv, kda torch, row chunks 1, token chunk 1024, no trim)"
  status b2 START "log $QL/b2_pass.log"
  RC=1
  for attempt in 1 2 3; do
    echo "== b2 attempt $attempt $(utc)" >> "$QL/b2_pass.log"
    CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$K3TOOL" run \
      --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
      --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
      --pass-dir "$B2/review-pass" --state-dir "$B2/state" --keep-state \
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
      --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --no-trim --group-tokens 32768 \
      --resume >> "$QL/b2_pass.log" 2>&1
    RC=$?
    echo "b2 attempt $attempt exit=$RC $(utc)"
    [ $RC -eq 0 ] && break
    tail -5 "$QL/b2_pass.log" | grep -q "OutOfMemoryError" || break
    sleep 60
  done
  if [ $RC -ne 0 ]; then
    alloc b2 ended "pass failed" $RC
    status b2 FAIL "pass exit=$RC (functional; see $QL/b2_pass.log)"
  else
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" parity --cache-dir "$B2/review-pass" \
      --reference "$REF/k3_review_input_ids_teacher_forced.safetensors" \
      --reference "$REF/k3_review_native_input_ids_teacher_forced.safetensors" \
      --out "$B2/parity_b2_receipt.json" > "$QL/b2_parity.log" 2>&1
    PRC=$?
    cat "$QL/b2_parity.log"
    # diagnostic: our block states vs the fold's teacher before_L61 (review rows at token 98,304, T = 512)
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" - "$B2/state" "$SNAP" "$B2/b2_block_compare.json" \
      "$OC/composite/corpus/k3_review_input_ids.safetensors" \
      "$OC/composite/corpus/k3_review_native_input_ids.safetensors" > "$QL/b2_block_compare.log" 2>&1 <<'PYC'
import json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import load_file
base, snap, out, *ids = sys.argv[1:]
dirs = [os.path.join(base, d) for d in sorted(os.listdir(base)) if os.path.exists(os.path.join(base, d, "state.json"))]
assert len(dirs) == 1, dirs
state = json.load(open(os.path.join(dirs[0], "state.json")))
files = [load_file(p) for p in ids]
T = int(files[0]["input_ids"].shape[1])
assert all(int(f["input_ids"].shape[1]) == T for f in files), "--no-trim layout expects one T"
lens = torch.cat([f["real_lengths"].long().flatten() for f in files])
N = int(lens.numel()) * T
mask = (torch.arange(T)[None, :] < lens[:, None]).flatten()
OFF = 24 * 4096  # the fold corpus: 24 x 4,096 evidence rows, then the review rows (layout k3_review_input_ids=98304:512)
res = {"what": "k3_teacher_cache B2 block states vs the fold engine's teacher before_L61 block_residual",
       "state_dir": dirs[0], "state_layer_done": state["layer_done"], "snapshot": snap, "offset": OFF, "T": T,
       "positions": N, "real_positions": int(mask.sum()), "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "blocks": []}
with safe_open(snap, framework="pt") as h:
    res["snapshot_metadata"] = {k: v for k, v in (h.metadata() or {}).items() if len(str(v)) < 200}
    bs = h.get_slice("block_residual")
    nb_ref = int(bs.get_shape()[1])
    for b, rec in enumerate(state["blocks"][:nb_ref]):
        mine = load_file(rec["path"])["block"]
        assert mine.shape[0] == N, (mine.shape, N)
        ref = bs[OFF:OFF + N, b:b + 1].squeeze(1)
        m, r = mine[mask].float(), ref[mask].float()
        d = m - r
        cos = torch.nn.functional.cosine_similarity(m, r, dim=1)
        row = {"block": b, "enters_layer": 12 * b, "relL2": float(d.norm() / r.norm().clamp_min(1e-30)),
               "cos_mean": float(cos.mean()), "cos_p01": float(torch.quantile(cos, 0.01)), "cos_min": float(cos.min()),
               "max_abs_diff": float(d.abs().max()), "bit_equal_positions": float((mine[mask] == ref[mask]).all(1).float().mean())}
        res["blocks"].append(row)
        print("BLOCK", json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
print("BLOCK_COMPARE ->", out)
PYC
    echo "block compare exit=$? (diagnostic)"; cat "$QL/b2_block_compare.log"
    alloc b2 ended "parity exit=$PRC" $PRC
    if [ $PRC -eq 0 ]; then
      B2_OK=1
      status b2 PASS "$(grep -o 'PARITY PASS.*' "$QL/b2_parity.log" | head -c 300)"
      # the kept checkpoints (root, ~6 GB) were only for a bisect: remove this run's own files, nothing else
      for d in "$B2"/state/*/; do
        [ -f "$d/state.json" ] || continue
        rm -f "$d"state.json "$d"block_*.safetensors "$d"prefix_L*.safetensors && rmdir "$d" 2>/dev/null
      done
      rmdir "$B2/state" 2>/dev/null
      echo "b2 kept state removed after PASS: $(ls -A "$B2/state" 2>/dev/null | wc -l) entries left"
    else
      status b2 FAIL "parity exit=$PRC $(grep -o 'PARITY FAIL.*' "$QL/b2_parity.log" | head -c 300); K3 cache NOT started; bisect L00-60"
    fi
  fi
fi
if [ "$START_AT" != b2 ]; then  # a relaunch past b2: its sealed parity receipt decides
  B2_OK=$(python3 -c "import json;print(1 if json.load(open('$B2/parity_b2_receipt.json')).get('verdict')=='pass' else 0)" 2>/dev/null || echo 0)
  echo "relaunch at $START_AT: B2 verdict from $B2/parity_b2_receipt.json -> B2_OK=$B2_OK"
fi

# ---------------------------------------------------------------- K3 2.0M cache, 4 passes
K3SHA16=$(python3 -c "import json;print(json.load(open('$KC/manifest.json'))['manifest_sha256'][:16])")
K3OUT=$CACHE_ROOT/k3/$K3SHA16
K3_ALL=1
for k in 0 1 2 3; do
  job=k3-pass-$k
  reached $job || continue
  if [ "$B2_OK" != 1 ]; then status $job SKIP "B2 did not pass"; K3_ALL=0; continue; fi
  if [ -f "$K3OUT/pass-$k-of-4/manifest.json" ] && grep -q '"verdict": "sealed"' "$K3OUT/pass-$k-of-4/manifest.json"; then
    status $job DONE "already sealed $K3OUT/pass-$k-of-4"; continue
  fi
  status $job WAIT "S8/GPU$GPU, /mnt/lexar >= 200 GB free, MemAvailable >= 135 GiB"
  wait_card; wait_lexar 200; wait_ram 135
  mkdir -p "$K3OUT" "$K3_STATE"
  DIGEST=""; [ $k -eq 0 ] && DIGEST="--inline-digest"
  alloc $job started "K3 teacher cache pass $k of 4 ($KC/pass-$k), group tokens 65536"
  status $job START "log $QL/$job.log out $K3OUT/pass-$k-of-4"
  RC=1; GT=65536
  for attempt in 1 2 3 4; do
    echo "== $job attempt $attempt group_tokens $GT $(utc)" >> "$QL/$job.log"
    CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$K3TOOL" run \
      --input-ids "$KC/pass-$k/k3_input_ids.safetensors" --pass-count 1 --pass-index 0 \
      --pass-dir "$K3OUT/pass-$k-of-4" --state-dir "$K3_STATE" \
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
      --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --group-tokens $GT \
      --checkpoint-every 2 --min-free-gib 60 $DIGEST --resume >> "$QL/$job.log" 2>&1
    RC=$?
    echo "$job attempt $attempt exit=$RC $(utc)"
    [ $RC -eq 0 ] && break
    tail -8 "$QL/$job.log" | grep -q "OutOfMemoryError" || break
    GT=32768   # same token chunks (T = 4096 rows, chunk 1024): resumable, numerics unchanged
    sleep 60
  done
  alloc $job ended "exit=$RC" $RC
  if [ $RC -eq 0 ]; then status $job PASS "sealed $K3OUT/pass-$k-of-4/manifest.json"
  else status $job FAIL "exit=$RC (see $QL/$job.log); later K3 passes not started"; K3_ALL=0; break; fi
done

# ---------------------------------------------------------------- K3 route probe
if reached k3-probe; then
  if [ "$K3_ALL" != 1 ] || [ "$B2_OK" != 1 ]; then status k3-probe SKIP "K3 cache not complete"
  elif ! "$PY" "$T/k3_teacher_cache.py" --help 2>/dev/null | grep -q probe-distill; then
    status k3-probe SKIP "needs k3_teacher_cache probe-distill (reads corpus/k3 boundary_map.safetensors); current tool: $("$PY" "$T/k3_teacher_cache.py" --version)"
  else
    status k3-probe START "log $QL/k3-probe.log"
    args=()
    for k in 0 1 2 3; do args+=(--pass "$K3OUT/pass-$k-of-4=$KC/pass-$k"); done
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$T/k3_teacher_cache.py" probe-distill "${args[@]}" \
      --out "$K3OUT/route_probe_k3.json" > "$QL/k3-probe.log" 2>&1
    RC=$?
    tail -30 "$QL/k3-probe.log"
    [ $RC -eq 0 ] && status k3-probe PASS "$K3OUT/route_probe_k3.json" || status k3-probe FAIL "exit=$RC"
  fi
fi

# ---------------------------------------------------------------- H3 (media-a 1.0.2, through run_media_a_wave2.sh)
RULING_FILE=$Q/gpu6_h3_thermal_ruling.txt
media_step() {  # $1 job $2 launcher step, rest = extra tool args; waits for the detached docker job to end
  local job=$1 step=$2; shift 2
  wait_card
  local out alloc_path rc
  out=$(bash "$T/run_media_a_wave2.sh" "$step" $GPU "$@" 2>&1); rc=$?
  echo "$out"
  if [ $rc -ne 0 ]; then status $job FAIL "launcher refused: $(echo "$out" | tail -1 | head -c 300)"; return 2; fi
  alloc_path=$(echo "$out" | grep -o 'allocator [^ ]*' | tail -1 | cut -d' ' -f2)
  alloc $job started "media-a $step via run_media_a_wave2.sh; its allocator $alloc_path"
  status $job START "$(echo "$out" | grep -o 'log [^ ]*' | tail -1)"
  while ! grep -q '"state":"ended"' "$alloc_path" 2>/dev/null; do sleep 60; done
  rc=$(python3 -c "import json;print(json.load(open('$alloc_path'))['exit'])")
  alloc $job ended "media-a $step exit=$rc" "$rc"
  return "$rc"
}
H3_STEPS="h3-p4-bf16 h3-p4-fp16 h3-encode h3-text h3-velocity"
if [ ! -s "$RULING_FILE" ]; then
  for j in $H3_STEPS; do
    reached $j && status $j BLOCKED "no GPU6 H3 thermal ruling ($RULING_FILE absent): minimax-h3-video skill + run_media_a_wave2.sh deny H3 on GPU6 after its native thermal abort until a thermal remediation decision exists"
  done
else
  export H3_GPU6_THERMAL_RULING="$(head -1 "$RULING_FILE")" ALLOW_STEP1_CARDS=1
  ITEMS=$OC/distill/corpus/media/h3/items.jsonl
  CARD=quadrortx8000
  H3OK=1
  if reached h3-p4-bf16; then media_step h3-p4-bf16 p4-h3-bf16; r=$?
    [ $r -eq 0 ] && status h3-p4-bf16 PASS "$TPS/p4_p4-h3-bf16-gpu6-$CARD.json" || status h3-p4-bf16 FAIL "exit=$r"; fi
  if reached h3-p4-fp16; then media_step h3-p4-fp16 p4-h3-fp16; r=$?
    [ $r -eq 0 ] && status h3-p4-fp16 PASS "$TPS/p4_p4-h3-fp16-gpu6-$CARD.json" || status h3-p4-fp16 FAIL "exit=$r"; fi
  # dtype: the guarded fp16 cast when its P4 is finite and faster than bf16, else bf16; targets per ruling G5-4
  SEL=$(python3 - "$TPS/p4_p4-h3-bf16-gpu6-$CARD.json" "$TPS/p4_p4-h3-fp16-gpu6-$CARD.json" "$Q/gpu6_h3_selection.json" <<'PY'
import json, os, sys, time
bf, fp, out = sys.argv[1:4]
def read(p):
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    ok = not r.get("non_finite") and all(f.get("finite") for f in r.get("forwards", [])) and r.get("seconds_per_forward_steady")
    return {"path": p, "s_per_forward": r.get("seconds_per_forward_steady"), "finite": bool(ok), "vram_peak": r.get("vram_peak")}
b, f = read(bf), read(fp)
if f and f["finite"] and (not b or not b["finite"] or f["s_per_forward"] < b["s_per_forward"]):
    dtype, chosen = "float16", f
elif b and b["finite"]:
    dtype, chosen = "default", b
else:
    dtype, chosen = None, None
per_item = (4 if chosen["s_per_forward"] <= 20.0 else 2) if chosen else 0
sel = {"bf16": b, "fp16": f, "weight_dtype": dtype, "sigmas_per_item": per_item, "targets": 512 * per_item,
       "rule": "fp16 cast if finite and faster, else bf16; 2,048 targets unless > 20 s/forward -> 1,024 (STAGE_D_RULINGS G5-4)",
       "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(sel, open(out, "w"), indent=1)
print(f"{dtype or 'none'} {per_item}")
PY
)
  DT=${SEL% *}; SPI=${SEL#* }
  echo "H3 selection: weight_dtype=$DT sigmas_per_item=$SPI ($Q/gpu6_h3_selection.json)"
  if [ "$DT" = none ]; then
    for j in h3-encode h3-text h3-velocity; do reached $j && status $j SKIP "no finite H3 P4 receipt"; done
  else
    export ITEMS_H3=$ITEMS
    if reached h3-encode; then media_step h3-encode c-h3-encode --cache-root "$CACHE_ROOT"; r=$?
      [ $r -eq 0 ] && status h3-encode PASS "$(ls -d $CACHE_ROOT/h3-vae/*/ 2>/dev/null | tail -1)" || { status h3-encode FAIL "exit=$r"; H3OK=0; }; fi
    if [ $H3OK = 1 ] && reached h3-text; then media_step h3-text c-h3-text --cache-root "$CACHE_ROOT"; r=$?
      [ $r -eq 0 ] && status h3-text PASS "$(ls -d $CACHE_ROOT/h3-te-*/*/ 2>/dev/null | tail -1)" || { status h3-text FAIL "exit=$r"; H3OK=0; }; fi
    if [ $H3OK = 1 ] && reached h3-velocity; then
      VAE=$(ls -d $CACHE_ROOT/h3-vae/*/ | tail -1); TE=$(ls -d $CACHE_ROOT/h3-te-*/*/ | tail -1)
      export H3_VAE_CACHE=${VAE%/} H3_TE_CACHE=${TE%/}
      media_step h3-velocity c-h3-velocity --cache-root "$CACHE_ROOT" --weight-dtype "$DT" --sigmas-per-item "$SPI"; r=$?
      [ $r -eq 0 ] && status h3-velocity PASS "$(ls -d $CACHE_ROOT/h3-velocity-*/*/ 2>/dev/null | tail -1) targets=$((512 * SPI)) dtype=$DT" \
        || status h3-velocity FAIL "exit=$r"
    fi
  fi
fi
status queue END "$VERSION"
echo "$VERSION end $(utc)"
