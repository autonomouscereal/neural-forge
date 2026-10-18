#!/bin/bash
# k3_multicard.sh -- K3 teacher-cache passes on a card beside GPU6 (lane k3plan, 2026-10-15; K3_ACCELERATION_PLAN_2026-10-14
# section 6).  One runner per card, detached:
#   setsid nohup bash $Q/k3_multicard.sh <gpu> > $Q/logs/k3mc-gpu<gpu>/runner.log 2>&1 < /dev/null &
# Gates, in order.  Each one WAITS (never competes, never kills); one status line per gate in $Q/k3mc-gpu<gpu>.status:
#   card     <gpu> is 0 or 4 (RTX 3090, sm_86).  GPU5 never; GPU6 belongs to queue_gpu6.sh; the P40s (1-3) are refused:
#            an MLA layer at T = 4,096 does not fit 24 GB with the math SDPA sm_61 runs (lane k3accel, 2026-10-14).
#   d1       $Q/d1.sealed exists: D1 owns GPU0 + GPU4 until it is sealed.
#   go       $Q/k3_multicard.go: line 1 "k3-multicard-go" and a line "gpus=<comma list>" naming this gpu.  The 3090s are
#            D2's upside cards; the lead writes this file only after choosing K3-first over D2-first (plan section 6).
#            A line "eqcheck_only=1" stops the runner after the eqcheck (the class is checked and its rate measured
#            before the K3-first / D2-first choice is made).
#   b2       the B2 PASS of the frozen 1.2.4 ($Q/frozen/k3_teacher_cache-gpu6.py), read with queue_gpu6's own b2_ok.
#   ruling   $Q/k3_card_class_ruling.txt, line 1 k3-card-class-equivalence-2026-10-15 (STAGE_D_RULINGS).
#   lock     $Q/gpu<gpu>.lock through lanes.py hold-lock (an flock) while this runner or its K3 job lives.
#   eqcheck  the ruling's card-class equivalence check, once per (card class, tool sha), on this card: a 93-layer bf16
#            review pass over the B2 review ids at the B2 geometry, then the UNCHANGED parity gate against the B2 fp32
#            reference (functional) + the same parity read against GPU6's bf16 pass and the fold dump (diagnostic).
#   per pass /mnt/lexar >= 200 GB free; MemAvailable >= the pass's whole footprint + headroom (ballast held as on GPU6).
# Passes are claimed from the top (3, 2, 1, 0) through k3_lib's contract; every pass on this card class runs at K3MC_GT
# (32,768 = the eqcheck's own geometry) group tokens, fixed per pass.  Tool: the frozen 1.3.0 when it was reviewed and
# frozen by queue_gpu6 >= 1.4.0, else the frozen 1.2.4; the eqcheck is run by that very tool.
# K3MC_SELFTEST=plan prints the resolved plan and exits; =gates exercises the gate readers on temp files.
set -u
VERSION="k3_multicard.sh 1.0.0"
GPU=${1:-}
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
Q=$OC/distill/queues
QL=$Q/logs/k3mc-gpu$GPU
STATUS=$Q/k3mc-gpu$GPU.status
ALLOCD=$OC/composite/receipts/allocator
PY=/home/cereal/model-surgery-lab/venv/bin/python
LEXAR=/mnt/lexar/omni-campaign
B2_BASE=$LEXAR/distill/parity/k3cache-2026-10-14-b2-gpu6
REBASE_RULING=$Q/k3_b2_rebaseline_ruling.txt
REF=$OC/k3-run-oct09-claude/fold/review
KC=$OC/distill/corpus/k3
K3_STATE_BASE=$LEXAR/scratch/k3cache-state
CACHE_ROOT=$LEXAR/teacher_cache
K3TOOL=$Q/frozen/k3_teacher_cache-gpu6.py
V130TOOL=$Q/frozen/k3_teacher_cache-gpu6-v130.py
PROBETOOL=$Q/frozen/k3_teacher_cache-gpu6-probe.py
V130_PIN=${K3_V130_SHA:-434ced6d7e8e22e4179bb2332d4695803dd89a38407805126261500c112fa3c7}
V130_REVIEW=${V130_REVIEW:-$Q/k3_v130_review.txt}
D1_SEALED=${D1_SEALED:-$Q/d1.sealed}
GO_FILE=${GO_FILE:-$Q/k3_multicard.go}
RULING_FILE=${RULING_FILE:-$Q/k3_card_class_ruling.txt}
RULING_ID=k3-card-class-equivalence-2026-10-15
K3LIB=${K3LIB:-$T/k3_lib.sh}
K3LIB_WANT="k3_lib.sh 1.0.0"
LANES_PY=${LANES_PY:-$T/lanes.py}
LANES_FILE=${LANES_FILE:-$Q/LANES.json}
LANE=L-K3
K3MC_GT=${K3MC_GT:-32768}
K3_RAM_HEADROOM_GIB=${K3_RAM_HEADROOM_GIB:-6}
K3_OVERHEAD_GIB=${K3_OVERHEAD_GIB:-20}
K3_OOM_ADJ=${K3_OOM_ADJ:-500}
RAM_FLOOR_GIB=${RAM_FLOOR_GIB:-8}
EQ_RAM_GIB=${EQ_RAM_GIB:-40}
THERMAL_RETRIES=${THERMAL_RETRIES:-6}
POLL_S=${POLL_S:-300}
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src

utc() { date -u +%FT%TZ; }
lane_hb() {  # $1 status $2 detail: heartbeat lane L-K3 (source = this runner); never fatal, at most 20 s
  [ -n "${K3MC_SELFTEST:-}" ] && return 0
  [ -f "$LANES_PY" ] || return 0
  timeout 20 python3 "$LANES_PY" --file "$LANES_FILE" heartbeat --lane "$LANE" --agent claude --source "$VERSION gpu$GPU" \
    --status "$1" --detail "$2" > /dev/null 2>&1 || echo "$(utc) lane heartbeat $LANE failed (ignored)"
  return 0
}
status() {  # $1 job $2 state $3 note
  echo "$(utc) gpu$GPU $1 $2 ${3:-}" | tee -a "$STATUS"
  lane_hb running "k3mc gpu$GPU $1 $2 ${3:0:280}"
}
alloc() {  # $1 job $2 started|ended $3 note [$4 exit]
  python3 - "$ALLOCD/gpu$GPU-k3mc-$1-claude.json" "$1" "$2" "$3" "${4:-}" "$GPU" <<'PY'
import json, os, subprocess, sys, time
path, job, what, note, rc, gpu = sys.argv[1:7]
rec = json.load(open(path)) if os.path.exists(path) else {
    "lane": "k3plan multicard", "job": job, "gpu": int(gpu), "by": "claude k3_multicard.sh",
    "queue": "/home/cereal/omni-campaign/distill/queues/k3_multicard.sh", "notes": []}
q = lambda f: subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits", "-i", gpu],
                             capture_output=True, text=True).stdout.strip()
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rec.setdefault("card", q("name")); rec.setdefault("uuid", q("uuid"))
rec[f"{what}_utc"] = stamp
rec["state"] = "running" if what == "started" else "ended"
if rc != "":
    rec["exit"] = int(rc)
rec["notes"].append(f"{stamp} {what}: {note}")
json.dump(rec, open(path, "w"), indent=1)
PY
}
mc_wait_card() {  # this card < 2000 MiB on two reads 30 s apart (D1 / D2 / a peer gone)
  local quiet=0 used
  while [ $quiet -lt 2 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -dc 0-9)
    if [ "${used:-99999}" -lt 2000 ]; then quiet=$((quiet + 1)); else quiet=0; fi
    [ $quiet -ge 2 ] || sleep 30
  done
}
go_names_me() {  # $1 go file: line 1 k3-multicard-go and a gpus= line that lists $GPU
  [ -s "$1" ] && [ "$(head -1 "$1" | tr -d '\r')" = k3-multicard-go ] && \
    grep -m1 '^gpus=' "$1" | cut -d= -f2 | tr ',' '\n' | tr -d ' \r' | grep -qx "$GPU"
}
ruling_ok() { [ -s "$1" ] && [ "$(head -1 "$1" | tr -d '\r')" = "$RULING_ID" ]; }
wait_for() {  # $1 gate name $2 test command (eval'd) $3 why: poll every POLL_S, one status line, heartbeat every 6 polls
  local n=0
  until eval "$2"; do
    [ $n -eq 0 ] && status gate WAIT "$1: $3"
    [ $((n % 6)) -eq 5 ] && lane_hb running "k3mc gpu$GPU waiting: $1"
    n=$((n + 1)); sleep "$POLL_S"
  done
  [ $n -gt 0 ] && status gate OK "$1"
  return 0
}

case "$GPU" in
  0|4) ;;
  5) echo "refuse: GPU5 is never used"; exit 2 ;;
  6) echo "refuse: GPU6 runs its K3 passes under queue_gpu6.sh"; exit 2 ;;
  1|2|3) echo "refuse: GPU$GPU is a P40 (sm_61): an MLA layer at T = 4,096 does not fit 24 GB with the math SDPA it runs (lane k3accel 2026-10-14)"; exit 2 ;;
  *) echo "usage: k3_multicard.sh <gpu 0|4>"; exit 2 ;;
esac
[ -s "$K3LIB" ] || { echo "refuse: $K3LIB missing"; exit 2; }
. "$K3LIB"
[ "${K3LIB_VERSION:-}" = "$K3LIB_WANT" ] || { echo "refuse: $K3LIB is '${K3LIB_VERSION:-?}', this runner needs '$K3LIB_WANT'"; exit 2; }

if [ "${K3MC_SELFTEST:-}" = gates ]; then  # the gate readers on temp files; no status lines, no GPU work
  D=$(mktemp -d)
  printf 'k3-multicard-go\ngpus=0,4\nby=selftest\n' > "$D/go"; echo "ST go listed: $(go_names_me "$D/go" && echo 1 || echo 0)"
  printf 'k3-multicard-go\ngpus=4\n' > "$D/go2";            echo "ST go not-listed: $(go_names_me "$D/go2" && echo 1 || echo 0)"
  printf 'k3-multicard-no\ngpus=0,4\n' > "$D/go3";          echo "ST go bad-line1: $(go_names_me "$D/go3" && echo 1 || echo 0)"
  printf 'k3-multicard-go\ngpus=10\n' > "$D/go4";           echo "ST go substring: $(go_names_me "$D/go4" && echo 1 || echo 0)"
  echo "ST go absent: $(go_names_me "$D/none" && echo 1 || echo 0)"
  echo "$RULING_ID" > "$D/r";                                echo "ST ruling ok: $(ruling_ok "$D/r" && echo 1 || echo 0)"
  echo "k3-card-class-equivalence-draft" > "$D/r2";          echo "ST ruling other: $(ruling_ok "$D/r2" && echo 1 || echo 0)"
  rm -rf "$D"; exit 0
fi
if [ -n "${K3MC_SELFTEST:-}" ] && [ "$K3MC_SELFTEST" != plan ]; then echo "refuse: unknown K3MC_SELFTEST=$K3MC_SELFTEST"; exit 2; fi

k3lib_thermal_init
CARD_CLASS=$(k3_card_class "$GPU")
[ -n "$CARD_CLASS" ] || { echo "refuse: GPU$GPU card class unreadable"; exit 2; }
K3SHA=$(sha256sum "$K3TOOL" 2>/dev/null | cut -c1-64)
[ -n "$K3SHA" ] || { echo "refuse: $K3TOOL missing (queue_gpu6.sh freezes it)"; exit 2; }
K3VER=$("$PY" "$K3TOOL" --version 2>/dev/null)
B2=$B2_BASE
if [ -s "$B2_BASE/queue_tool.sha256" ]; then [ "$(cat "$B2_BASE/queue_tool.sha256")" = "$K3SHA" ] || B2=$B2_BASE-tool${K3SHA:0:16}
elif [ -n "$(ls -A "$B2_BASE" 2>/dev/null)" ]; then B2=$B2_BASE-tool${K3SHA:0:16}; fi
REBASE=0; [ -s "$REBASE_RULING" ] && REBASE=1
# the tool this card's passes (and its eqcheck) run: the reviewed + frozen 1.3.0, else the B2-proven 1.2.4
PASSKIND=v124; PASSTOOL=$K3TOOL; PASSSHA=$K3SHA; PASSVER=$K3VER
if k3_v130_review_ok "$V130_REVIEW" "$V130_PIN" && [ "$(sha256sum "$V130TOOL" 2>/dev/null | cut -c1-64)" = "$V130_PIN" ]; then
  PASSKIND=v130; PASSTOOL=$V130TOOL; PASSSHA=$V130_PIN; PASSVER=$("$PY" "$V130TOOL" --version 2>/dev/null)
fi
EQ=$LEXAR/distill/parity/k3-cardclass-$CARD_CLASS-tool${PASSSHA:0:16}
EQV=$Q/k3_cardclass/$CARD_CLASS/verdict.json
K3SHA16=$(python3 -c "import json;print(json.load(open('$KC/manifest.json'))['manifest_sha256'][:16])")
K3OUT=$CACHE_ROOT/k3/$K3SHA16
K3_STATE=$K3_STATE_BASE/tool-${PASSSHA:0:16}-gpu$GPU   # per card: the lock helper matches this card's jobs by it
GT=$K3MC_GT; MOEB=1; K3_WAIT_CARD=mc_wait_card
echo "$VERSION gpu $GPU ($CARD_CLASS) $(utc): tool $PASSKIND $PASSVER ${PASSSHA:0:16}, group tokens $GT, B2 dir $B2 (rebase $REBASE)"
echo "gates: d1 $D1_SEALED | go $GO_FILE | b2 b2_ok | ruling $RULING_FILE ($RULING_ID) | lock $Q/gpu$GPU.lock | eqcheck $EQV in $EQ"
echo "thermal: stop ${STOP_C} C x3, hard ${HARD_C} C, resume <= ${RESUME_C} C (card max operating $T_MAXOP, slowdown $T_SLOW, shutdown $T_SHUT)"
echo "K3 out $K3OUT; state $K3_STATE; lib $K3LIB_VERSION $(sha256sum "$K3LIB" | cut -c1-16)"
[ "${K3MC_SELFTEST:-}" = plan ] && { echo "SELFTEST plan: exit before any gate"; exit 0; }
mkdir -p "$QL" "$ALLOCD"

status runner START "$VERSION ($CARD_CLASS) tool $PASSKIND ${PASSSHA:0:16} group tokens $GT"
wait_for d1 "[ -e '$D1_SEALED' ]" "D1 owns GPU$GPU until $D1_SEALED exists"
wait_for go "go_names_me '$GO_FILE'" "no K3-first decision for GPU$GPU in $GO_FILE (the 3090s are D2's cards)"
wait_for b2 "b2_ok '$B2' '$K3SHA' '$REBASE'" "no B2 PASS for the frozen 1.2.4 ${K3SHA:0:16} in $B2"
wait_for ruling "ruling_ok '$RULING_FILE'" "no card-class ruling $RULING_ID in $RULING_FILE"
# the card lock: an flock held by a lanes.py helper while this runner or a job of THIS card lives (its K3 passes use the
# per-card state dir, its eqcheck $EQ/state-gpu<gpu>): a job that outlives this runner keeps the card locked
take_card_lock() {  # RC 0 once the helper holds $Q/gpu$GPU.lock (it waits up to 600 s per try, never competes)
  local ready=$QL/lock.ready i
  rm -f "$ready"
  python3 "$LANES_PY" hold-lock --lock "$Q/gpu$GPU.lock" --while-pid $$ --cmd-match k3_multicard.sh \
    --also-match="--state-dir $K3_STATE " --also-match="--state-dir $EQ/state-gpu$GPU " \
    --wait 600 --poll 10 --ready-file "$ready" >> "$QL/lock.log" 2>&1 < /dev/null &
  LOCK_HELPER=$!
  for i in $(seq 1 630); do
    [ -s "$ready" ] && { rm -f "$ready"; return 0; }
    [ -d /proc/$LOCK_HELPER ] || return 1
    sleep 1
  done
  kill "$LOCK_HELPER" 2>/dev/null; return 1
}
until take_card_lock; do
  status gate WAIT "lock: $Q/gpu$GPU.lock held by another lane ($(tail -1 "$QL/lock.log" | head -c 200)); retry in ${POLL_S}s"
  sleep "$POLL_S"
done
status gate OK "lock $Q/gpu$GPU.lock (helper pid $LOCK_HELPER)"
mc_wait_card

# ---------------------------------------------------------------- eqcheck: the card class against the B2 bar
eqcheck() {  # sets RC: 0 pass, 3 fail (functional), other = did not complete
  local th=0 accel=() prc
  [ "$PASSKIND" = v130 ] && accel=(--no-fadvise-drop --kda-graph-keep kda-only --moe-group-batch 1)
  mkdir -p "$EQ"
  while :; do
    echo "== eqcheck $(utc) $PASSKIND ${PASSSHA:0:16} on $CARD_CLASS" >> "$QL/eqcheck.log"
    run_guarded eqcheck "$QL/eqcheck.log" env CUDA_VISIBLE_DEVICES="$GPU" nice -n 15 "$PY" "$PASSTOOL" run \
      --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
      --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
      --pass-dir "$EQ/review-pass" --state-dir "$EQ/state-gpu$GPU" \
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
      --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --no-trim --group-tokens 32768 "${accel[@]}" --resume
    [ "$RC" -eq 0 ] && break
    if [ -s "$QL/eqcheck.thermal_stop" ] && [ $th -lt "$THERMAL_RETRIES" ]; then th=$((th + 1)); wait_cool; mc_wait_card; continue; fi
    return
  done
  manifest_sealed_by "$EQ/review-pass/manifest.json" "$PASSSHA" || { RC=70; return; }
  # the gate: the B2 parity code (the frozen 1.2.4's, PARITY_THRESHOLDS unchanged) against the B2 fp32 reference
  CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" parity --cache-dir "$EQ/review-pass" \
    --reference "$B2/ref-fp32/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$B2/ref-fp32/k3_review_native_input_ids_teacher_forced.safetensors" \
    --out "$EQ/parity_vs_b2_fp32.json" > "$QL/eqcheck_parity.log" 2>&1
  prc=$?
  # diagnostics (never gates, owner policy 2026-10-06): against GPU6's own bf16 review pass and the fold dump
  mkdir -p "$EQ/ref-gpu6-bf16"
  for f in k3_review_input_ids k3_review_native_input_ids; do
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" export-reference --cache-dir "$B2/review-pass" \
      --positions-from "$REF/${f}_teacher_forced.safetensors" --out "$EQ/ref-gpu6-bf16/${f}_teacher_forced.safetensors" \
      >> "$QL/eqcheck_diag.log" 2>&1
  done
  CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" parity --cache-dir "$EQ/review-pass" \
    --reference "$EQ/ref-gpu6-bf16/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$EQ/ref-gpu6-bf16/k3_review_native_input_ids_teacher_forced.safetensors" \
    --out "$EQ/parity_vs_gpu6_bf16.json" >> "$QL/eqcheck_diag.log" 2>&1
  CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" parity --cache-dir "$EQ/review-pass" \
    --reference "$REF/k3_review_input_ids_teacher_forced.safetensors" \
    --reference "$REF/k3_review_native_input_ids_teacher_forced.safetensors" \
    --out "$EQ/parity_vs_fold.json" >> "$QL/eqcheck_diag.log" 2>&1
  mkdir -p "$(dirname "$EQV")"
  python3 - "$EQV" "$prc" "$PASSSHA" "$PASSVER" "$CARD_CLASS" "$GPU" "$EQ" "$RULING_ID" "$VERSION" "$K3SHA" <<'PY'
import hashlib, json, os, sys, time
out, prc, sha, ver, cls, gpu, eq, ruling, by, ref_tool = sys.argv[1:11]
def rd(name):
    p = os.path.join(eq, name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as handle:
        body = handle.read()
    try:
        v = json.loads(body).get("verdict")
    except ValueError:
        v = None
    return {"path": p, "sha256": hashlib.sha256(body).hexdigest(), "verdict": v}
rec = {"schema": "omni-k3-cardclass-v1", "ruling": ruling, "card_class": cls, "gpu": int(gpu), "tool_sha256": sha,
       "tool_version": ver, "parity_code_tool_sha256": ref_tool,
       "verdict": "pass" if prc == "0" else "fail",
       "gate": rd("parity_vs_b2_fp32.json"),
       "diagnostics": {"vs_gpu6_bf16": rd("parity_vs_gpu6_bf16.json"), "vs_fold": rd("parity_vs_fold.json")},
       "geometry": "B2: k3_review + k3_review_native, --no-trim, group tokens 32768, kda torch, row chunks 1, token chunk 1024",
       "by": by, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tmp = out + ".tmp"
with open(tmp, "w") as handle:
    json.dump(rec, handle, indent=1)
os.replace(tmp, out)
print(f"EQCHECK {rec['verdict']} {cls} tool {sha[:16]} -> {out}")
PY
  RC=$([ "$prc" -eq 0 ] && echo 0 || echo 3)
}
# one card of a class runs the eqcheck; a second runner of that class waits on this flock, then reads the verdict
mkdir -p "$(dirname "$EQV")"
exec 7> "$(dirname "$EQV").lock"
flock 7
if k3_gate_ok "$EQV" "$PASSSHA" "$CARD_CLASS"; then
  status eqcheck PASS "already passed ($EQV)"
else
  [ -s "$EQV" ] && python3 -c "import json,sys;v=json.load(open(sys.argv[1]));sys.exit(0 if v.get('verdict')=='fail' and v.get('tool_sha256')==sys.argv[2] else 1)" "$EQV" "$PASSSHA" && {
    status eqcheck REFUSED "$CARD_CLASS already FAILED the equivalence check with ${PASSSHA:0:16} ($EQV): no K3 pass on this class"; exit 3; }
  wait_ram "$EQ_RAM_GIB"
  status eqcheck START "93-layer bf16 review pass at the B2 geometry on $CARD_CLASS, then the B2 parity gate vs the fp32 reference; log $QL/eqcheck.log"
  alloc eqcheck started "card-class equivalence check ($RULING_ID) on $CARD_CLASS"
  eqcheck
  alloc eqcheck ended "eqcheck exit $RC" "$RC"
  case $RC in
    0) status eqcheck PASS "$(grep -o 'PARITY PASS.*' "$QL/eqcheck_parity.log" | head -c 300) ($EQV)" ;;
    3) status eqcheck FAIL "functional: $CARD_CLASS does not meet the B2 bar ($(grep -o 'PARITY FAIL.*' "$QL/eqcheck_parity.log" | head -c 300)); no K3 pass on this class"; exit 3 ;;
    *) status eqcheck HALT "did not complete (exit $RC, $QL/eqcheck.log)"; exit 1 ;;
  esac
fi
flock -u 7; exec 7>&-
if grep -qx 'eqcheck_only=1' "$GO_FILE" 2>/dev/null; then  # the lead's measuring run: class checked, rate in the log
  status runner END "eqcheck_only=1 in $GO_FILE: $CARD_CLASS checked, its 93-layer rate is in $QL/eqcheck.log; no pass taken"
  exit 0
fi

# ---------------------------------------------------------------- the passes, from the top
mkdir -p "$K3OUT" "$K3_STATE"
while :; do
  PICK=$(k3_pick_pass "$K3OUT" "3 2 1 0" "$CARD_CLASS" "$PASSSHA" "$GT" "$K3SHA,$V130_PIN" "k3_multicard.sh")
  case "$PICK" in "PICK "*) ;; *) status runner END "no pass left for $CARD_CLASS [${PICK#NONE }]"; break ;; esac
  k=${PICK#PICK }; job=k3-pass-$k; PD=$K3OUT/pass-$k-of-4
  if ! k3_size_pass "$k"; then status $job FAIL "cannot size $KC/pass-$k"; k3_release_claim "$PD"; break; fi
  status $job WAIT "/mnt/lexar >= 200 GB free, MemAvailable >= $K3_GATE GiB (footprint $K3_RES + $K3_RAM_HEADROOM_GIB)"
  mc_wait_card; wait_lexar 200; wait_ram "$K3_GATE"
  DIGEST=""; [ "$k" -eq 0 ] && DIGEST="--inline-digest"
  alloc $job started "K3 pass $k of 4 on $CARD_CLASS, $PASSKIND ${PASSSHA:0:16}, group tokens $GT, reserve $K3_RES GiB"
  status $job START "$PASSKIND ${PASSSHA:0:16} on $CARD_CLASS, group tokens $GT; log $QL/$job.log out $PD"
  k3_run_pass "$k" "$PD"
  alloc $job ended "exit=$RC $K3_END; thermal: $(thermal_note $job)" "$RC"
  k3_release_claim "$PD"
  if [ "$K3_END" != PASS ]; then status $job "$K3_END" "$K3_NOTE"; break; fi
  k3_record_pass "$k" "$PD" "$PASSSHA" "$PASSVER" "$CARD_CLASS" "$EQV"
  status $job PASS "$K3_NOTE ($CARD_CLASS, recorded in $Q/k3_pass_tools.json)"
  if k3_all_sealed "$K3OUT" "$K3SHA,$V130_PIN"; then
    if "$PY" "$PROBETOOL" --help 2>/dev/null | grep -q probe-distill; then
      k3_probe "$PROBETOOL" "$K3OUT" "$KC" "$QL/k3-probe.log"
      [ "$RC" -eq 0 ] && status k3-probe PASS "$K3OUT/route_probe_k3.json" || status k3-probe FAIL "exit=$RC ($QL/k3-probe.log)"
    else
      status k3-probe SKIP "frozen probe tool $PROBETOOL has no probe-distill"
    fi
  fi
done
status runner EXIT "$(utc)"
