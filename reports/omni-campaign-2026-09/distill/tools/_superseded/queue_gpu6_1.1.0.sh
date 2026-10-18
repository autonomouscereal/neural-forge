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
#                 checkpoints in /mnt/lexar/omni-campaign/scratch/k3cache-state/tool-<frozen sha16>/<pass_sha16>/
#                 (declared scratch; keyed by the frozen tool so another tool build never resumes onto them)
#   k3-probe      the K3 route bits-per-byte probe over the 4 sealed passes, with the probe tool FROZEN at launch
#                 ($Q/frozen/k3_teacher_cache-gpu6-probe.py, sha re-checked before it runs) -- only if that tool
#                 has probe-distill (reads corpus/k3 boundary_map.safetensors); otherwise SKIP, recorded
#   h3-*          ONLY with a GPU6 H3 thermal ruling in $Q/gpu6_h3_thermal_ruling.txt (minimax-h3-video skill:
#                 GPU6 is denied to H3 after its native thermal abort until a separate thermal remediation
#                 decision exists): P4 bf16, P4 fp16, encode, text, then the guided-velocity cache (media-a 1.0.2)
#                 with 2,048 targets, or 1,024 when the chosen dtype's P4 is > 20 s/forward (ruling G5-4)
# If b2 FAILS the K3 cache does not start; the H3 jobs run (if ruled) and the lane bisects L00-60 by hand.
# Every GPU job first waits until step 1's S8 review (pid 917640) has exited AND GPU6 reads < 2000 MiB twice.
# START_AT=<job> skips the jobs before it (a relaunch after a fix); the frozen K3 tool is reused if present.
#
# 1.1.0 (review fixes, 2026-10-13):
#  * thermal watchdog on b2 and every K3 pass (GPU6 sits at its 82 C max operating temperature under B2 with the
#    driver's SW thermal slowdown active, so 82 C is the regulated operating point, not the stop line): SIGTERM at
#    the card's slowdown temperature (84 C) held for 3 polls 10 s apart, or at shutdown - 2 (85 C) on one poll;
#    cool to max operating - 10 (72 C; or <= 78 C after 30 min), then --resume from the last checkpoint. Up to
#    THERMAL_RETRIES (6) per job; the trace is $QL/<job>.thermal.log (K3_THERMAL_STOP_C / _HARD_C / _RESUME_C override)
#  * b2 re-runs: the B2 dir is keyed by the frozen tool (queue_tool.sha256 in it; another tool -> <B2>-tool<sha16>),
#    a tool that already FAILED B2 is not re-run, and REFREEZE=1 archives the frozen copy by sha and freezes the
#    live tool; the K3 passes start only if the B2 parity receipt passed AND the B2 manifest's tool_sha256 is the
#    frozen tool's; a partial K3 pass dir written by another tool is moved aside, a sealed one is refused
#  * a b2 run started by an earlier queue is ADOPTED (watched, its exit read from its manifest) instead of re-run
#  * the probe tool is frozen at launch (REFREEZE_PROBE=1 to refresh) and its sha re-checked before k3-probe
# GPU6_QUEUE_SELFTEST=plan prints the resolved plan and exits; =thermal exercises the watchdog on dummy processes.
set -u
VERSION="queue_gpu6.sh 1.1.0"
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
B2_BASE=$OC/distill/parity/k3cache-2026-10-13-b2-gpu6
REF=$OC/k3-run-oct09-claude/fold/review
SNAP=$LEXAR/k3-continuation-oct08-claude/fold/snapshots/before_L61.safetensors
KC=$OC/distill/corpus/k3
K3_STATE_BASE=$LEXAR/scratch/k3cache-state
K3_STALE=$LEXAR/scratch/k3cache-stale
CACHE_ROOT=$LEXAR/teacher_cache
TPS=$OC/distill/tps
START_AT=${START_AT:-b2}
THERMAL_RETRIES=${THERMAL_RETRIES:-6}
mkdir -p "$QL" "$ALLOCD" "$TPS" "$Q/frozen"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src

utc() { date -u +%FT%TZ; }
status() {  # $1 job $2 state $3 note
  echo "$(utc) $1 $2 $3" | tee -a "$STATUS"
}
alloc() {  # $1 job $2 started|adopted|ended $3 note [$4 exit]
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
rec["state"] = "running" if what in ("started", "adopted") else "ended"
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
# ---------------------------------------------------------------- thermal guard (GPU6 has no fan control here)
tq() { nvidia-smi -q -d TEMPERATURE -i "$GPU" | awk -F: -v k="$1" '$1 ~ k && !seen {v=$2; gsub(/[^0-9]/, "", v); print v; seen=1}'; }
T_MAXOP=$(tq "GPU Max Operating Temp"); T_SLOW=$(tq "GPU Slowdown Temp"); T_SHUT=$(tq "GPU Shutdown Temp")
if [ -z "$T_MAXOP" ] || [ -z "$T_SLOW" ] || [ -z "$T_SHUT" ]; then
  echo "refuse: GPU$GPU thresholds unreadable (max operating '$T_MAXOP' slowdown '$T_SLOW' shutdown '$T_SHUT')"; exit 2
fi
STOP_C=${K3_THERMAL_STOP_C:-$T_SLOW}              # held 3 polls: the driver's regulation at max operating failed
HARD_C=${K3_THERMAL_HARD_C:-$((T_SHUT - 2))}      # one poll
RESUME_C=${K3_THERMAL_RESUME_C:-$((T_MAXOP - 10))}
[ "$HARD_C" -lt "$T_SHUT" ] || HARD_C=$((T_SHUT - 1))
[ "$STOP_C" -le "$HARD_C" ] || STOP_C=$HARD_C
thermal_watch() {  # $1 pid $2 job $3 job log: poll every 10 s while pid lives; SIGTERM it at STOP_C x3 or HARD_C x1
  local pid=$1 job=$2 logf=$3 n=0 hot=0 peak=0 sw=0 hw=0 why="" line t r c u p k
  local trace=$QL/$job.thermal.log marker=$QL/$job.thermal_stop
  echo "$(utc) watch pid $pid: stop ${STOP_C} C x3 polls / hard ${HARD_C} C (card: max operating $T_MAXOP, slowdown $T_SLOW, shutdown $T_SHUT)" >> "$trace"
  while [ -d /proc/$pid ]; do
    line=$(nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active,clocks.sm,utilization.gpu,power.draw \
           --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
    IFS=, read -r t r c u p <<< "$line"
    case "$t" in ''|*[!0-9]*) sleep 10; continue;; esac
    case "$r" in 0x*) ;; *) r=0;; esac
    n=$((n + 1)); [ "$t" -gt "$peak" ] && peak=$t
    [ $((r & 0x20)) -ne 0 ] && sw=$((sw + 1)); [ $((r & 0x40)) -ne 0 ] && hw=$((hw + 1))
    [ $((n % 6)) -eq 1 ] && echo "$(utc) T=${t}C reasons=$r sm=${c}MHz util=${u}% power=${p}W" >> "$trace"
    if [ "$t" -ge "$HARD_C" ]; then hot=3; elif [ "$t" -ge "$STOP_C" ]; then hot=$((hot + 1)); else hot=0; fi
    if [ "$hot" -ge 3 ]; then
      why="GPU$GPU at ${t} C (stop ${STOP_C} C x3 / hard ${HARD_C} C)"
      echo "$(utc) $why" > "$marker"
      echo "[thermal watchdog] $(utc) $why: SIGTERM pid $pid" | tee -a "$trace" >> "$logf"
      kill -TERM "$pid" 2>/dev/null
      for k in $(seq 24); do [ -d /proc/$pid ] || break; sleep 5; done
      [ -d /proc/$pid ] && { echo "$(utc) pid $pid alive 120 s after SIGTERM: SIGKILL" >> "$trace"; kill -KILL "$pid" 2>/dev/null; }
      break
    fi
    sleep 10
  done
  echo "$(utc) watch end pid $pid: polls $n peak ${peak} C sw_thermal_slowdown_polls $sw hw_thermal_slowdown_polls $hw stopped=${why:-no}" >> "$trace"
}
run_guarded() {  # $1 job $2 log; rest = the command, run under thermal_watch; sets RC
  local job=$1 logf=$2 pid wpid; shift 2
  rm -f "$QL/$job.thermal_stop"
  "$@" >> "$logf" 2>&1 < /dev/null &
  pid=$!
  thermal_watch "$pid" "$job" "$logf" &
  wpid=$!
  wait "$pid"; RC=$?
  wait "$wpid" 2>/dev/null
}
adopt_guarded() {  # $1 job $2 pid started by an earlier queue (not our child) $3 manifest $4 job log: RC = sealed by K3SHA
  local job=$1 pid=$2 sealed=$3 logf=$4 wpid
  rm -f "$QL/$job.thermal_stop"
  thermal_watch "$pid" "$job" "$logf" &
  wpid=$!
  while [ -d /proc/$pid ]; do sleep 20; done
  wait "$wpid" 2>/dev/null
  if manifest_sealed_by "$sealed" "$K3SHA"; then RC=0; else RC=1; fi
}
manifest_sealed_by() {  # $1 manifest.json $2 tool sha256: sealed by exactly that tool
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
m = json.load(open(sys.argv[1]))
sys.exit(0 if m.get("verdict") == "sealed" and m.get("tool_sha256") == sys.argv[2] else 1)
PY
}
wait_cool() {  # GPU6 <= RESUME_C, or <= max operating - 4 once 30 min have passed
  local s t
  s=$(date +%s)
  while :; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i $GPU | tr -dc 0-9)
    if [ -n "$t" ]; then
      [ "$t" -le "$RESUME_C" ] && break
      [ $(($(date +%s) - s)) -ge 1800 ] && [ "$t" -le $((T_MAXOP - 4)) ] && break
    fi
    sleep 30
  done
  echo "$(utc) GPU$GPU cooled to ${t} C after $(($(date +%s) - s)) s"
}
thermal_note() { grep "watch end" "$QL/$1.thermal.log" 2>/dev/null | tail -1 | cut -d' ' -f2-; }
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
# the K3 tool B2 proves is the tool the passes run: frozen once, by sha; REFREEZE=1 archives it by sha first
freeze() {  # $1 frozen path $2 refreeze flag
  local dst=$1 old
  if [ -e "$dst" ] && [ "$2" = 1 ]; then
    old=$(sha256sum "$dst" | cut -c1-16)
    mv -f "$dst" "${dst%.py}.$old.py"; echo "refreeze: $dst archived as ${dst%.py}.$old.py"
  fi
  if [ ! -e "$dst" ]; then cp -p "$T/k3_teacher_cache.py" "$dst"; chmod a-w "$dst"; fi
}
K3TOOL=$Q/frozen/k3_teacher_cache-gpu6.py
PROBETOOL=$Q/frozen/k3_teacher_cache-gpu6-probe.py
freeze "$K3TOOL" "${REFREEZE:-0}"
freeze "$PROBETOOL" "${REFREEZE_PROBE:-0}"
K3SHA=$(sha256sum "$K3TOOL" | cut -c1-64)
K3VER=$("$PY" "$K3TOOL" --version 2>/dev/null)
PROBESHA=$(sha256sum "$PROBETOOL" | cut -c1-64)
PROBEVER=$("$PY" "$PROBETOOL" --version 2>/dev/null)
echo "frozen K3 tool $K3TOOL $K3VER sha256 $K3SHA"
echo "frozen probe tool $PROBETOOL $PROBEVER sha256 $PROBESHA"
# the B2 dir belongs to one frozen tool: a different (or unrecorded) tool gets its own dir, never another's state
B2=$B2_BASE
if [ -s "$B2_BASE/queue_tool.sha256" ]; then
  [ "$(cat "$B2_BASE/queue_tool.sha256")" = "$K3SHA" ] || B2=$B2_BASE-tool${K3SHA:0:16}
elif [ -n "$(ls -A "$B2_BASE" 2>/dev/null)" ]; then
  B2=$B2_BASE-tool${K3SHA:0:16}
fi
K3_STATE=$K3_STATE_BASE/tool-${K3SHA:0:16}
# a b2 run of THIS tool into THIS dir left running by an earlier queue: adopt it
ADOPT=""
[ "$START_AT" = b2 ] && ADOPT=$(pgrep -f -- "$K3TOOL run .*--pass-dir $B2/review-pass --state-dir" | head -1)
echo "B2 dir $B2 (tool ${K3SHA:0:16}); K3 state $K3_STATE; adopt b2 pid: ${ADOPT:-none}"
echo "thermal: stop ${STOP_C} C x3 polls, hard ${HARD_C} C, resume <= ${RESUME_C} C; card max operating $T_MAXOP slowdown $T_SLOW shutdown $T_SHUT; retries $THERMAL_RETRIES"
if [ "${GPU6_QUEUE_SELFTEST:-}" = plan ]; then echo "SELFTEST plan: exit before any job"; exit 0; fi
if [ "${GPU6_QUEUE_SELFTEST:-}" = thermal ]; then  # the watchdog on dummy processes; no status lines, no GPU work
  QL=$(mktemp -d); echo "SELFTEST thermal in $QL"
  STOP_C=1 HARD_C=200; run_guarded st-hot "$QL/st-hot.log" sleep 600
  echo "ST hot: rc=$RC marker=$([ -s "$QL/st-hot.thermal_stop" ] && echo yes || echo no)"; tail -1 "$QL/st-hot.thermal.log"
  STOP_C=150 HARD_C=1; run_guarded st-hard "$QL/st-hard.log" sleep 600
  echo "ST hard: rc=$RC marker=$([ -s "$QL/st-hard.thermal_stop" ] && echo yes || echo no)"
  STOP_C=150 HARD_C=200; run_guarded st-cool "$QL/st-cool.log" sleep 25
  echo "ST cool: rc=$RC marker=$([ -s "$QL/st-cool.thermal_stop" ] && echo yes || echo no)"; tail -1 "$QL/st-cool.thermal.log"
  run_guarded st-fail "$QL/st-fail.log" bash -c "sleep 3; exit 7"
  echo "ST fail: rc=$RC marker=$([ -s "$QL/st-fail.thermal_stop" ] && echo yes || echo no)"
  echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"$K3SHA\"}" > "$QL/m_ok.json"
  echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"other\"}" > "$QL/m_other.json"
  setsid sleep 600 < /dev/null > /dev/null 2>&1 & OP=$!; disown
  STOP_C=1 HARD_C=200; adopt_guarded st-adopt "$OP" "$QL/none.json" "$QL/st-adopt.log"
  echo "ST adopt-hot: rc=$RC marker=$([ -s "$QL/st-adopt.thermal_stop" ] && echo yes || echo no) alive=$([ -d /proc/$OP ] && echo yes || echo no)"
  setsid sleep 15 < /dev/null > /dev/null 2>&1 & OP=$!; disown
  STOP_C=150 HARD_C=200; adopt_guarded st-adopt2 "$OP" "$QL/m_ok.json" "$QL/st-adopt2.log"
  echo "ST adopt-sealed: rc=$RC"
  setsid sleep 15 < /dev/null > /dev/null 2>&1 & OP=$!; disown
  adopt_guarded st-adopt3 "$OP" "$QL/m_other.json" "$QL/st-adopt3.log"
  echo "ST adopt-other-tool: rc=$RC"
  rm -rf "$QL"; exit 0
fi
mkdir -p "$B2"; [ -s "$B2/queue_tool.sha256" ] || echo "$K3SHA" > "$B2/queue_tool.sha256"
status queue START "$VERSION k3tool=$K3VER sha256=${K3SHA:0:16} probetool=$PROBEVER sha256=${PROBESHA:0:16} b2dir=$B2 thermal stop=${STOP_C}x3/hard=${HARD_C}/resume=${RESUME_C} start_at=$START_AT"

# ---------------------------------------------------------------- b2: the deciding parity run
B2_OK=0
B2_MAN=$B2/review-pass/manifest.json
if reached b2; then
  RC=1
  if [ -s "$B2/parity_b2_receipt.json" ]; then
    # one frozen tool, one B2 verdict: a re-run of the same tool reproduces it (and --resume would reuse its layers)
    RC=-1
    if python3 -c "import json,sys;sys.exit(0 if json.load(open('$B2/parity_b2_receipt.json')).get('verdict')=='pass' else 1)"; then
      status b2 PASS "already passed with frozen tool ${K3SHA:0:16} ($B2/parity_b2_receipt.json)"
    else
      status b2 REFUSED "frozen tool ${K3SHA:0:16} already failed B2 ($B2/parity_b2_receipt.json); fix the tool, then REFREEZE=1 START_AT=b2 (fresh tool-keyed B2 dir)"
    fi
  elif manifest_sealed_by "$B2_MAN" "$K3SHA"; then
    RC=0; echo "b2 pass already sealed by tool ${K3SHA:0:16}: parity only"
  else
    if [ -n "$ADOPT" ]; then
      alloc b2 adopted "adopted by $VERSION: pid $ADOPT started by the earlier queue, now under the thermal watchdog"
      status b2 ADOPT "pid $ADOPT (started by the earlier queue) under the thermal watchdog; log $QL/b2_pass.log"
    else
      status b2 WAIT "S8 pid $S8_PID + GPU$GPU < 2000 MiB"
      wait_card
      alloc b2 started "K3 full-pass review parity, reference env (lab venv, kda torch, row chunks 1, token chunk 1024, no trim)"
      status b2 START "log $QL/b2_pass.log"
    fi
    OOM=0; TH=0; attempt=0
    while :; do
      attempt=$((attempt + 1))
      if [ -n "$ADOPT" ]; then
        echo "== b2 attempt $attempt: adopted pid $ADOPT by $VERSION $(utc)" >> "$QL/b2_pass.log"
        adopt_guarded b2 "$ADOPT" "$B2_MAN" "$QL/b2_pass.log"; ADOPT=""
      else
        echo "== b2 attempt $attempt $(utc)" >> "$QL/b2_pass.log"
        run_guarded b2 "$QL/b2_pass.log" env CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$K3TOOL" run \
          --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
          --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
          --pass-dir "$B2/review-pass" --state-dir "$B2/state" --keep-state \
          --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
          --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --no-trim --group-tokens 32768 \
          --resume
      fi
      echo "b2 attempt $attempt exit=$RC $(utc)"
      [ $RC -eq 0 ] && break
      if [ -s "$QL/b2.thermal_stop" ]; then
        TH=$((TH + 1))
        [ $TH -gt "$THERMAL_RETRIES" ] && { echo "b2: thermal stop $TH > $THERMAL_RETRIES retries"; break; }
        status b2 THERMAL "stop $TH of $THERMAL_RETRIES: $(cut -d' ' -f2- "$QL/b2.thermal_stop"); cooling, then --resume"
        wait_cool; wait_card; continue
      fi
      tail -5 "$QL/b2_pass.log" | grep -q "OutOfMemoryError" || break
      OOM=$((OOM + 1)); [ $OOM -ge 3 ] && break
      sleep 60
    done
  fi
  if [ $RC -eq -1 ]; then
    :
  elif [ $RC -ne 0 ]; then
    alloc b2 ended "pass failed; thermal: $(thermal_note b2)" $RC
    if [ -s "$QL/b2.thermal_stop" ]; then status b2 HALT "thermal: $THERMAL_RETRIES watchdog stops used; last $(cut -d' ' -f2- "$QL/b2.thermal_stop") (not a numerics verdict)"
    else status b2 FAIL "pass exit=$RC (functional; see $QL/b2_pass.log)"; fi
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
    alloc b2 ended "parity exit=$PRC; thermal: $(thermal_note b2)" $PRC
    if [ $PRC -eq 0 ]; then
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
# the K3 passes run only on THIS frozen tool's B2 PASS: the parity receipt passed AND the B2 manifest it read was
# sealed by this very tool (a relaunch past b2, a refreeze, or a re-run all land here)
B2_OK=0
if [ -s "$B2/parity_b2_receipt.json" ] && manifest_sealed_by "$B2_MAN" "$K3SHA" && \n   python3 -c "import json,sys;sys.exit(0 if json.load(open('$B2/parity_b2_receipt.json')).get('verdict')=='pass' else 1)" 2>/dev/null; then
  B2_OK=1
fi
echo "B2 verdict for frozen tool ${K3SHA:0:16} from $B2/parity_b2_receipt.json + $B2_MAN -> B2_OK=$B2_OK"

# ---------------------------------------------------------------- K3 2.0M cache, 4 passes
K3SHA16=$(python3 -c "import json;print(json.load(open('$KC/manifest.json'))['manifest_sha256'][:16])")
K3OUT=$CACHE_ROOT/k3/$K3SHA16
K3_ALL=1
for k in 0 1 2 3; do
  job=k3-pass-$k
  reached $job || continue
  if [ "$B2_OK" != 1 ]; then status $job SKIP "B2 did not pass for frozen tool ${K3SHA:0:16}"; K3_ALL=0; continue; fi
  PD=$K3OUT/pass-$k-of-4
  if [ -f "$PD/manifest.json" ] && grep -q '"verdict": "sealed"' "$PD/manifest.json"; then
    if manifest_sealed_by "$PD/manifest.json" "$K3SHA"; then status $job DONE "already sealed $PD by tool ${K3SHA:0:16}"; continue; fi
    status $job FAIL "$PD is sealed by another tool than the B2-proven ${K3SHA:0:16}: decide by hand (not moved); later K3 passes not started"
    K3_ALL=0; break
  fi
  # a partial pass dir resumes (head_progress.json) only under the tool that wrote it; another tool's goes aside
  if [ -n "$(ls -A "$PD" 2>/dev/null)" ] && [ "$(cat "$PD.queue_tool.sha256" 2>/dev/null)" != "$K3SHA" ]; then
    mkdir -p "$K3_STALE"
    STALE=$K3_STALE/pass-$k-of-4.$(date -u +%Y%m%dT%H%M%SZ).tool$(cut -c1-16 "$PD.queue_tool.sha256" 2>/dev/null || echo unknown)
    mv "$PD" "$STALE" && echo "$(utc) $PD was written by another tool: moved aside to $STALE"
  fi
  status $job WAIT "S8/GPU$GPU, /mnt/lexar >= 200 GB free, MemAvailable >= 135 GiB"
  wait_card; wait_lexar 200; wait_ram 135
  mkdir -p "$K3OUT" "$K3_STATE"
  echo "$K3SHA" > "$PD.queue_tool.sha256"
  DIGEST=""; [ $k -eq 0 ] && DIGEST="--inline-digest"
  alloc $job started "K3 teacher cache pass $k of 4 ($KC/pass-$k), group tokens 65536, frozen tool ${K3SHA:0:16}, thermal stop ${STOP_C}x3/hard ${HARD_C}"
  status $job START "log $QL/$job.log out $PD state $K3_STATE thermal $QL/$job.thermal.log"
  RC=1; GT=65536; OOM=0; TH=0; attempt=0
  while :; do
    attempt=$((attempt + 1))
    echo "== $job attempt $attempt group_tokens $GT $(utc)" >> "$QL/$job.log"
    run_guarded $job "$QL/$job.log" env CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$K3TOOL" run \
      --input-ids "$KC/pass-$k/k3_input_ids.safetensors" --pass-count 1 --pass-index 0 \
      --pass-dir "$PD" --state-dir "$K3_STATE" \
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
      --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --group-tokens $GT \
      --checkpoint-every 2 --min-free-gib 60 $DIGEST --resume
    echo "$job attempt $attempt exit=$RC $(utc)"
    [ $RC -eq 0 ] && break
    if [ -s "$QL/$job.thermal_stop" ]; then
      TH=$((TH + 1))
      [ $TH -gt "$THERMAL_RETRIES" ] && { echo "$job: thermal stop $TH > $THERMAL_RETRIES retries"; break; }
      status $job THERMAL "stop $TH of $THERMAL_RETRIES: $(cut -d' ' -f2- "$QL/$job.thermal_stop"); cooling, then --resume"
      wait_cool; wait_card; wait_ram 135; continue
    fi
    tail -8 "$QL/$job.log" | grep -q "OutOfMemoryError" || break
    OOM=$((OOM + 1)); [ $OOM -ge 4 ] && break
    GT=32768   # same token chunks (T = 4096 rows, chunk 1024): resumable, numerics unchanged
    sleep 60
  done
  if [ $RC -eq 0 ] && ! manifest_sealed_by "$PD/manifest.json" "$K3SHA"; then
    echo "$job: exit 0 but $PD/manifest.json is not sealed by ${K3SHA:0:16}"; RC=70
  fi
  alloc $job ended "exit=$RC; thermal: $(thermal_note $job)" $RC
  if [ $RC -eq 0 ]; then status $job PASS "sealed $PD/manifest.json"
  elif [ -s "$QL/$job.thermal_stop" ]; then status $job HALT "thermal: $THERMAL_RETRIES watchdog stops used (see $QL/$job.thermal.log); resumable with START_AT=$job; later K3 passes not started"; K3_ALL=0; break
  else status $job FAIL "exit=$RC (see $QL/$job.log); later K3 passes not started"; K3_ALL=0; break; fi
done

# ---------------------------------------------------------------- K3 route probe (the tool frozen at launch)
if reached k3-probe; then
  if [ "$K3_ALL" != 1 ] || [ "$B2_OK" != 1 ]; then status k3-probe SKIP "K3 cache not complete"
  elif [ "$(sha256sum "$PROBETOOL" | cut -c1-64)" != "$PROBESHA" ]; then
    status k3-probe FAIL "frozen probe tool $PROBETOOL changed since launch (want ${PROBESHA:0:16})"
  elif ! "$PY" "$PROBETOOL" --help 2>/dev/null | grep -q probe-distill; then
    status k3-probe SKIP "needs k3_teacher_cache probe-distill (reads corpus/k3 boundary_map.safetensors); frozen probe tool: $PROBEVER"
  else
    status k3-probe START "log $QL/k3-probe.log tool $PROBEVER sha256 ${PROBESHA:0:16}"
    args=()
    for k in 0 1 2 3; do args+=(--pass "$K3OUT/pass-$k-of-4=$KC/pass-$k"); done
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$PROBETOOL" probe-distill "${args[@]}" \
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
