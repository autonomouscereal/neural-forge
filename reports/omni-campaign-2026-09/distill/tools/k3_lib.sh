#!/bin/bash
# k3_lib.sh -- the K3 teacher-cache helpers shared by queue_gpu6.sh (>= 1.4.0, GPU6) and k3_multicard.sh (>= 1.0.0, the
# extra cards).  Sourced, never run.  Lane k3plan, 2026-10-15 (K3_ACCELERATION_PLAN_2026-10-14.md).
#
# Part 1 (verbatim from queue_gpu6.sh 1.3.2 lines 199-389; only the thermal thresholds moved into k3lib_thermal_init so
# a caller picks its card first): wait_lexar, wait_ram, k3_reserve, ram_ballast, tq, thermal_watch, run_guarded,
# adopt_guarded, manifest_sealed_by, wait_cool, thermal_note, receipt_passed, identity_ok, b2_ok.
# Part 2 (new): the K3 pass contract every runner obeys, so passes can run on several cards at once:
#   * a pass is CLAIMED under flock $Q/k3_passes.lock in <pass dir>.claim (runner, gpu, card class, pid); a claim is live
#     while its runner pid lives with its cmd match, or any process runs that pass (--pass-dir <pd> in its cmdline)
#   * a pass runs start to finish on ONE card class, ONE tool sha and ONE group-token count, recorded in <pass dir>.geometry
#     at its first start; a resume that differs is REFUSED (queue_gpu6.sh <= 1.3.2 resumed an OOM at 32,768 group tokens,
#     and the routed MoE's bf16 numbers depend on its GEMM row count: review 2026-10-15 item 1)
#   * an out-of-memory never changes the group tokens: 1.3.0 drops --moe-group-batch to 1 (each group keeps its own rows,
#     GEMMs and accumulation order: parity w4, bit-identical), then one plain retry, then HALT for a decision
#   * every sealed pass is recorded in $Q/k3_pass_tools.json (tool sha, card class, gate receipt) for d2_plan
# The caller defines: utc, status <job> <state> <note>, alloc <job> <what> <note> [rc], lane_hb <status> <detail>, and the
# globals GPU, QL, Q, OC, PY, KC, K3_STATE, K3_OVERHEAD_GIB, RAM_FLOOR_GIB, K3_OOM_ADJ, THERMAL_RETRIES.
K3LIB_VERSION="k3_lib.sh 1.0.0"

# ================================================================ part 1: moved verbatim from queue_gpu6.sh 1.3.2
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
wait_ram() {  # $1 = GiB MemAvailable (a K3 pass: k3_reserve = its whole footprint, ~85 GiB at 499,712 positions)
  local n=0 avail
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge "$1" ] && break
    [ $((n % 10)) -eq 0 ] && echo "$(utc) MemAvailable ${avail} GiB < $1; waiting"
    [ $((n % 30)) -eq 0 ] && lane_hb running "waiting for MemAvailable >= $1 GiB (now ${avail})"
    n=$((n + 1)); sleep 60
  done
  echo "$(utc) MemAvailable ${avail} GiB"
}
k3_reserve() {  # $1 a K3 pass input_ids file -> "<whole footprint GiB> <one block slot GiB>"
  python3 - "$1" "$K3_OVERHEAD_GIB" <<'PY'
import json, math, struct, sys
path, overhead = sys.argv[1], float(sys.argv[2])
with open(path, "rb") as handle:
    head = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
rows, T = head["input_ids"]["shape"][:2]
# k3_teacher_cache.state_bytes: positions x hidden 7,168 x bf16 x (8 AttnRes block slots [93 layers / block 12] + 2)
H, SLOTS, ELEM, G = 7168, 8, 2, 1 << 30
positions = int(rows) * int(T)
print(math.ceil(positions * H * ELEM * (SLOTS + 2) / G + overhead), math.ceil(positions * H * ELEM / G))
PY
}
ram_ballast() {  # $1 pid $2 reserve GiB $3 slack GiB $4 log: hold (reserve - RssAnon(pid) - slack) GiB committed while pid lives
  python3 - "$1" "$2" "$3" "$RAM_FLOOR_GIB" "$K3_OOM_ADJ" >> "$4" 2>&1 <<'PY'
import mmap, os, sys, time
pid, reserve, slack, floor, adj = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
G = 1 << 30
stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def avail():
    with open("/proc/meminfo") as handle:
        return next(int(l.split()[1]) * 1024 for l in handle if l.startswith("MemAvailable:"))
def anon(p):
    try:
        with open(f"/proc/{p}/status") as handle:
            v = {l.split(":")[0]: int(l.split()[1]) for l in handle if l.startswith(("RssAnon:", "RssShmem:"))}
        return (v.get("RssAnon", 0) + v.get("RssShmem", 0)) * 1024
    except OSError:
        return None
for path, val in (("/proc/self/oom_score_adj", 1000), (f"/proc/{pid}/oom_score_adj", adj)):
    try:
        with open(path, "w") as handle:
            handle.write(str(val))
    except OSError as exc:
        print(f"{stamp()} {path} <- {val} failed: {exc}", flush=True)
chunks = []
def grab():
    m = mmap.mmap(-1, G)
    view = memoryview(m)
    view[::4096] = b"\x01" * (G // 4096)        # one write per page: every page committed
    view.release()
    chunks.append(m)
print(f"{stamp()} ballast for pid {pid}: reserve {reserve:g} GiB, slack {slack:g} GiB, floor {floor:g} GiB, "
      f"oom_score_adj self 1000 / pid {adj}", flush=True)
last, logged = -1, 0.0
while True:
    rss = anon(pid)
    if rss is None:
        break
    want = max(0, int((reserve * G - rss - slack * G) // G))
    have = avail()
    if have < floor * G:                         # never the cause of pressure: yield what the floor needs
        want = min(want, max(0, len(chunks) - int(-(-(floor * G - have) // G))))
    while len(chunks) > want:
        chunks.pop().close()
    while len(chunks) < want and avail() > (floor + 2) * G:
        grab()
    if len(chunks) != last or time.time() - logged > 600:
        print(f"{stamp()} pid {pid} RssAnon {rss / G:.1f} GiB, ballast {len(chunks)} GiB (want {want}), "
              f"MemAvailable {avail() / G:.1f} GiB", flush=True)
        last, logged = len(chunks), time.time()
    time.sleep(5)
for m in chunks:
    m.close()
print(f"{stamp()} pid {pid} gone: ballast released", flush=True)
PY
}
# ---------------------------------------------------------------- thermal guard (GPU6 has no fan control here; any card)
tq() { nvidia-smi -q -d TEMPERATURE -i "$GPU" | awk -F: -v k="$1" '$1 ~ k && !seen {v=$2; gsub(/[^0-9]/, "", v); print v; seen=1}'; }
k3lib_thermal_init() {  # the card's thresholds ($GPU): 1.3.2 ran these lines at the top level for GPU6
  T_MAXOP=$(tq "GPU Max Operating Temp"); T_SLOW=$(tq "GPU Slowdown Temp"); T_SHUT=$(tq "GPU Shutdown Temp")
  if [ -z "$T_MAXOP" ] || [ -z "$T_SLOW" ] || [ -z "$T_SHUT" ]; then
    echo "refuse: GPU$GPU thresholds unreadable (max operating '$T_MAXOP' slowdown '$T_SLOW' shutdown '$T_SHUT')"; exit 2
  fi
  STOP_C=${K3_THERMAL_STOP_C:-$T_SLOW}              # held 3 polls: the driver's regulation at max operating failed
  HARD_C=${K3_THERMAL_HARD_C:-$((T_SHUT - 2))}      # one poll
  RESUME_C=${K3_THERMAL_RESUME_C:-$((T_MAXOP - 10))}
  [ "$HARD_C" -lt "$T_SHUT" ] || HARD_C=$((T_SHUT - 1))
  [ "$STOP_C" -le "$HARD_C" ] || STOP_C=$HARD_C
}
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
    if [ $((n % 180)) -eq 1 ]; then lane_hb running "$job pid $pid T=${t}C (watch poll $n)" >> "$trace" 2>&1 & fi
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
run_guarded() {  # $1 job $2 log; rest = the command, run under thermal_watch (+ $GUARD_SIDECAR <pid> if set); sets RC
  local job=$1 logf=$2 pid wpid spid=""; shift 2
  rm -f "$QL/$job.thermal_stop"
  "$@" >> "$logf" 2>&1 < /dev/null &
  pid=$!
  thermal_watch "$pid" "$job" "$logf" &
  wpid=$!
  if [ -n "${GUARD_SIDECAR:-}" ]; then "$GUARD_SIDECAR" "$pid" & spid=$!; fi
  wait "$pid"; RC=$?
  wait "$wpid" 2>/dev/null
  [ -n "$spid" ] && wait "$spid" 2>/dev/null
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
receipt_passed() {  # $1 parity receipt: verdict pass
  python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get('verdict')=='pass' else 1)" "$1" 2>/dev/null
}
identity_ok() {  # $1 b2_identity.json $2 frozen K3 tool sha: a PASS by that very tool over all of K3's 93 layers
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
r = json.load(open(sys.argv[1]))
v = r.get("verdict") or {}
sys.exit(0 if r.get("mode") == "identity" and v.get("pass") is True and
         (r.get("frozen_tool") or {}).get("sha256") == sys.argv[2] and
         v.get("layers_checked") == v.get("layers_expected") == 93 else 1)
PY
}
b2_ok() {  # $1 B2 dir $2 frozen tool sha $3 rebase (0/1): the K3 passes run only on this
  [ -s "$1/parity_b2_receipt.json" ] || return 1
  manifest_sealed_by "$1/review-pass/manifest.json" "$2" || return 1
  if [ "$3" = 1 ]; then
    manifest_sealed_by "$1/review-pass-fp32/manifest.json" "$2" || return 1
    identity_ok "$1/identity/b2_identity.json" "$2" || return 1   # the fold anchor: bf16-vs-fp32 alone is one tool
  fi
  receipt_passed "$1/parity_b2_receipt.json"
}
# ================================================================ part 2: the K3 pass contract (new in k3_lib 1.0.0)
K3_LEGACY_CLASS=${K3_LEGACY_CLASS:-quadro-rtx-8000-sm75}  # every pass started before 1.4.0 ran on GPU6 (queue_gpu6 <= 1.3.2)
k3_card_class() {  # $1 gpu index -> "<name>-sm<cc>", e.g. quadro-rtx-8000-sm75, nvidia-geforce-rtx-3090-sm86
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i "$1" 2>/dev/null | head -1 | \
    awk -F', *' '{n = tolower($1); gsub(/[^a-z0-9]+/, "-", n); c = $2; gsub(/[^0-9]/, "", c); if (n != "" && c != "") print n "-sm" c}'
}
k3_v130_review_ok() {  # $1 review file $2 pinned sha: line 1 is "k3-v130-review-pass" and one line is "sha256=<pin>"
  [ -s "$1" ] && [ "$(head -1 "$1" | tr -d '\r')" = k3-v130-review-pass ] && grep -qx "sha256=$2" "$1"
}
k3_gate_ok() {  # $1 verdict.json $2 tool sha $3 card class: a PASS of exactly that tool on exactly that card class
  python3 - "$1" "$2" "$3" <<'PY' 2>/dev/null
import json, sys
v = json.load(open(sys.argv[1]))
sys.exit(0 if v.get("verdict") == "pass" and v.get("tool_sha256") == sys.argv[2] and v.get("card_class") == sys.argv[3] else 1)
PY
}
k3_legacy_geometry() {  # $1 K3 out dir: a pass dir a <= 1.3.2 queue began (no .geometry) gets the geometry it ran with
  local k pd sha gt
  for k in 0 1 2 3; do
    pd=$1/pass-$k-of-4
    [ -e "$pd.geometry" ] && continue
    { [ -n "$(ls -A "$pd" 2>/dev/null)" ] && [ -s "$pd.queue_tool.sha256" ]; } || continue
    sha=$(cat "$pd.queue_tool.sha256")
    # the last attempt line names the group tokens it ran with (1.3.x switched to 32768 after an OOM)
    gt=$(grep -o "^== k3-pass-$k attempt [0-9]* group_tokens [0-9]*" "$QL/k3-pass-$k.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    printf '{"tool_sha256": "%s", "card_class": "%s", "group_tokens": %s, "by": "%s legacy (from %s)", "utc": "%s"}\n' \
      "$sha" "$K3_LEGACY_CLASS" "${gt:-null}" "$K3LIB_VERSION" "$QL/k3-pass-$k.log" "$(utc)" > "$pd.geometry"
    echo "$(utc) $pd: legacy pass dir, geometry recorded from its log: tool ${sha:0:16} class $K3_LEGACY_CLASS group tokens ${gt:-unknown}"
  done
}
k3_pick_pass() {  # $1 out dir $2 order ("0 1 2 3") $3 card class $4 tool sha $5 group tokens $6 allowed sealer shas (a,b)
  # $7 this runner's cmdline match.  Under flock $Q/k3_passes.lock (60 s at most): claims the first pass this runner may
  # run (not sealed, no live claim, no geometry of another class / tool / group tokens) and prints "PICK <k>"; otherwise
  # "NONE sealed=<ks> busy=<k:gpu> bound=<k:class/tool/gt> foreign=<k:sha16>".  A claim is live while its runner pid lives
  # with its cmd match, or while any process runs the pass (--pass-dir <pass dir> in its cmdline).
  local lock=${K3_PASSES_LOCK:-$Q/k3_passes.lock}
  ( flock -w 60 9 || { echo "NONE lock-timeout"; exit 0; }
    python3 - "$@" "$GPU" "$$" "$(utc)" <<'PY'
import json, os, sys
out, order, cls, tool, gt, allowed, match, gpu, pid, now = sys.argv[1:11]
allowed = {a for a in allowed.split(",") if a}
def cmdline(p):
    try:
        with open(f"/proc/{p}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""
def runs(pd):
    needle = f"--pass-dir {pd} "
    return [int(d) for d in os.listdir("/proc") if d.isdigit() and needle in cmdline(d)]
def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
sealed, busy, bound, foreign = [], [], [], []
for k in order.split():
    pd = os.path.join(out, f"pass-{k}-of-4")
    man = load(os.path.join(pd, "manifest.json")) or {}
    if man.get("verdict") == "sealed":
        if man.get("tool_sha256") in allowed:
            sealed.append(k)
        else:
            foreign.append(f"{k}:{str(man.get('tool_sha256'))[:16]}")
        continue
    claim = load(pd + ".claim") or {}
    mine = str(claim.get("pid")) == pid and claim.get("match") == match
    live = bool(runs(pd)) or bool(claim and claim.get("match") and claim["match"] in cmdline(claim.get("pid", 0)))
    if live and not mine:
        busy.append(f"{k}:gpu{claim.get('gpu', '?')}")
        continue
    geo = load(pd + ".geometry")
    if geo and (geo.get("card_class") != cls or geo.get("tool_sha256") != tool or str(geo.get("group_tokens")) != str(gt)):
        bound.append(f"{k}:{geo.get('card_class')}/{str(geo.get('tool_sha256'))[:16]}/{geo.get('group_tokens')}")
        continue
    tmp = f"{pd}.claim.tmp{pid}"
    with open(tmp, "w") as handle:
        json.dump({"pid": int(pid), "match": match, "gpu": gpu, "card_class": cls, "tool_sha256": tool,
                   "group_tokens": int(gt), "utc": now}, handle)
    os.replace(tmp, pd + ".claim")
    print(f"PICK {k}")
    sys.exit(0)
print(f"NONE sealed={','.join(sealed) or '-'} busy={','.join(busy) or '-'} bound={','.join(bound) or '-'} "
      f"foreign={','.join(foreign) or '-'}")
PY
  ) 9> "$lock"
}
k3_release_claim() {  # $1 pass dir: drop this runner's claim (a claim of another pid is never touched)
  python3 - "$1.claim" "$$" <<'PY' 2>/dev/null
import json, os, sys
try:
    if str(json.load(open(sys.argv[1])).get("pid")) == sys.argv[2]:
        os.remove(sys.argv[1])
except (OSError, ValueError):
    pass
PY
}
k3_geometry() {  # $1 pass dir $2 tool sha $3 card class $4 group tokens: recorded at the first start; rc 1 + why if it differs
  local pd=$1
  if [ ! -s "$pd.geometry" ]; then
    printf '{"tool_sha256": "%s", "card_class": "%s", "group_tokens": %s, "by": "%s", "utc": "%s"}\n' \
      "$2" "$3" "$4" "$K3LIB_VERSION" "$(utc)" > "$pd.geometry"
    return 0
  fi
  python3 - "$pd.geometry" "$2" "$3" "$4" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
want = {"tool_sha256": sys.argv[2], "card_class": sys.argv[3], "group_tokens": sys.argv[4]}
bad = [f"{k} {g.get(k)} != {v}" for k, v in want.items() if str(g.get(k)) != v]
print("; ".join(bad))
sys.exit(1 if bad else 0)
PY
}
k3_record_pass() {  # $1 k $2 pass dir $3 tool sha $4 tool version $5 card class $6 gate receipt -> $Q/k3_pass_tools.json
  local lock=${K3_PASSES_LOCK:-$Q/k3_passes.lock}
  ( flock -w 60 9 || exit 1
    python3 - "${K3_PASS_TOOLS:-$Q/k3_pass_tools.json}" "$@" "$GPU" "$(utc)" "$K3LIB_VERSION" <<'PY'
import hashlib, json, os, sys
path, k, pd, sha, ver, cls, gate, gpu, now, lib = sys.argv[1:11]
rec = json.load(open(path)) if os.path.exists(path) else {
    "schema": "omni-k3-pass-tools-v1", "what": "the tool, card class and gate that sealed each K3 pass (d2_plan reads it)",
    "passes": {}}
with open(os.path.join(pd, "manifest.json"), "rb") as handle:
    man_sha = hashlib.sha256(handle.read()).hexdigest()
rec["passes"][str(k)] = {"dir": pd, "tool_sha256": sha, "tool_version": ver, "card_class": cls, "gpu": int(gpu),
                         "gate_receipt": gate or None, "manifest_sha256": man_sha, "recorded_utc": now, "by": lib}
tmp = f"{path}.tmp{os.getpid()}"
with open(tmp, "w") as handle:
    json.dump(rec, handle, indent=1)
os.replace(tmp, path)
PY
  ) 9> "$lock"
}
k3_ballast() { ram_ballast "$1" "$K3_RES" "$K3_SLOT" "$QL/$K3_JOB.ram.log"; }
k3_size_pass() {  # $1 k: K3_RES (whole footprint GiB), K3_SLOT (one block slot GiB), K3_GATE (MemAvailable to start)
  local res
  res=$(k3_reserve "$KC/pass-$1/k3_input_ids.safetensors") || return 1
  K3_SLOT=${res##* }; K3_RES=${res%% *}
  K3_GATE=${K3_RAM_GIB:-$((K3_RES + K3_RAM_HEADROOM_GIB))}
}
k3_run_pass() {  # $1 k $2 pass dir.  Globals in: PASSTOOL PASSSHA PASSKIND (v124|v130) GT MOEB CARD_CLASS DIGEST
  # K3_STATE K3_GATE K3_WAIT_CARD (the caller's card-free wait).  Runs pass k to its seal under the thermal watchdog and
  # the RAM ballast (the caller has waited for RAM once); RC = the tool's exit; K3_END = PASS | HALT | FAIL, K3_NOTE why.
  local k=$1 pd=$2 attempt=0 th=0 kills=0 oom=0 moeb=${MOEB:-1} accel=() why off
  K3_JOB=k3-pass-$k; K3_END=FAIL; K3_NOTE=""; RC=1
  if ! why=$(k3_geometry "$pd" "$PASSSHA" "$CARD_CLASS" "$GT"); then
    K3_NOTE="geometry differs from the pass's first start ($why): refused, never resumed with other numerics"; RC=65
    return
  fi
  echo "$PASSSHA" > "$pd.queue_tool.sha256"
  GUARD_SIDECAR=k3_ballast
  while :; do
    attempt=$((attempt + 1))
    accel=()
    [ "$PASSKIND" = v130 ] && accel=(--no-fadvise-drop --kda-graph-keep kda-only --moe-group-batch "$moeb")
    off=$(stat -c %s "$QL/$K3_JOB.log" 2>/dev/null || echo 0)   # this attempt's output starts here
    echo "== $K3_JOB attempt $attempt group_tokens $GT tool ${PASSSHA:0:16} ($PASSKIND) class $CARD_CLASS gpu $GPU ${accel[*]} $(utc)" >> "$QL/$K3_JOB.log"
    run_guarded "$K3_JOB" "$QL/$K3_JOB.log" env CUDA_VISIBLE_DEVICES="$GPU" nice -n 15 "$PY" "$PASSTOOL" run \
      --input-ids "$KC/pass-$k/k3_input_ids.safetensors" --pass-count 1 --pass-index 0 \
      --pass-dir "$pd" --state-dir "$K3_STATE" \
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
      --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --group-tokens "$GT" \
      --checkpoint-every 2 --min-free-gib 60 $DIGEST "${accel[@]}" --resume
    echo "$K3_JOB attempt $attempt exit=$RC $(utc)"
    [ "$RC" -eq 0 ] && break
    if [ -s "$QL/$K3_JOB.thermal_stop" ]; then
      th=$((th + 1))
      if [ $th -gt "$THERMAL_RETRIES" ]; then K3_END=HALT; K3_NOTE="thermal: $THERMAL_RETRIES watchdog stops used (see $QL/$K3_JOB.thermal.log)"; break; fi
      status "$K3_JOB" THERMAL "stop $th of $THERMAL_RETRIES: $(cut -d' ' -f2- "$QL/$K3_JOB.thermal_stop"); cooling, then --resume"
      wait_cool; "$K3_WAIT_CARD"; wait_ram "$K3_GATE"; continue
    fi
    if [ "$RC" -eq 137 ] && [ $kills -lt 3 ]; then   # SIGKILL: the OOM killer (K3 is its preferred victim by design)
      kills=$((kills + 1))
      status "$K3_JOB" KILLED "SIGKILL $kills of 3 (oom_score_adj $K3_OOM_ADJ: preferred over peers; see $QL/$K3_JOB.ram.log); MemAvailable >= $K3_GATE GiB, then --resume"
      "$K3_WAIT_CARD"; wait_ram "$K3_GATE"; continue
    fi
    if tail -c +$((off + 1)) "$QL/$K3_JOB.log" | tail -8 | grep -q "OutOfMemoryError"; then  # THIS attempt's last lines only
      if [ "$PASSKIND" = v130 ] && [ "$moeb" != 1 ]; then  # the one numerics-neutral memory knob
        status "$K3_JOB" OOM "card out of memory at --moe-group-batch $moeb: --resume at 1 (each row group keeps its own rows, GEMMs and order; group tokens stay $GT)"
        moeb=1; "$K3_WAIT_CARD"; sleep "${K3_OOM_SLEEP_S:-60}"; continue
      fi
      oom=$((oom + 1))
      if [ $oom -le 1 ]; then
        status "$K3_JOB" OOM "card out of memory at group tokens $GT (moe-group-batch $moeb): one plain --resume once the card is free"
        "$K3_WAIT_CARD"; sleep "${K3_OOM_SLEEP_S:-60}"; continue
      fi
      K3_END=HALT; K3_NOTE="out of memory twice at group tokens $GT: needs a decision (a begun pass never changes its group tokens: its bf16 MoE numbers depend on them)"
      break
    fi
    K3_NOTE="exit=$RC (see $QL/$K3_JOB.log)"
    break
  done
  GUARD_SIDECAR=""
  if [ "$RC" -eq 0 ]; then
    if manifest_sealed_by "$pd/manifest.json" "$PASSSHA"; then K3_END=PASS; K3_NOTE="sealed $pd/manifest.json"
    else K3_END=FAIL; K3_NOTE="exit 0 but $pd/manifest.json is not sealed by ${PASSSHA:0:16}"; RC=70; fi
  fi
}
k3_choose_tool() {  # $1 review file $2 pinned 1.3.0 sha $3 frozen 1.3.0 copy $4 gate verdict.json $5 card class
  # prints "v130" when 1.3.0 may seal passes on this card class: the review passed for exactly the pinned sha, the
  # frozen copy hashes to it, and the gate on this class passed for it; otherwise "v124 <reason>" (the B2-proven tool)
  if ! k3_v130_review_ok "$1" "$2"; then echo "v124 no 1.3.0 review pass for sha ${2:0:16} ($1)"; return; fi
  if [ ! -s "$3" ] || [ "$(sha256sum "$3" | cut -c1-64)" != "$2" ]; then echo "v124 frozen 1.3.0 copy $3 absent or not ${2:0:16}"; return; fi
  if ! k3_gate_ok "$4" "$2" "$5"; then echo "v124 no 1.3.0 gate PASS on $5 ($4)"; return; fi
  echo v130
}
k3_all_sealed() {  # $1 out dir $2 allowed sealer shas (a,b): all four passes sealed by one of them
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, os, sys
out, allowed = sys.argv[1], set(sys.argv[2].split(","))
def sealed(k):
    try:
        m = json.load(open(os.path.join(out, f"pass-{k}-of-4", "manifest.json")))
    except (OSError, ValueError):
        return False
    return m.get("verdict") == "sealed" and m.get("tool_sha256") in allowed
sys.exit(0 if all(sealed(k) for k in range(4)) else 1)
PY
}
k3_probe() {  # $1 frozen probe tool $2 out dir $3 corpus dir $4 log: the K3 route probe over the 4 sealed passes (CPU);
  # once, under flock $Q/k3_probe.lock (the runner that seals the last pass runs it); RC 0 written or already there
  local args=() k
  [ -s "$2/route_probe_k3.json" ] && { RC=0; return; }
  for k in 0 1 2 3; do args+=(--pass "$2/pass-$k-of-4=$3/pass-$k"); done
  ( flock -w 10 9 || exit 75
    [ -s "$2/route_probe_k3.json" ] && exit 0
    CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$1" probe-distill "${args[@]}" --out "$2/route_probe_k3.json" > "$4" 2>&1
  ) 9> "${K3_PROBE_LOCK:-$Q/k3_probe.lock}"
  RC=$?
}
