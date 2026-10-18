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
#
# 1.2.0 (lane gpu6, plan Step 2b, 2026-10-14):
#  * FIX: the B2_OK test carried a literal "\n" inside its `if` (1.1.0 line 399), so bash ran a command named `n`
#    and B2_OK could never be 1 -- a passing B2 would have skipped every K3 pass. The test is now b2_ok(), and
#    GPU6_QUEUE_SELFTEST=b2ok checks it on synthetic receipts (1.1.0 has no such path: its selftest cannot pass).
#  * a FRESH B2 dir on Lexar ($LEXAR/distill/parity/k3cache-2026-10-14-b2-gpu6, receipts copied to
#    $OC/distill/parity/k3cache-2026-10-14-b2-gpu6); the 2026-10-13 B2 dir is refused as a B2 dir (no --resume onto it)
#  * B2 RE-BASELINE, only under $Q/k3_b2_rebaseline_ruling.txt (k3_b2_geometry_diag: the L2 split is bf16 batch-shape
#    numerics in the routed MoE, not padding/state): the bf16 review pass at production geometry is judged against an
#    fp32 review pass of the SAME tool and geometry (k3_teacher_cache export-reference at the fold dump's positions),
#    with parity's code and PARITY_THRESHOLDS unchanged; bf16-vs-fold and fp32-vs-fold are recorded as diagnostics.
#    Without the ruling file B2 is the 1.1.0 comparison against the fold dump.
#  * K3 passes wait for MemAvailable >= 90 GiB (K3_RAM_GIB; was 135): wait, never kill  [1.3.0: the pass footprint + headroom]
#  * H3: the launcher's temperature stop (run_media_a_wave2.sh, the card's 82 C max operating temperature) is abort +
#    resume (the media-a tool resumes from its sealed shards), up to THERMAL_RETRIES per step -- never a gate
#  * j6: the vision-in / video-in answer cache, last, only when lane vl has written $Q/gpu0.vl_go, GPU0 is still busy
#    and j6 is neither done nor running; under flock $Q/j6.lock (runs once); lane vl's $Q/j6_vl.sh if present
#
# 1.3.0 (review fixes, 2026-10-14):
#  * FOLD-IDENTITY GATE (re-baseline mode): bf16-vs-fp32 of one tool is a self-comparison, and the diagnosis proved
#    identity only at L00-L02 (all KDA).  b2-identity runs between the bf16 and the fp32 pass: k3_b2_geometry_diag
#    1.1.0 --mode identity (frozen at launch as $Q/frozen/k3_b2_geometry_diag-gpu6.py) over the fold's own corpus --
#    the single-call prefix sha at EVERY layer L00-L92 against the fold teacher sha, and the frozen tool's production
#    CachePass.layer on identical inputs at every MLA layer, every block push, L1/L2/L61/L92 (group tokens 65,536 =
#    the K3 passes' geometry).  b2_ok needs its PASS receipt (by the frozen K3 tool's sha, all 93 layers); a FAIL is
#    functional: the fp32 pass and the K3 cache do not start.  Thermal stops resume from its checkpoints.
#  * K3 HOST RAM: a pass's AttnRes block slots are torch.empty and fill over ~45 h, so MemAvailable at launch shows
#    only part of the pass.  The gate is now the pass's WHOLE footprint (k3_reserve: state_bytes of the pass file's
#    positions + measured overhead) + K3_RAM_HEADROOM_GIB, and a sidecar ballast (ram_ballast) holds the part the
#    pass has not touched yet as committed memory, shrinking as the pass grows, so a later MemAvailable-gated job
#    sees the real remainder.  The ballast yields below RAM_FLOOR_GIB (never the cause of pressure) and sets its own
#    oom_score_adj 1000 and the K3 process's K3_OOM_ADJ (500): under OOM the resumable K3 pass goes before a peer
#    (D1).  A K3 pass killed by SIGKILL resumes from its checkpoint (up to 3x) after the RAM gate.
#
# 1.3.1 (lane infra, cards X-T2, 2026-10-14; no job, gate or tool changes):
#  * LANE HEARTBEAT: every status line, every 30 min of a thermal watch and every 30 min of a RAM wait heartbeats lane
#    L-K3 in $Q/LANES.json (lanes.py heartbeat --source "$VERSION"); a failing or missing lanes.py never stops the queue
#    (20 s cap, the watch's heartbeat runs in the background).  Selftests other than `lanes` never heartbeat.
#  * CARD LOCK: the queue holds $Q/gpu6.lock (plan 7.4.1; other launchers check it before admitting a GPU6 job) through
#    a `lanes.py hold-lock --while-pid $$` helper, so the jobs it starts never inherit the lock and a job adopted by the
#    next queue does not block it.  The lock is waited for (CARD_LOCK_WAIT_S, 600), never competed for; still held
#    after that -> refuse (exit 2) before any job.  GPU6_QUEUE_SELFTEST=lanes checks both on temp files.
#  * an unknown GPU6_QUEUE_SELFTEST value is refused (exit 2); 1.3.0 ran the REAL queue on one (never test 1.3.0 so).
# 1.3.2 (lane infra review, 2026-10-15; no job, gate or tool changes; needs lanes.py >= 1.0.1):
#  * the card lock no longer goes free while a GPU6 job outlives its queue: the helper holds it while this queue lives
#    OR any process matches CARD_JOB_MATCH ($Q/frozen/ = the b2 / identity / K3 pass / probe tools, and $Q/j6_vl.sh).
#    1.3.1 released it with the queue, so an upgrade or a crash left GPU6 busy behind a free gpu6.lock.
#  * a new queue meeting an ORPHANED helper (its queue gone, holding only for the job this queue adopts) joins it
#    (--join-helper): it starts at once and its own helper takes the lock the moment the old one releases.  A helper
#    whose queue still lives is never joined: a second queue still waits CARD_LOCK_WAIT_S and refuses (exit 2).
#  * not covered by CARD_JOB_MATCH: H3 (run_media_a_wave2.sh; denied on GPU6 without a thermal ruling) and gpu0.sh's
#    job_j6 fallback ($T/qtext_teacher_cache.py build-vl, run in the foreground under j6.lock).
set -u
VERSION="queue_gpu6.sh 1.3.2"
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
B2_BASE=${B2_BASE:-$LEXAR/distill/parity/k3cache-2026-10-14-b2-gpu6}
B2_OLD=$OC/distill/parity/k3cache-2026-10-13-b2-gpu6
B2_RCPT=$OC/distill/parity/k3cache-2026-10-14-b2-gpu6
REBASE_RULING=$Q/k3_b2_rebaseline_ruling.txt
K3_RAM_HEADROOM_GIB=${K3_RAM_HEADROOM_GIB:-6}
# host overhead beside the state model: bisect peak RSS 32.1 GB over a 16.4 GB state model (143,360 positions);
# B2 1.2.4 20.05 GB RSS with ~1.9 GB of state touched at L07 -> ~16-18 GB measured; 20 GiB reserved
K3_OVERHEAD_GIB=${K3_OVERHEAD_GIB:-20}
K3_OOM_ADJ=${K3_OOM_ADJ:-500}
RAM_FLOOR_GIB=${RAM_FLOOR_GIB:-8}
J6_RAM_GIB=${J6_RAM_GIB:-70}
ID_RAM_GIB=${ID_RAM_GIB:-70}      # identity job: ~53 GB host (2 x [prefix 2.1 GB + 8 block slots 16.4 GB] + overhead)
ID_GT=${ID_GT:-65536}             # the K3 passes' group tokens; an OOM retries at 32768 (resumes)
ID_THREADS=${ID_THREADS:-22}      # the fold receipt's runtime num_threads
REF=$OC/k3-run-oct09-claude/fold/review
SNAP=$LEXAR/k3-continuation-oct08-claude/fold/snapshots/before_L61.safetensors
KC=$OC/distill/corpus/k3
K3_STATE_BASE=$LEXAR/scratch/k3cache-state
K3_STALE=$LEXAR/scratch/k3cache-stale
CACHE_ROOT=$LEXAR/teacher_cache
TPS=$OC/distill/tps
START_AT=${START_AT:-b2}
THERMAL_RETRIES=${THERMAL_RETRIES:-6}
LANE=L-K3
LANES_PY=${LANES_PY:-$T/lanes.py}
LANES_FILE=${LANES_FILE:-$Q/LANES.json}
GPU6_LOCK=${GPU6_LOCK:-$Q/gpu6.lock}
CARD_LOCK_WAIT_S=${CARD_LOCK_WAIT_S:-600}
CARD_JOB_MATCH=${CARD_JOB_MATCH:-$Q/frozen/|$Q/j6_vl.sh}  # '|'-separated: a live process matching one keeps gpu6.lock
mkdir -p "$QL" "$ALLOCD" "$TPS" "$Q/frozen"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src

utc() { date -u +%FT%TZ; }
lane_hb() {  # $1 lane status $2 detail: heartbeat $LANE in $LANES_FILE; never fatal, at most 20 s
  case "${GPU6_QUEUE_SELFTEST:-}" in ''|lanes) ;; *) return 0;; esac
  [ -f "$LANES_PY" ] || return 0
  timeout 20 python3 "$LANES_PY" --file "$LANES_FILE" heartbeat --lane "$LANE" --agent claude --source "$VERSION" \
    --status "$1" --detail "$2" > /dev/null 2>&1 || echo "$(utc) lane heartbeat $LANE failed (ignored)"
  return 0
}
status() {  # $1 job $2 state $3 note
  local note="${3:-}"
  echo "$(utc) $1 $2 $note" | tee -a "$STATUS"
  lane_hb "$([ "$1 $2" = "queue END" ] && echo ended || echo running)" "$1 $2 ${note:0:300}"
}
take_card_lock() {  # $1 lock $2 wait s: a lanes.py hold-lock helper holds $1 while this queue ($$) or a CARD_JOB_MATCH
  # job lives; RC 0 once held, or once joined behind an orphaned helper (LOCK_STATE = "<helper pid>[ joined <pids>]")
  local ready i m pats also=()
  IFS='|' read -ra pats <<< "$CARD_JOB_MATCH"
  for m in "${pats[@]}"; do [ -n "$m" ] && also+=(--also-match "$m"); done
  ready=$(mktemp -u "${TMPDIR:-/tmp}/gpu6-card-lock.$$.XXXXXX")
  python3 "$LANES_PY" hold-lock --lock "$1" --while-pid $$ --cmd-match gpu6.sh "${also[@]}" --join-helper \
    --wait "$2" --poll 10 --ready-file "$ready" < /dev/null &
  LOCK_HELPER=$!
  for i in $(seq 1 $(( $2 + 30 ))); do
    if [ -s "$ready" ]; then LOCK_STATE=$(cat "$ready"); rm -f "$ready"; return 0; fi
    [ -d /proc/$LOCK_HELPER ] || return 1
    sleep 1
  done
  kill "$LOCK_HELPER" 2>/dev/null; return 1
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
reached() {  # true once START_AT has been passed
  case " $ORDER " in *" $START_AT "*) ;; *) echo "unknown START_AT $START_AT"; exit 2;; esac
  [ "$PASSED_START" = 1 ] && return 0
  [ "$1" = "$START_AT" ] && { PASSED_START=1; return 0; }
  status "$1" SKIP "before START_AT=$START_AT"
  return 1
}
ORDER="b2 k3-pass-0 k3-pass-1 k3-pass-2 k3-pass-3 k3-probe h3-p4-bf16 h3-p4-fp16 h3-encode h3-text h3-velocity j6"
PASSED_START=0

echo "$VERSION start $(utc) START_AT=$START_AT host $(hostname)"
# the K3 tool B2 proves is the tool the passes run: frozen once, by sha; REFREEZE=1 archives it by sha first
freeze() {  # $1 frozen path $2 refreeze flag [$3 source, default the live k3_teacher_cache]
  local dst=$1 old src=${3:-$T/k3_teacher_cache.py}
  if [ -e "$dst" ] && [ "$2" = 1 ]; then
    old=$(sha256sum "$dst" | cut -c1-16)
    mv -f "$dst" "${dst%.py}.$old.py"; echo "refreeze: $dst archived as ${dst%.py}.$old.py"
  fi
  if [ ! -e "$dst" ]; then cp -p "$src" "$dst"; chmod a-w "$dst"; fi
}
K3TOOL=$Q/frozen/k3_teacher_cache-gpu6.py
PROBETOOL=$Q/frozen/k3_teacher_cache-gpu6-probe.py
IDTOOL=$Q/frozen/k3_b2_geometry_diag-gpu6.py
freeze "$K3TOOL" "${REFREEZE:-0}"
freeze "$PROBETOOL" "${REFREEZE_PROBE:-0}"
freeze "$IDTOOL" "${REFREEZE_ID:-0}" "$T/k3_b2_geometry_diag.py"
K3SHA=$(sha256sum "$K3TOOL" | cut -c1-64)
K3VER=$("$PY" "$K3TOOL" --version 2>/dev/null)
PROBESHA=$(sha256sum "$PROBETOOL" | cut -c1-64)
PROBEVER=$("$PY" "$PROBETOOL" --version 2>/dev/null)
IDSHA=$(sha256sum "$IDTOOL" | cut -c1-64)
IDVER=$(PYTHONPATH=$T "$PY" "$IDTOOL" --version 2>/dev/null)
echo "frozen K3 tool $K3TOOL $K3VER sha256 $K3SHA"
echo "frozen probe tool $PROBETOOL $PROBEVER sha256 $PROBESHA"
echo "frozen identity tool $IDTOOL $IDVER sha256 $IDSHA"
if [ "$(realpath -m "$B2_BASE")" = "$(realpath -m "$B2_OLD")" ]; then
  echo "refuse: $B2_BASE is the 2026-10-13 B2 dir; B2 runs in a fresh dir and never resumes from it"; exit 2
fi
REBASE=0; REBASE_ID=""
if [ -s "$REBASE_RULING" ]; then REBASE=1; REBASE_ID=$(head -1 "$REBASE_RULING"); fi
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
echo "B2 reference: $([ $REBASE = 1 ] && echo "fp32 pass of the frozen tool at production geometry (ruling $REBASE_ID)" || echo "the fold's review dump $REF")"
echo "thermal: stop ${STOP_C} C x3 polls, hard ${HARD_C} C, resume <= ${RESUME_C} C; card max operating $T_MAXOP slowdown $T_SLOW shutdown $T_SHUT; retries $THERMAL_RETRIES"
K3_PLAN=$(k3_reserve "$KC/pass-0/k3_input_ids.safetensors" 2>/dev/null)
echo "B2 identity gate: $([ $REBASE = 1 ] && echo "$IDVER --mode identity (L00-L92 vs the fold teacher sha + production closed loop, group tokens $ID_GT, $ID_THREADS threads) in $B2/identity" || echo "none (B2 compares with the fold dump itself)")"
echo "RAM gates: K3 pass = whole footprint ${K3_PLAN%% *} GiB (pass-0; one block slot ${K3_PLAN##* } GiB) + $K3_RAM_HEADROOM_GIB GiB headroom, ballast to that footprint (floor $RAM_FLOOR_GIB GiB, K3 oom_score_adj $K3_OOM_ADJ); identity >= $ID_RAM_GIB GiB; j6 >= $J6_RAM_GIB GiB (wait, never kill)"
if [ "${GPU6_QUEUE_SELFTEST:-}" = plan ]; then echo "SELFTEST plan: exit before any job"; exit 0; fi
if [ "${GPU6_QUEUE_SELFTEST:-}" = b2ok ]; then  # b2_ok on synthetic receipts; no status lines, no GPU work
  D=$(mktemp -d); mkdir -p "$D/review-pass" "$D/review-pass-fp32" "$D/identity"
  seal() { echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"$2\"}" > "$1/manifest.json"; }
  verdict() { echo "{\"verdict\": \"$1\"}" > "$D/parity_b2_receipt.json"; }
  ident() { echo "{\"mode\": \"identity\", \"frozen_tool\": {\"sha256\": \"$2\"}, \"verdict\": {\"pass\": $1, \"layers_checked\": $3, \"layers_expected\": 93}}" > "$D/identity/b2_identity.json"; }
  r() { if b2_ok "$D" "$K3SHA" "$1"; then echo 1; else echo 0; fi; }
  seal "$D/review-pass" "$K3SHA"; verdict pass;  echo "ST b2ok pass: $(r 0)"
  verdict fail;                                  echo "ST b2ok fail: $(r 0)"
  verdict pass; seal "$D/review-pass" other;     echo "ST b2ok other-tool: $(r 0)"
  seal "$D/review-pass" "$K3SHA"; rm -f "$D/review-pass-fp32/manifest.json"; echo "ST b2ok rebase-no-fp32: $(r 1)"
  seal "$D/review-pass-fp32" "$K3SHA";           echo "ST b2ok rebase-no-identity: $(r 1)"
  ident false "$K3SHA" 93;                       echo "ST b2ok rebase-identity-fail: $(r 1)"
  ident true other 93;                           echo "ST b2ok rebase-identity-other-tool: $(r 1)"
  ident true "$K3SHA" 61;                        echo "ST b2ok rebase-identity-incomplete: $(r 1)"
  ident true "$K3SHA" 93;                        echo "ST b2ok rebase-pass: $(r 1)"
  verdict fail;                                  echo "ST b2ok rebase-identity-pass-parity-fail: $(r 1)"
  rm -f "$D/parity_b2_receipt.json";             echo "ST b2ok no-receipt: $(r 0)"
  rm -rf "$D"; exit 0
fi
if [ "${GPU6_QUEUE_SELFTEST:-}" = ballast ]; then  # ram_ballast on a dummy job that grows 3 x 1 GiB; ~5 GiB, no GPU
  QL=$(mktemp -d)
  python3 -c "
import mmap, time
held = []
for i in range(3):
    time.sleep(6)
    m = mmap.mmap(-1, 1 << 30); v = memoryview(m); v[::4096] = b'\x02' * (1 << 18); v.release(); held.append(m)
time.sleep(12)" & DP=$!
  RAM_FLOOR_GIB=1 ram_ballast "$DP" 4 0 "$QL/b.log" & BP=$!
  sleep 4; ADJ=$(cat /proc/$DP/oom_score_adj 2>/dev/null)
  wait "$DP"; wait "$BP"
  held=$(grep -o 'ballast [0-9]* GiB' "$QL/b.log" | tr -dc '0-9\n')
  echo "ST ballast: first=$(echo "$held" | head -1) max=$(echo "$held" | sort -n | tail -1) last=$(echo "$held" | tail -1) released=$(grep -c 'ballast released' "$QL/b.log") job_oom_adj=$ADJ"
  sleep 300 & DP=$!
  RAM_FLOOR_GIB=100000 ram_ballast "$DP" 4 0 "$QL/f.log" & BP=$!
  sleep 8; kill "$DP"; wait "$BP"
  echo "ST ballast-floor: max=$(grep -o 'ballast [0-9]* GiB' "$QL/f.log" | tr -dc '0-9\n' | sort -n | tail -1)"
  cat "$QL/b.log"; rm -rf "$QL"; exit 0
fi
if [ "${GPU6_QUEUE_SELFTEST:-}" = lanes ]; then  # 1.3.1: lane heartbeat + card lock on temp files; no status lines, no GPU
  D=$(mktemp -d); LANES_FILE=$D/LANES.json; CARD_JOB_MATCH="sleep 611.2"  # never the live $Q/frozen/ jobs
  echo '{"created_by": "queue_gpu6 selftest", "lanes": [{"lane": "L-K3", "owner": "claude", "scope": "selftest",
   "lock_paths": [], "status": "registered", "transferable": false, "transferable_reason": "selftest"}]}' > "$D/seed.json"
  python3 "$LANES_PY" --file "$LANES_FILE" init --seed "$D/seed.json" > /dev/null
  lane_hb running "b2 START selftest"
  echo "ST lanes heartbeat: $(python3 -c 'import json, sys
l = json.load(open(sys.argv[1]))["lanes"]["L-K3"]
print(l["heartbeat_source"], "|", l["status"], "|", l["detail"])' "$LANES_FILE")"
  OUTN=$(LANES_FILE=$D/absent/LANES.json lane_hb running x); echo "ST lanes no-registry: rc=$? said=$(echo "$OUTN" | grep -c 'failed (ignored)')"
  LANES_PY=$D/absent.py lane_hb running x; echo "ST lanes no-tool: rc=$?"
  take_card_lock "$D/gpu6.lock" 5; echo "ST lock free: rc=$?"
  H1=$LOCK_HELPER
  take_card_lock "$D/gpu6.lock" 3; echo "ST lock held: rc=$?"
  kill "$H1"; wait "$H1" 2>/dev/null
  echo "ST lock released: $(python3 -c 'import sys; sys.path.insert(0, sys.argv[2]); import lanes
print(len(lanes.lock_holders(sys.argv[1])))' "$D/gpu6.lock" "$(dirname "$LANES_PY")")"
  # 1.3.2: an earlier queue's helper that holds only for a job (its queue gone) is joined, then handed the lock
  setsid sleep 611.1 < /dev/null > /dev/null 2>&1 & FQ=$!
  setsid sleep 611.2 < /dev/null > /dev/null 2>&1 & FJ=$!
  python3 "$LANES_PY" hold-lock --lock "$D/gpu6.lock" --while-pid $FQ --cmd-match "sleep 611.1" --also-match "sleep 611.2" \
    --poll 1 > "$D/old_helper.log" 2>&1 < /dev/null & OLDH=$!
  sleep 2; kill $FQ; wait $FQ 2>/dev/null; sleep 2
  echo "ST lock job-holds: $([ -d /proc/$OLDH ] && echo yes || echo no)"
  take_card_lock "$D/gpu6.lock" 3; echo "ST lock join: rc=$? state=$([ "$LOCK_STATE" = "$LOCK_HELPER joined $OLDH" ] && echo joined-old || echo "$LOCK_STATE")"
  kill $FJ; wait $FJ 2>/dev/null; wait $OLDH 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do
    HOLDER=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[2]); import lanes
print(",".join(str(h["pid"]) for h in lanes.lock_holders(sys.argv[1])))' "$D/gpu6.lock" "$(dirname "$LANES_PY")")
    [ "$HOLDER" = "$LOCK_HELPER" ] && break; sleep 0.5
  done
  echo "ST lock handover: $([ "$HOLDER" = "$LOCK_HELPER" ] && echo new-helper || echo "holder=$HOLDER")"
  kill "$LOCK_HELPER"; wait "$LOCK_HELPER" 2>/dev/null
  rm -rf "$D"; exit 0
fi
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
if [ -n "${GPU6_QUEUE_SELFTEST:-}" ]; then  # 1.3.0 and earlier fell through to the REAL queue on an unknown mode
  echo "refuse: unknown GPU6_QUEUE_SELFTEST=$GPU6_QUEUE_SELFTEST; no job started"; exit 2
fi
if ! take_card_lock "$GPU6_LOCK" "$CARD_LOCK_WAIT_S"; then
  echo "refuse: $GPU6_LOCK is still held after $CARD_LOCK_WAIT_S s (another GPU6 queue that still lives, or a holder"
  echo "        that is not a hold-lock helper: lanes.py show --lane $LANE); no job started"; exit 2
fi
case "$LOCK_STATE" in
  *joined*) echo "card lock $GPU6_LOCK: joined behind orphaned helper(s) ${LOCK_STATE##* } (holding for the job this queue adopts); helper pid $LOCK_HELPER takes it when they release" ;;
  *) echo "card lock $GPU6_LOCK held by helper pid $LOCK_HELPER while pid $$ or a '$CARD_JOB_MATCH' job lives" ;;
esac
mkdir -p "$B2"; [ -s "$B2/queue_tool.sha256" ] || echo "$K3SHA" > "$B2/queue_tool.sha256"
status queue START "$VERSION k3tool=$K3VER sha256=${K3SHA:0:16} probetool=$PROBEVER sha256=${PROBESHA:0:16} b2dir=$B2 b2ref=$([ $REBASE = 1 ] && echo "fp32 ($REBASE_ID)" || echo fold) thermal stop=${STOP_C}x3/hard=${HARD_C}/resume=${RESUME_C} identity=$IDVER:${IDSHA:0:16} k3ram=${K3_PLAN%% *}+${K3_RAM_HEADROOM_GIB}GiB(pass-0) start_at=$START_AT"

# ---------------------------------------------------------------- b2: the deciding parity run
review_pass() {  # $1 job $2 pass dir $3 state dir $4 compute dtype; rest = extra run args. Sets RC.
  local job=$1 pd=$2 sd=$3 dt=$4 logf=$QL/$1_pass.log oom=0 th=0 attempt=0
  shift 4
  while :; do
    attempt=$((attempt + 1))
    if [ "$job" = b2 ] && [ -n "$ADOPT" ]; then
      echo "== $job attempt $attempt: adopted pid $ADOPT by $VERSION $(utc)" >> "$logf"
      adopt_guarded b2 "$ADOPT" "$B2_MAN" "$logf"; ADOPT=""
    else
      echo "== $job attempt $attempt $(utc) compute dtype $dt" >> "$logf"
      run_guarded "$job" "$logf" env CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$K3TOOL" run \
        --input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" \
        --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" \
        --pass-dir "$pd" --state-dir "$sd" "$@" \
        --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype "$dt" \
        --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --no-trim --group-tokens 32768 \
        --resume
    fi
    echo "$job attempt $attempt exit=$RC $(utc)"
    [ $RC -eq 0 ] && break
    if [ -s "$QL/$job.thermal_stop" ]; then
      th=$((th + 1))
      [ $th -gt "$THERMAL_RETRIES" ] && { echo "$job: thermal stop $th > $THERMAL_RETRIES retries"; break; }
      status "$job" THERMAL "stop $th of $THERMAL_RETRIES: $(cut -d' ' -f2- "$QL/$job.thermal_stop"); cooling, then --resume"
      wait_cool; wait_card; continue
    fi
    tail -5 "$logf" | grep -q "OutOfMemoryError" || break
    oom=$((oom + 1)); [ $oom -ge 3 ] && break
    sleep 60
  done
}
identity_run() {  # the fold-identity gate on GPU6; sets RC (0 PASS, 3 FAIL); thermal stops cool + --resume; an OOM at
  # the K3 group size retries at 32768 (the single-call chain and its checkpoints do not depend on the group size)
  local logf=$QL/b2-identity_pass.log th=0 attempt=0 gt=$ID_GT C=$OC/composite/corpus
  while :; do
    attempt=$((attempt + 1))
    echo "== b2-identity attempt $attempt $(utc) group tokens $gt threads $ID_THREADS tool $IDVER ${IDSHA:0:16}" >> "$logf"
    run_guarded b2-identity "$logf" env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=$ID_THREADS MKL_NUM_THREADS=$ID_THREADS \
      OPENBLAS_NUM_THREADS=$ID_THREADS nice -n 15 "$PY" "$IDTOOL" --mode identity \
      --frozen-tool "$K3TOOL" --frozen-sha "$K3SHA" \
      --input-ids "$C/k3_input_ids.safetensors" --input-ids "$C/k3_review_input_ids.safetensors" \
      --input-ids "$C/k3_review_native_input_ids.safetensors" \
      --group-tokens "$gt" --closed-loop-layers structural --min-mem-available-gib "$ID_RAM_GIB" \
      --out "$B2/identity" --resume
    echo "b2-identity attempt $attempt exit=$RC $(utc)"
    { [ $RC -eq 0 ] || [ $RC -eq 3 ]; } && break
    if [ -s "$QL/b2-identity.thermal_stop" ]; then
      th=$((th + 1))
      [ $th -gt "$THERMAL_RETRIES" ] && { echo "b2-identity: thermal stop $th > $THERMAL_RETRIES retries"; break; }
      status b2-identity THERMAL "stop $th of $THERMAL_RETRIES: $(cut -d' ' -f2- "$QL/b2-identity.thermal_stop"); cooling, then --resume"
      wait_cool; wait_card; continue
    fi
    if [ "$gt" != 32768 ] && tail -30 "$logf" | grep -q "OutOfMemoryError"; then
      status b2-identity OOM "group tokens $gt out of memory; --resume at 32768"; gt=32768; sleep 60; continue
    fi
    break
  done
}
B2_OK=0
B2_MAN=$B2/review-pass/manifest.json
B2F_MAN=$B2/review-pass-fp32/manifest.json
ID_RCPT=$B2/identity/b2_identity.json
FAILJOB=b2
if reached b2; then
  RC=1
  if [ -s "$B2/parity_b2_receipt.json" ]; then
    # one frozen tool, one B2 verdict: a re-run of the same tool reproduces it (and --resume would reuse its layers)
    RC=-1
    if receipt_passed "$B2/parity_b2_receipt.json"; then
      status b2 PASS "already passed with frozen tool ${K3SHA:0:16} ($B2/parity_b2_receipt.json)"
    else
      status b2 REFUSED "frozen tool ${K3SHA:0:16} already failed B2 ($B2/parity_b2_receipt.json); fix the tool, then REFREEZE=1 START_AT=b2 (fresh tool-keyed B2 dir)"
    fi
  else
    if manifest_sealed_by "$B2_MAN" "$K3SHA"; then
      RC=0; echo "b2 bf16 pass already sealed by tool ${K3SHA:0:16}"
    else
      if [ -n "$ADOPT" ]; then
        alloc b2 adopted "adopted by $VERSION: pid $ADOPT started by the earlier queue, now under the thermal watchdog"
        status b2 ADOPT "pid $ADOPT (started by the earlier queue) under the thermal watchdog; log $QL/b2_pass.log"
      else
        status b2 WAIT "S8 pid $S8_PID + GPU$GPU < 2000 MiB"
        wait_card
        alloc b2 started "K3 full-pass review parity, production geometry (lab venv, kda torch, row chunks 1, token chunk 1024, no trim, group tokens 32768)$([ $REBASE = 1 ] && echo ", bf16 then the fp32 reference pass")"
        status b2 START "bf16 review pass, log $QL/b2_pass.log"
      fi
      review_pass b2 "$B2/review-pass" "$B2/state" auto --keep-state
    fi
    if [ $RC -eq 0 ] && [ $REBASE = 1 ]; then
      FAILJOB=b2-identity
      if identity_ok "$ID_RCPT" "$K3SHA"; then
        echo "b2 fold-identity gate already PASSED for tool ${K3SHA:0:16} ($ID_RCPT)"
      else
        status b2-identity START "fold-identity gate: $IDVER (${IDSHA:0:16}) --mode identity over the fold's corpus, single call L00-L92 vs the fold teacher sha + the production CachePass.layer closed loop (group tokens $ID_GT); log $QL/b2-identity_pass.log"
        wait_card
        identity_run
        if [ $RC -eq 0 ] && ! identity_ok "$ID_RCPT" "$K3SHA"; then echo "b2-identity: exit 0 but $ID_RCPT is not a PASS by ${K3SHA:0:16}"; RC=71; fi
        mkdir -p "$B2_RCPT"; cp -p "$ID_RCPT" "$B2_RCPT/" 2>/dev/null
        if [ $RC -eq 0 ]; then
          status b2-identity PASS "$(grep -o 'IDENTITY .*' "$QL/b2-identity_pass.log" | tail -1 | head -c 400)"
          rm -f "$B2"/identity/state/*.safetensors   # its checkpoints (~18.5 GB) only served a resume; state.json kept
        fi
      fi
    fi
    if [ $RC -eq 0 ] && [ $REBASE = 1 ]; then
      FAILJOB=b2-fp32
      if manifest_sealed_by "$B2F_MAN" "$K3SHA"; then
        echo "b2 fp32 reference pass already sealed by tool ${K3SHA:0:16}"
      else
        status b2-fp32 START "fp32 reference review pass, same tool and geometry (ruling $REBASE_ID), log $QL/b2-fp32_pass.log"
        wait_card
        review_pass b2-fp32 "$B2/review-pass-fp32" "$B2/state-fp32" fp32
      fi
    fi
  fi
  if [ $RC -eq -1 ]; then
    :
  elif [ $RC -ne 0 ]; then
    alloc b2 ended "$FAILJOB pass failed; thermal: $(thermal_note $FAILJOB)" $RC
    if [ -s "$QL/$FAILJOB.thermal_stop" ]; then status b2 HALT "$FAILJOB thermal: $THERMAL_RETRIES watchdog stops used; last $(cut -d' ' -f2- "$QL/$FAILJOB.thermal_stop") (not a numerics verdict)"
    elif [ "$FAILJOB" = b2-identity ] && [ $RC -eq 3 ]; then status b2 FAIL "fold-identity gate FAILED (functional: identity / structural): $(grep -o 'IDENTITY .*' "$QL/b2-identity_pass.log" | tail -1 | head -c 400); fp32 pass and K3 cache NOT started; receipt $ID_RCPT"
    else status b2 FAIL "$FAILJOB pass exit=$RC (functional; see $QL/${FAILJOB}_pass.log); K3 cache NOT started"; fi
  else
    PRC=0
    if [ $REBASE = 1 ]; then
      # the fp32 pass in the fold dump's layout at the fold dump's own (row, position)s; then the UNCHANGED parity
      mkdir -p "$B2/ref-fp32"
      : > "$QL/b2_export.log"
      for f in k3_review_input_ids k3_review_native_input_ids; do
        CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" export-reference --cache-dir "$B2/review-pass-fp32" \
          --positions-from "$REF/${f}_teacher_forced.safetensors" --out "$B2/ref-fp32/${f}_teacher_forced.safetensors" \
          >> "$QL/b2_export.log" 2>&1 || PRC=4
      done
      cat "$QL/b2_export.log"
      GATE_REF=("$B2/ref-fp32/k3_review_input_ids_teacher_forced.safetensors" "$B2/ref-fp32/k3_review_native_input_ids_teacher_forced.safetensors")
      # diagnostics (never gates): each pass against the fold's bf16 one-call dump
      for which in review-pass:bf16 review-pass-fp32:fp32; do
        CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" parity --cache-dir "$B2/${which%%:*}" \
          --reference "$REF/k3_review_input_ids_teacher_forced.safetensors" \
          --reference "$REF/k3_review_native_input_ids_teacher_forced.safetensors" \
          --out "$B2/parity_b2_${which#*:}_vs_fold.json" > "$QL/b2_parity_${which#*:}_vs_fold.log" 2>&1
        echo "diagnostic ${which#*:} vs fold exit=$?"; cat "$QL/b2_parity_${which#*:}_vs_fold.log"
      done
    else
      GATE_REF=("$REF/k3_review_input_ids_teacher_forced.safetensors" "$REF/k3_review_native_input_ids_teacher_forced.safetensors")
    fi
    if [ $PRC -eq 0 ]; then
      CUDA_VISIBLE_DEVICES="" nice -n 15 "$PY" "$K3TOOL" parity --cache-dir "$B2/review-pass" \
        --reference "${GATE_REF[0]}" --reference "${GATE_REF[1]}" \
        --out "$B2/parity_b2_receipt.json" > "$QL/b2_parity.log" 2>&1
      PRC=$?
      cat "$QL/b2_parity.log"
    else
      echo "export-reference failed: no gate parity (functional)" | tee "$QL/b2_parity.log"
    fi
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
    alloc b2 ended "parity exit=$PRC; thermal: $(thermal_note b2) $(thermal_note b2-fp32)" $PRC
    mkdir -p "$B2_RCPT"
    cp -p "$B2"/parity_b2*.json "$B2"/b2_block_compare.json "$B2_RCPT/" 2>/dev/null
    [ -d "$B2/ref-fp32" ] && cp -p "$B2"/ref-fp32/*.safetensors "$B2_RCPT/" 2>/dev/null
    REFNOTE=$([ $REBASE = 1 ] && echo "vs the fp32 pass (ruling $REBASE_ID; fold-identity gate PASS $ID_RCPT)" || echo "vs the fold dump")
    if [ $PRC -eq 0 ]; then
      status b2 PASS "$REFNOTE $(grep -o 'PARITY PASS.*' "$QL/b2_parity.log" | head -c 300)"
      # the kept checkpoints were only for a bisect: remove this run's own files, nothing else
      for d in "$B2"/state/*/; do
        [ -f "$d/state.json" ] || continue
        rm -f "$d"state.json "$d"block_*.safetensors "$d"prefix_L*.safetensors && rmdir "$d" 2>/dev/null
      done
      rmdir "$B2/state" 2>/dev/null
      echo "b2 kept state removed after PASS: $(ls -A "$B2/state" 2>/dev/null | wc -l) entries left"
    else
      status b2 FAIL "$REFNOTE parity exit=$PRC $(grep -o 'PARITY FAIL.*' "$QL/b2_parity.log" | head -c 300); K3 cache NOT started; verdict in $B2_RCPT"
    fi
  fi
fi
# the K3 passes run only on THIS frozen tool's B2 PASS: the parity receipt passed AND the B2 manifest(s) it read were
# sealed by this very tool (a relaunch past b2, a refreeze, or a re-run all land here)
B2_OK=0
if b2_ok "$B2" "$K3SHA" "$REBASE"; then B2_OK=1; fi
echo "B2 verdict for frozen tool ${K3SHA:0:16} from $B2/parity_b2_receipt.json + $B2_MAN (rebase $REBASE) -> B2_OK=$B2_OK"

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
  # the pass's WHOLE host footprint (its block slots are torch.empty and fill over the pass): gate on it, and hold the
  # untouched part as a ballast so later MemAvailable-gated jobs see the real remainder
  K3_RES=$(k3_reserve "$KC/pass-$k/k3_input_ids.safetensors") || { status $job FAIL "cannot size $KC/pass-$k (k3_reserve)"; K3_ALL=0; break; }
  K3_SLOT=${K3_RES##* }; K3_RES=${K3_RES%% *}
  K3_GATE=${K3_RAM_GIB:-$((K3_RES + K3_RAM_HEADROOM_GIB))}
  k3_ballast() { ram_ballast "$1" "$K3_RES" "$K3_SLOT" "$QL/$job.ram.log"; }
  status $job WAIT "S8/GPU$GPU, /mnt/lexar >= 200 GB free, MemAvailable >= $K3_GATE GiB (footprint $K3_RES GiB + $K3_RAM_HEADROOM_GIB; wait, never kill)"
  wait_card; wait_lexar 200; wait_ram $K3_GATE
  mkdir -p "$K3OUT" "$K3_STATE"
  echo "$K3SHA" > "$PD.queue_tool.sha256"
  DIGEST=""; [ $k -eq 0 ] && DIGEST="--inline-digest"
  alloc $job started "K3 teacher cache pass $k of 4 ($KC/pass-$k), group tokens 65536, frozen tool ${K3SHA:0:16}, thermal stop ${STOP_C}x3/hard ${HARD_C}, host reserve $K3_RES GiB (ballast, slack one slot $K3_SLOT GiB)"
  status $job START "log $QL/$job.log out $PD state $K3_STATE thermal $QL/$job.thermal.log ram $QL/$job.ram.log"
  RC=1; GT=65536; OOM=0; TH=0; KILLS=0; attempt=0
  GUARD_SIDECAR=k3_ballast
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
      wait_cool; wait_card; wait_ram $K3_GATE; continue
    fi
    if [ $RC -eq 137 ] && [ $KILLS -lt 3 ]; then   # SIGKILL: the OOM killer (K3 is its preferred victim by design)
      KILLS=$((KILLS + 1))
      status $job KILLED "SIGKILL $KILLS of 3 (oom_score_adj $K3_OOM_ADJ: preferred over peers; see $QL/$job.ram.log); MemAvailable >= $K3_GATE GiB, then --resume"
      wait_card; wait_ram $K3_GATE; continue
    fi
    tail -8 "$QL/$job.log" | grep -q "OutOfMemoryError" || break
    OOM=$((OOM + 1)); [ $OOM -ge 4 ] && break
    GT=32768   # same token chunks (T = 4096 rows, chunk 1024): resumable, numerics unchanged
    sleep 60
  done
  GUARD_SIDECAR=""
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
wait_launch_temp() {  # run_media_a_wave2.sh refuses a card within 5 C of its stop line (max operating): wait below it
  local t n=0
  while :; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i $GPU | tr -dc 0-9)
    [ -n "$t" ] && [ "$t" -le $((T_MAXOP - 6)) ] && break
    [ $((n % 10)) -eq 0 ] && echo "$(utc) GPU$GPU at ${t} C > $((T_MAXOP - 6)) C (launcher margin); waiting"
    n=$((n + 1)); sleep 30
  done
}
media_step() {  # $1 job $2 launcher step, rest = extra tool args; waits for the detached docker job to end.
  # The launcher's temperature stop (the card's max operating temperature) is abort + resume: cool, relaunch (the
  # media-a tool resumes from its sealed shards), up to THERMAL_RETRIES; telemetry is the launcher's end receipt.
  local job=$1 step=$2; shift 2
  local out alloc_path rc fired th=0
  while :; do
    wait_card; wait_launch_temp
    out=$(bash "$T/run_media_a_wave2.sh" "$step" $GPU "$@" 2>&1); rc=$?
    echo "$out"
    if [ $rc -ne 0 ]; then status $job FAIL "launcher refused: $(echo "$out" | tail -1 | head -c 300)"; return 2; fi
    alloc_path=$(echo "$out" | grep -o 'allocator [^ ]*' | tail -1 | cut -d' ' -f2)
    alloc $job started "media-a $step via run_media_a_wave2.sh; its allocator $alloc_path"
    status $job START "$(echo "$out" | grep -o 'log [^ ]*' | tail -1)"
    while ! grep -q '"state":"ended"' "$alloc_path" 2>/dev/null; do sleep 60; done
    rc=$(python3 -c "import json;print(json.load(open('$alloc_path'))['exit'])")
    fired=$(python3 -c "import json;r=json.load(open('$alloc_path'));print(int(r.get('thermal_stop_fired') or 0), r.get('peak_temp_c'))")
    alloc $job ended "media-a $step exit=$rc thermal_stop_fired/peak_c=$fired" "$rc"
    if [ "$rc" != 0 ] && [ "${fired%% *}" = 1 ]; then
      th=$((th + 1))
      if [ $th -gt "$THERMAL_RETRIES" ]; then status $job HALT "launcher thermal stop $th > $THERMAL_RETRIES retries (not a numerics verdict)"; return "$rc"; fi
      status $job THERMAL "launcher stop $th of $THERMAL_RETRIES (peak ${fired#* } C); cooling, then relaunch (resumes from sealed shards)"
      wait_cool; continue
    fi
    return "$rc"
  done
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
# ---------------------------------------------------------------- j6: vision-in / video-in answers (handoff)
# GPU0's j6 (qtext build-vl over corpus/vision_in) runs here last, only when lane vl has released it
# ($Q/gpu0.vl_go), GPU0 is still busy (D1), j6 is neither done nor running, and this queue holds $Q/j6.lock.
# Lane vl's $Q/j6_vl.sh (J6_GPU=<index> in the environment) is used when it exists; otherwise gpu0.sh's job_j6
# command with GPU6 as the device.  HostResidentReader holds ~54 GB: MemAvailable >= J6_RAM_GIB first.
J6_LOCK=$Q/j6.lock
J6_DONE=$Q/j6.done
if reached j6; then
  g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -dc 0-9)
  if [ -e "$J6_DONE" ] || grep -q "job=j6-vl state=done" "$Q/gpu0.status" 2>/dev/null; then
    status j6 SKIP "already done ($J6_DONE / gpu0.status)"
  elif [ ! -f "$Q/gpu0.vl_go" ]; then
    status j6 SKIP "lane vl has not released j6 ($Q/gpu0.vl_go absent)"
  elif [ "${g0:-0}" -lt 2000 ]; then
    status j6 SKIP "GPU0 is free (${g0} MiB): j6 belongs to GPU0"
  elif pgrep -f "qtext_teacher_cache.py build-vl" > /dev/null; then
    status j6 SKIP "a qtext build-vl is already running: $(pgrep -af 'qtext_teacher_cache.py build-vl' | head -1 | cut -c1-200)"
  else
    exec 8> "$J6_LOCK"
    if ! flock -n 8; then
      status j6 SKIP "$J6_LOCK is held by another runner"
    else
      status j6 WAIT "GPU$GPU, MemAvailable >= $J6_RAM_GIB GiB"
      wait_card; wait_ram "$J6_RAM_GIB"
      HEAD=fp32
      [ -f "$Q/gpu0.head_decision.json" ] && HEAD=$("$PY" -c "import json;print(json.load(open('$Q/gpu0.head_decision.json'))['head_precision'])")
      UUID6=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
      alloc j6 started "j6 VL answers handoff from gpu0 (go: $(head -c 200 "$Q/gpu0.vl_go" | tr '\n' ' '))"
      if [ -x "$Q/j6_vl.sh" ] || [ -s "$Q/j6_vl.sh" ]; then
        status j6 START "lane vl's $Q/j6_vl.sh (sha256 $(sha256sum "$Q/j6_vl.sh" | cut -c1-16)) on GPU$GPU, log $QL/j6-vl.log"
        J6_GPU=$GPU CUDA_VISIBLE_DEVICES=$UUID6 bash "$Q/j6_vl.sh" >> "$QL/j6-vl.log" 2>&1 < /dev/null
        RC=$?
      else
        status j6 START "gpu0.sh job_j6 command on GPU$GPU (head $HEAD), qtext sha256 $(sha256sum "$T/qtext_teacher_cache.py" | cut -c1-16), log $QL/j6-vl.log"
        RC=1
        for attempt in 1 2; do
          CUDA_VISIBLE_DEVICES=$UUID6 nice -n 15 "$PY" "$T/qtext_teacher_cache.py" build-vl \
            --model-dir /home/cereal/models/Qwen3.8-27B-Uncensored --device cuda:0 --dtype bfloat16 \
            --attn-implementation sdpa --linear-attn-kernel auto --digest-threads 8 --teacher q38-27bu \
            --corpus "$OC/distill/corpus/vision_in/vision_in.jsonl" --corpus "$OC/distill/corpus/vision_in/video_in.jsonl" \
            --out-root "$LEXAR/teacher_cache/qtext" --micro-batch 8 --max-batch-tokens 32768 --batch-order length \
            --rows-per-shard 64 --head-precision "$HEAD" --chat-template-kwargs '{"enable_thinking": false}' \
            --purpose "gpu6 j6 handoff: T(i) vision_in + video_in answer caches, teacher-forced, top-64" \
            --allocator-receipt "$ALLOCD/gpu6-qtext-vl.json" >> "$QL/j6-vl.log" 2>&1 < /dev/null
          RC=$?
          echo "j6 attempt $attempt exit=$RC $(utc)" >> "$QL/j6-vl.log"
          { [ $RC -eq 0 ] || [ $RC -eq 2 ]; } && break   # qtext: 2 is a functional failure and final
          sleep 60
        done
      fi
      alloc j6 ended "exit=$RC" $RC
      if [ $RC -eq 0 ]; then
        echo "$(utc) j6 done on GPU$GPU by $VERSION (log $QL/j6-vl.log)" > "$J6_DONE"
        status j6 PASS "$(grep -c 'SEALED' "$QL/j6-vl.log") sealed lines; $J6_DONE"
      else
        status j6 FAIL "exit=$RC (see $QL/j6-vl.log)"
      fi
    fi
    exec 8>&-
  fi
fi
status queue END "$VERSION"
echo "$VERSION end $(utc)"
