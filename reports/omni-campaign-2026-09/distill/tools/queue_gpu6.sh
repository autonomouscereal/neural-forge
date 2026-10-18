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
#   k3-v130-gate  [1.4.0] ONLY IF b2 PASSES and 1.3.0 was reviewed: 1.3.0 vs 1.2.4 bit-identity on GPU6 (see 1.4.0)
#   k3-pass-0..3  ONLY IF b2 PASSES: the K3 2.0M-token cache, one pass per $OC/distill/corpus/k3/pass-k file
#                 [1.4.0: claimed one at a time; k3_multicard.sh may run some of them on other cards]
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
# 1.4.0 (lane k3plan, 2026-10-15; K3_ACCELERATION_PLAN_2026-10-14.md; needs k3_lib.sh 1.0.0 beside it in $T):
#  * SHARED HELPERS: wait_lexar .. b2_ok moved verbatim into $T/k3_lib.sh (k3_multicard.sh runs K3 passes on the extra
#    cards with the same code); the thermal thresholds are read by k3lib_thermal_init for $GPU.
#  * PASS GEOMETRY IS FIXED (review 2026-10-15 item 1, must-fix): an out-of-memory no longer resumes a begun pass at
#    32,768 group tokens (1.3.2 line 842, "numerics unchanged" was false: the routed MoE's bf16 numbers depend on the
#    GEMM row count).  Every pass records tool sha + card class + group tokens in <pass dir>.geometry at its first start
#    and a resume that differs is refused.  OOM: 1.3.0 drops --moe-group-batch to 1 (numerics-neutral, parity w4), then
#    one plain retry, then HALT for a decision.
#  * k3-v130-gate (new job, after b2): only with $Q/k3_v130_review.txt (line 1 "k3-v130-review-pass", a line
#    "sha256=<the 1.3.0 sha>"): frozen 1.3.0 ($Q/frozen/k3_teacher_cache-gpu6-v130.py, pinned sha) vs the B2-proven
#    frozen 1.2.4 ON GPU6 at the passes' flags, every prefix / block slot hashed per layer (k3_cache_parity 1.0.0):
#    w1 L84-85, w2p L83 (MLA at T=4,096), w4 L84-85 in four groups, w1 NEG (+1 ulp after L84 must DIFFER).  PASS ->
#    the K3 passes run 1.3.0 (--no-fadvise-drop --kda-graph-keep kda-only --moe-group-batch $K3_MOEB); anything else
#    keeps 1.2.4.  Receipt $Q/k3_v130_gate/<card class>/verdict.json.
#  * PASS CLAIMS: GPU6 takes the lowest unsealed pass nobody else runs (k3_pick_pass under flock $Q/k3_passes.lock);
#    k3_multicard.sh takes them from the top on other cards; $Q/k3_gpu6.max_passes (re-read before every pick, else
#    K3_MAX_PASSES, default 4) caps the passes GPU6 starts.  When GPU6 has no pass left to run it goes on to H3 / j6;
#    the route probe runs where the last pass seals.  Each seal is recorded in $Q/k3_pass_tools.json (d2_plan must
#    read it: a pass sealed by 1.3.0 or on another card class is not "sealed by $Q/frozen/k3_teacher_cache-gpu6.py").
#  * a b2-identity run or a b2-fp32 pass left by an earlier queue is ADOPTED (watched to its end) instead of waited out
#    and run again.  An OOM is now read from the current attempt's output only (k3_run_pass; the b2 / identity
#    loops keep 1.3.2's tail check).
#  * card lock: unchanged from 1.3.2 (lanes.py hold-lock on $Q/gpu6.lock, an flock held while this queue or a GPU6 job
#    lives).
set -u
VERSION="queue_gpu6.sh 1.4.0"
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
K3LIB=${K3LIB:-$T/k3_lib.sh}
K3LIB_WANT="k3_lib.sh 1.0.0"
V130_SRC=${V130_SRC:-$T/k3_teacher_cache_v130.py}
V130_PIN=${K3_V130_SHA:-434ced6d7e8e22e4179bb2332d4695803dd89a38407805126261500c112fa3c7}  # the reviewed 1.3.0
V130_REVIEW=${V130_REVIEW:-$Q/k3_v130_review.txt}
K3_MOEB=${K3_MOEB:-4}            # 1.3.0 --moe-group-batch on GPU6: 4 = the P40-proven count (w4); auto (~7) never ran
PARITY_PY=${PARITY_PY:-$T/k3_cache_parity.py}
K3_MAX_PASSES=${K3_MAX_PASSES:-4}  # passes GPU6 may start; $Q/k3_gpu6.max_passes overrides it at run time
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
# ---------------------------------------------------------------- shared K3 helpers (1.4.0: k3_lib.sh)
# wait_lexar, wait_ram, k3_reserve, ram_ballast, tq, thermal_watch, run_guarded, adopt_guarded, manifest_sealed_by,
# wait_cool, thermal_note, receipt_passed, identity_ok, b2_ok (verbatim from 1.3.2) + the K3 pass contract
[ -s "$K3LIB" ] || { echo "refuse: $K3LIB missing (queue_gpu6.sh 1.4.0 needs $K3LIB_WANT)"; exit 2; }
. "$K3LIB"
[ "${K3LIB_VERSION:-}" = "$K3LIB_WANT" ] || { echo "refuse: $K3LIB is '${K3LIB_VERSION:-?}', this queue needs '$K3LIB_WANT'"; exit 2; }
k3lib_thermal_init
CARD_CLASS=$(k3_card_class "$GPU")
[ -n "$CARD_CLASS" ] || { echo "refuse: GPU$GPU card class unreadable (nvidia-smi name, compute_cap)"; exit 2; }
reached() {  # true once START_AT has been passed
  case " $ORDER " in *" $START_AT "*) ;; *) echo "unknown START_AT $START_AT"; exit 2;; esac
  [ "$PASSED_START" = 1 ] && return 0
  [ "$1" = "$START_AT" ] && { PASSED_START=1; return 0; }
  status "$1" SKIP "before START_AT=$START_AT"
  return 1
}
ORDER="b2 k3-v130-gate k3-pass-0 k3-pass-1 k3-pass-2 k3-pass-3 k3-probe h3-p4-bf16 h3-p4-fp16 h3-encode h3-text h3-velocity j6"
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
V130TOOL=$Q/frozen/k3_teacher_cache-gpu6-v130.py
V130SHA=""; V130VER=""
if k3_v130_review_ok "$V130_REVIEW" "$V130_PIN"; then
  if [ ! -e "$V130TOOL" ] && [ "$(sha256sum "$V130_SRC" | cut -c1-64)" = "$V130_PIN" ]; then
    cp -p "$V130_SRC" "$V130TOOL"; chmod a-w "$V130TOOL"
  fi
  if [ -e "$V130TOOL" ] && [ "$(sha256sum "$V130TOOL" | cut -c1-64)" = "$V130_PIN" ]; then
    V130SHA=$V130_PIN; V130VER=$("$PY" "$V130TOOL" --version 2>/dev/null)
  fi
fi
echo "1.3.0 candidate: ${V130VER:-none} ${V130SHA:+sha256 $V130SHA} (review $V130_REVIEW: $(k3_v130_review_ok "$V130_REVIEW" "$V130_PIN" && echo pass || echo absent); pin ${V130_PIN:0:16}); card class $CARD_CLASS; lib $K3LIB_VERSION $(sha256sum "$K3LIB" | cut -c1-16)"
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
echo "K3 passes: tool decided after b2 + k3-v130-gate: $(k3_choose_tool "$V130_REVIEW" "$V130_PIN" "$V130TOOL" "$Q/k3_v130_gate/$CARD_CLASS/verdict.json" "$CARD_CLASS"); group tokens 65536 (fixed per pass), 1.3.0 --moe-group-batch $K3_MOEB"
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
identity_adopt() {  # $1 pid of a b2-identity run an earlier queue started: watch it to its end, then RC as identity_run
  local pid=$1 wpid logf=$QL/b2-identity_pass.log
  rm -f "$QL/b2-identity.thermal_stop"
  echo "== b2-identity adopted pid $pid by $VERSION $(utc)" >> "$logf"
  thermal_watch "$pid" b2-identity "$logf" &
  wpid=$!
  while [ -d /proc/$pid ]; do sleep 20; done
  wait "$wpid" 2>/dev/null
  if identity_ok "$ID_RCPT" "$K3SHA"; then RC=0; return; fi
  if python3 -c "import json,sys;v=json.load(open(sys.argv[1])).get('verdict') or {};sys.exit(0 if v.get('pass') is False else 1)" "$ID_RCPT" 2>/dev/null; then RC=3; return; fi
  if [ -s "$QL/b2-identity.thermal_stop" ]; then
    status b2-identity THERMAL "adopted run stopped: $(cut -d' ' -f2- "$QL/b2-identity.thermal_stop"); cooling, then --resume"; wait_cool
  fi
  wait_card; identity_run   # ended without a verdict: resume it the normal way
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
        status b2-identity START "$(pgrep -f -- "$IDTOOL --mode identity" > /dev/null && echo "ADOPT running pid; ")fold-identity gate: $IDVER (${IDSHA:0:16}) --mode identity over the fold's corpus, single call L00-L92 vs the fold teacher sha + the production CachePass.layer closed loop (group tokens $ID_GT); log $QL/b2-identity_pass.log"
        ADOPT_ID=$(pgrep -f -- "$IDTOOL --mode identity" | head -1)
        if [ -n "$ADOPT_ID" ]; then
          alloc b2 adopted "b2-identity pid $ADOPT_ID adopted by $VERSION (started by an earlier queue)"
          identity_adopt "$ADOPT_ID"
        else
          wait_card
          identity_run
        fi
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
        ADOPT_FP=$(pgrep -f -- "$K3TOOL run .*--pass-dir $B2/review-pass-fp32 --state-dir" | head -1)
        if [ -n "$ADOPT_FP" ]; then
          echo "== b2-fp32 adopted pid $ADOPT_FP by $VERSION $(utc)" >> "$QL/b2-fp32_pass.log"
          adopt_guarded b2-fp32 "$ADOPT_FP" "$B2F_MAN" "$QL/b2-fp32_pass.log"
          if [ $RC -ne 0 ]; then
            [ -s "$QL/b2-fp32.thermal_stop" ] && wait_cool
            wait_card; review_pass b2-fp32 "$B2/review-pass-fp32" "$B2/state-fp32" fp32
          fi
        else
          wait_card
          review_pass b2-fp32 "$B2/review-pass-fp32" "$B2/state-fp32" fp32
        fi
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

# ---------------------------------------------------------------- k3-v130-gate: 1.3.0 == 1.2.4 bit for bit ON GPU6
# The P40 proved it (lane k3accel, 2026-10-14: w1-w4, b2geo, mini, head61 BIT-IDENTICAL on GPU3); GPU6 runs torch 2.11 on
# sm_75, where the KDA graph, the sorted MoE and the shared dequant have never run.  Windows on real pass-0 rows at the
# passes' flags; A = the B2-proven frozen 1.2.4, B = the frozen 1.3.0 with the pass flags.  PASS = every A-B compare
# EQUAL and the NEG compare DIFFER.  Anything else keeps 1.2.4 for the passes: never a K3 failure.
GATE_DIR=$Q/k3_v130_gate/$CARD_CLASS
GATE_VERDICT=$GATE_DIR/verdict.json
gate_side() {  # $1 run dir $2 tag $3 A|B|NEG, rest = window args; RC 0 = digest written (thermal stops cool + rerun)
  local R=$1 tag=$2 side=$3 th=0 tool=$K3TOOL tsha=$K3SHA inj=() acc=()
  shift 3
  if [ "$side" != A ]; then tool=$V130TOOL; tsha=$V130SHA; acc=(--no-fadvise-drop --kda-graph-keep kda-only --moe-group-batch "$K3_MOEB"); fi
  [ "$side" = NEG ] && inj=(--inject-ulp 84)
  while :; do
    rm -rf "$R/${tag}_$side-probe"
    echo "== gate $tag $side $(utc) tool ${tsha:0:16}" >> "$QL/k3-v130-gate.log"
    run_guarded k3-v130-gate "$R/${tag}_$side.log" env CUDA_VISIBLE_DEVICES=$GPU nice -n 15 "$PY" "$PARITY_PY" digest \
      --tool "$tool" --tool-sha "$tsha" --out "$R/digest_${tag}_$side.json" "${inj[@]}" -- \
      run "$@" --probe --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto \
      --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --pass-count 1 --pass-index 0 --no-head \
      "${acc[@]}" --pass-dir "$R/${tag}_$side-probe"
    echo "gate $tag $side exit=$RC $(utc) : $(grep -h 'digest 1\.' "$R/${tag}_$side.log" | tail -1)" | tee -a "$QL/k3-v130-gate.log"
    [ "$RC" -eq 0 ] && return 0
    if [ -s "$QL/k3-v130-gate.thermal_stop" ] && [ $th -lt "$THERMAL_RETRIES" ]; then
      th=$((th + 1)); status k3-v130-gate THERMAL "$tag $side stop $th of $THERMAL_RETRIES; cooling, then rerun"
      wait_cool; wait_card; continue
    fi
    return 1
  done
}
v130_gate() {  # the gate windows on GPU6; writes $GATE_VERDICT; GATE_RC 0 pass, 3 differ, 4 inconclusive
  local R=$LEXAR/scratch/k3accel/gpu6-gate-$(date -u +%Y%m%dT%H%M%SZ) P0=$KC/pass-0/k3_input_ids.safetensors w rc
  local -A CMP=()
  mkdir -p "$R" "$GATE_DIR"
  GATE_RC=0
  for w in w1 w2p w4; do
    case $w in
      w1)  set -- --input-ids "$P0" --row-limit 16 --layers 84-85 --synthetic-blocks 7 --group-tokens 65536 ;;
      w2p) set -- --input-ids "$P0" --row-limit 16 --layers 83 --synthetic-blocks 7 --group-tokens 65536 ;;
      w4)  set -- --input-ids "$P0" --row-limit 16 --layers 84-85 --synthetic-blocks 7 --group-tokens 16384 ;;
    esac
    gate_side "$R" "$w" A "$@" && gate_side "$R" "$w" B "$@" || { GATE_RC=4; break; }
    "$PY" "$PARITY_PY" compare "$R/digest_${w}_A.json" "$R/digest_${w}_B.json" --out "$R/compare_${w}_A_B.json" \
      --lines "$R/compare_${w}_A_B.lines.txt" --quiet | tee -a "$QL/k3-v130-gate.log"
    rc=${PIPESTATUS[0]}; CMP[$w]=$rc
    [ "$rc" -eq 0 ] || { GATE_RC=$([ "$rc" -eq 3 ] && echo 3 || echo 4); break; }
    if [ $w = w1 ]; then  # the negative control: +1 ulp after L84 must be seen
      gate_side "$R" w1 NEG "$@" || { GATE_RC=4; break; }
      "$PY" "$PARITY_PY" compare "$R/digest_w1_A.json" "$R/digest_w1_NEG.json" --out "$R/compare_w1_A_NEG.json" \
        --lines "$R/compare_w1_A_NEG.lines.txt" --quiet | tee -a "$QL/k3-v130-gate.log"
      rc=${PIPESTATUS[0]}; CMP[w1_NEG]=$rc
      [ "$rc" -eq 3 ] || { GATE_RC=4; echo "negative control did not DIFFER (exit $rc): the harness cannot see a 1-ulp change"; break; }
    fi
  done
  python3 - "$GATE_VERDICT" "$GATE_RC" "$V130SHA" "$K3SHA" "$CARD_CLASS" "$GPU" "$R" "$(sha256sum "$PARITY_PY" | cut -c1-64)" \
    "$VERSION" "w1=${CMP[w1]:-} w2p=${CMP[w2p]:-} w4=${CMP[w4]:-} w1_NEG=${CMP[w1_NEG]:-}" <<'PY'
import glob, hashlib, json, os, sys, time
out, rc, b, a, cls, gpu, run, psha, ver, cmps = sys.argv[1:11]
verdict = {"0": "pass", "3": "differ"}.get(rc, "inconclusive")
compares = {}
for path in sorted(glob.glob(os.path.join(run, "compare_*.json"))):
    with open(path, "rb") as handle:
        compares[os.path.basename(path)] = hashlib.sha256(handle.read()).hexdigest()
rec = {"schema": "omni-k3-v130-gate-v1", "verdict": verdict, "tool_sha256": b, "reference_tool_sha256": a,
       "card_class": cls, "gpu": int(gpu), "run_dir": run, "compare_exits": cmps, "compare_receipts_sha256": compares,
       "parity_tool_sha256": psha, "rule": "every A-B compare EQUAL (exit 0) and w1 NEG DIFFER (exit 3); else 1.2.4 runs",
       "by": ver, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tmp = out + ".tmp"
with open(tmp, "w") as handle:
    json.dump(rec, handle, indent=1)
os.replace(tmp, out)
print(f"GATE {verdict} {cmps} -> {out}")
PY
}
if reached k3-v130-gate; then
  if [ "$B2_OK" != 1 ]; then status k3-v130-gate SKIP "B2 did not pass for frozen tool ${K3SHA:0:16}"
  elif [ -z "$V130SHA" ]; then status k3-v130-gate SKIP "no 1.3.0 review pass ($V130_REVIEW) or no frozen 1.3.0 at ${V130_PIN:0:16}: the passes run the B2-proven 1.2.4"
  elif k3_gate_ok "$GATE_VERDICT" "$V130SHA" "$CARD_CLASS"; then status k3-v130-gate PASS "already passed ($GATE_VERDICT)"
  else
    status k3-v130-gate WAIT "GPU$GPU free, MemAvailable >= 40 GiB"
    wait_card; wait_ram 40
    status k3-v130-gate START "frozen 1.3.0 ${V130SHA:0:16} vs frozen 1.2.4 ${K3SHA:0:16} on $CARD_CLASS: w1, w2p, w4, w1 NEG; log $QL/k3-v130-gate.log"
    alloc k3-v130-gate started "1.3.0 vs 1.2.4 bit-identity windows on GPU$GPU (k3_cache_parity)"
    v130_gate
    alloc k3-v130-gate ended "gate exit $GATE_RC ($GATE_VERDICT)" "$GATE_RC"
    case $GATE_RC in
      0) status k3-v130-gate PASS "BIT-IDENTICAL on $CARD_CLASS, NEG DIFFER ($GATE_VERDICT)" ;;
      3) status k3-v130-gate DIFFER "1.3.0 differs from 1.2.4 on $CARD_CLASS ($GATE_VERDICT): the passes run 1.2.4" ;;
      *) status k3-v130-gate INCONCLUSIVE "gate did not complete ($GATE_VERDICT, $QL/k3-v130-gate.log): the passes run 1.2.4" ;;
    esac
  fi
fi
PASSKIND=v124; PASSTOOL=$K3TOOL; PASSSHA=$K3SHA; PASSVER=$K3VER; PASSGATE=""
CHOICE=$(k3_choose_tool "$V130_REVIEW" "$V130_PIN" "$V130TOOL" "$GATE_VERDICT" "$CARD_CLASS")
if [ "$CHOICE" = v130 ]; then PASSKIND=v130; PASSTOOL=$V130TOOL; PASSSHA=$V130SHA; PASSVER=$V130VER; PASSGATE=$GATE_VERDICT; fi
echo "K3 pass tool on $CARD_CLASS: $PASSKIND $PASSVER ${PASSSHA:0:16} ($CHOICE)"

# ---------------------------------------------------------------- K3 2.0M cache, 4 passes, claimed one at a time
K3SHA16=$(python3 -c "import json;print(json.load(open('$KC/manifest.json'))['manifest_sha256'][:16])")
K3OUT=$CACHE_ROOT/k3/$K3SHA16
K3_ALL=0; K3_LEFT=""; K3_RUN=0; K3_MIN=0
case "$START_AT" in k3-pass-[0-3]) K3_MIN=${START_AT#k3-pass-} ;; esac
for j in k3-pass-0 k3-pass-1 k3-pass-2 k3-pass-3; do reached $j && K3_RUN=1; done
if [ "$K3_RUN" = 1 ] && [ "$B2_OK" != 1 ]; then
  for k in 0 1 2 3; do status k3-pass-$k SKIP "B2 did not pass for frozen tool ${K3SHA:0:16}"; done
elif [ "$K3_RUN" = 1 ]; then
  mkdir -p "$K3OUT"
  k3_legacy_geometry "$K3OUT"
  GT=65536; MOEB=$K3_MOEB; K3_WAIT_CARD=wait_card
  TOOLS="$PASSKIND"; [ "$PASSKIND" = v130 ] && TOOLS="v130 v124"   # a pass begun by 1.2.4 finishes on 1.2.4
  K3_STARTED=0
  while :; do
    # how many passes GPU6 itself may start: $Q/k3_gpu6.max_passes (re-read before every pick, so the lead can hand the
    # rest to faster cards without restarting this queue), else K3_MAX_PASSES (default all 4)
    MAXP=$(tr -dc 0-9 < "$Q/k3_gpu6.max_passes" 2>/dev/null); MAXP=${MAXP:-$K3_MAX_PASSES}
    if [ "$K3_STARTED" -ge "$MAXP" ]; then
      K3_LEFT="GPU$GPU limit $MAXP pass(es) reached ($Q/k3_gpu6.max_passes / K3_MAX_PASSES)"; break
    fi
    PICK=""
    for kind in $TOOLS; do
      if [ "$kind" = v130 ]; then T_TOOL=$V130TOOL; T_SHA=$V130SHA; T_VER=$V130VER; T_GATE=$GATE_VERDICT
      else T_TOOL=$K3TOOL; T_SHA=$K3SHA; T_VER=$K3VER; T_GATE=""; fi
      PICK=$(k3_pick_pass "$K3OUT" "$(seq -s ' ' "$K3_MIN" 3)" "$CARD_CLASS" "$T_SHA" "$GT" "$K3SHA,$V130_PIN" "distill/queues/gpu6.sh")
      case "$PICK" in "PICK "*) break ;; esac
    done
    case "$PICK" in
      "PICK "*) ;;
      *) K3_LEFT=${PICK#NONE }; break ;;
    esac
    k=${PICK#PICK }; job=k3-pass-$k; PD=$K3OUT/pass-$k-of-4
    PASSKIND=$kind; PASSTOOL=$T_TOOL; PASSSHA=$T_SHA; PASSVER=$T_VER; PASSGATE=$T_GATE
    K3_STATE=$K3_STATE_BASE/tool-${PASSSHA:0:16}
    mkdir -p "$K3_STATE"
    if ! k3_size_pass "$k"; then status $job FAIL "cannot size $KC/pass-$k (k3_reserve)"; k3_release_claim "$PD"; break; fi
    status $job WAIT "S8/GPU$GPU, /mnt/lexar >= 200 GB free, MemAvailable >= $K3_GATE GiB (footprint $K3_RES GiB + $K3_RAM_HEADROOM_GIB; wait, never kill)"
    wait_card; wait_lexar 200; wait_ram $K3_GATE
    DIGEST=""; [ "$k" -eq 0 ] && DIGEST="--inline-digest"
    alloc $job started "K3 teacher cache pass $k of 4 ($KC/pass-$k), group tokens $GT, $PASSKIND $PASSVER ${PASSSHA:0:16}, thermal stop ${STOP_C}x3/hard ${HARD_C}, host reserve $K3_RES GiB (ballast, slack one slot $K3_SLOT GiB)"
    status $job START "$PASSKIND ${PASSSHA:0:16} on $CARD_CLASS, group tokens $GT; log $QL/$job.log out $PD state $K3_STATE thermal $QL/$job.thermal.log ram $QL/$job.ram.log"
    K3_STARTED=$((K3_STARTED + 1))
    k3_run_pass "$k" "$PD"
    alloc $job ended "exit=$RC $K3_END; thermal: $(thermal_note $job)" "$RC"
    k3_release_claim "$PD"
    if [ "$K3_END" = PASS ]; then
      k3_record_pass "$k" "$PD" "$PASSSHA" "$PASSVER" "$CARD_CLASS" "$PASSGATE"
      status $job PASS "$K3_NOTE ($PASSKIND ${PASSSHA:0:16}; recorded in $Q/k3_pass_tools.json)"
    else
      status $job "$K3_END" "$K3_NOTE; resumable with START_AT=$job; later K3 passes not started by this queue"
      break
    fi
  done
  case "$K3_LEFT" in
    *foreign=-*|"") ;;
    *foreign=*) status k3 FAIL "a K3 pass is sealed by a tool other than ${K3SHA:0:16} / ${V130_PIN:0:16}: decide by hand (not moved) [$K3_LEFT]" ;;
  esac
  if k3_all_sealed "$K3OUT" "$K3SHA,$V130_PIN"; then K3_ALL=1
  elif [ -n "$K3_LEFT" ]; then echo "$(utc) K3: nothing left for GPU$GPU [$K3_LEFT]; GPU$GPU goes on, the other runners finish their passes"; fi
fi

# ---------------------------------------------------------------- K3 route probe (the tool frozen at launch)
if reached k3-probe; then
  if [ "$K3_ALL" != 1 ] && [ "$B2_OK" = 1 ] && [ -n "$K3_LEFT" ] && ! echo "$K3_LEFT" | grep -q "busy=-"; then
    status k3-probe DEFERRED "K3 passes still run on other cards [$K3_LEFT]: the runner that seals the last pass runs the probe (k3_multicard.sh), or START_AT=k3-probe"
  elif [ "$K3_ALL" != 1 ] || [ "$B2_OK" != 1 ]; then status k3-probe SKIP "K3 cache not complete"
  elif [ "$(sha256sum "$PROBETOOL" | cut -c1-64)" != "$PROBESHA" ]; then
    status k3-probe FAIL "frozen probe tool $PROBETOOL changed since launch (want ${PROBESHA:0:16})"
  elif ! "$PY" "$PROBETOOL" --help 2>/dev/null | grep -q probe-distill; then
    status k3-probe SKIP "needs k3_teacher_cache probe-distill (reads corpus/k3 boundary_map.safetensors); frozen probe tool: $PROBEVER"
  else
    status k3-probe START "log $QL/k3-probe.log tool $PROBEVER sha256 ${PROBESHA:0:16}"
    k3_probe "$PROBETOOL" "$K3OUT" "$KC" "$QL/k3-probe.log"
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
