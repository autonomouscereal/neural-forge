#!/usr/bin/env bash
# $OC/distill/queues/post_d1_queue.sh -- GPU0 / GPU4 work after the D1 seal, UNATTENDED.  Lane d1seal, Claude,
# 2026-10-14 (plan STAGE_D_RECONCILED_PLAN_2026-10-14 section 8 rows 8-9, section 3.4 C1 / C2 / C3; WAVE3 section 6
# option 2).  NOT ARMED by its author.
#
#   post_d1_queue.sh --dry-run   every job, its inputs / gates as they stand, its command; writes NOTHING; exit 0
#   post_d1_queue.sh             armed (detached: setsid nohup bash $Q/post_d1_queue.sh > $Q/logs/post_d1.log 2>&1 &)
#   post_d1_queue.sh <job>...    only those jobs, in that order (p2 c1 c2 j6 h3 p11)
# Starts when $Q/d1.sealed exists (written by d1_seal_queue.sh phase A, which then releases gpu0.lock / gpu4.lock).
# Single instance ($Q/post_d1.lock); ONE job at a time (host RAM: every job below gates on its own MemAvailable and a
# K3 pass needs 93 GiB, so two of ours never stack); a failed job stops only itself (status line, receipt, next job).
# A job whose input is not there yet is DEFERRED to the end of the queue and waited for there (bounded), so a late
# input never holds the cards from the others.  Every job: $Q/post_d1/<job>.json (rc, log, outputs + sha256) and
# status lines in $Q/post_d1.status.  Done jobs are skipped on a re-run ($Q/post_d1/<job>.json with rc 0).
#
# JOBS (card, lock, gate):
#   p2   GPU0  S0 P2 rate cell: run_p2_v133.sh 0 (driver 1.1.0, p2p11_card_guard: d1.sealed, no D1 lease newer, card
#              < 2,000 MiB, MemAvailable >= 60 GiB; it takes gpu0.lock itself) with WANT_T = the DEPLOYED
#              streamed_kd_trainer.py (latest); cells L4096-mb8, L4096-mb4, L32768-mb1 as prepared in tps/p2p11 ->
#              $Q/p2.verdict written by THE CONSUMER'S OWN WRITER, `d2_plan.py verdict --write` (STAGE_D_RULINGS G5-1:
#              >= 151 tok/s on a 3090 -> streamed + its micro / accumulation, else the split cell; no depth cut; nothing
#              written while no cell completed; an existing verdict is kept) -- under the standing GO, recorded for
#              review.  d1_seal_lib p2-verdict's reading goes to $R/p2.reading (diagnostic only).
#              1.2.0 (d1seal review): 1.1.x wrote d1_seal_lib's two-line reading (a text line + a JSON line) INTO $Q/p2.verdict;
#              d2_plan.read_verdict_file wants key=value lines, refused that file ("engine must be streamed or
#              split") and, because the file existed, never derived its own -- D2 would have waited forever.
#   p11  GPU4  OMNI1 256k needle rerun with the fixed scorer: run_p11_omni1_v110.sh 4 (p11_needle_engine 1.1.1 pinned;
#              takes gpu4.lock itself; MemAvailable >= 40 GiB)
#   c1   GPU0  tool-call self-gen bf16 rescore (C1): waits for toolcall_selfgen `export` (the sealed
#              rescore/toolcall-q38-2026-10-14.rows.json naming the corpus sha) and the toolcall RELEASE line; then
#              qtext_teacher_cache build-text (deployed version, Q38 bf16 top-64, head precision of
#              gpu0.head_decision.json) under gpu0.lock, MemAvailable >= 70 GiB, then `verify`
#   c2   GPU0  BFCL-native rows rescore (C2): waits for the d2_bfcl_native_copy output (card D2-M2, amendment A2 2b):
#              ONE distill/corpus/bfcl_native_v1/bfcl-native-<N>.safetensors (N = 500 after the BFCL-62 twin drop)
#              whose sha256 its rows.json names; same build-text + verify
#   j6   GPU0  VL / video answers: J6_GPU=0 $Q/j6_vl.sh 0 under gpu0.lock (its own gates: MemAvailable >= 70 GiB,
#              no live K3 process, j6.lock, never twice, j6.done ends it at once).  1.1.0: the queue checks j6's two
#              WAITING gates (no `k3_teacher_cache` process, MemAvailable >= 70 GiB) BEFORE it takes gpu0.lock and
#              defers j6 while either fails -- j6_vl.sh waits up to 48 h on them, and a K3 pass runs for days, so 1.0.x
#              would have held GPU0 idle for 48 h.  queue_gpu6 runs j6 after its H3 jobs anyway (j6.done / j6.lock
#              keep it to one run).
#   h3   --    H3 production caches run here ONLY with a proven 3090 path ($Q/h3_3090_path.proven naming PASS
#              receipts for int8-convrot loadability and parity).  None exists, and no 3090 H3 runner is wired in this
#              version, so the job records that H3 stays in the gpu6 queue (queue_gpu6 1.3.0 order) and ends.
#   p11  (last, 1.1.0) GPU4 does not wait on the GPU0 jobs' inputs: deferred jobs go behind it.
# 1.1.0 (2026-10-14, lane d1seal resume, on top of infra's 1.0.1): job order p2 c1 c2 j6 h3 p11 (the task's order);
# C2 reads the d2prep lane's real output dir (1.0.x waited for toolcall/bfcl-native/bfcl-native-506.safetensors, a
# path nothing writes); the j6 pre-gate above.
# 1.3.0 (2026-10-14, d1seal review): c1 / c2 check their WAITING gates (MemAvailable >= 70 GiB, / >= 60 GB,
# /mnt/lexar >= 20 GB) BEFORE they take gpu0.lock and are deferred (rc 75, lock never taken) while one fails, as j6.
# 1.2.x took gpu0.lock first, then waited up to 6 h for RAM and FAILED for good (no receipt, never retried by the
# queue) -- with K3 live MemAvailable sits ~4.5 GiB above the gate.  Under the lock the gates are a short re-check
# (30 min); a dip there drops the lock and defers too.  Deferred / gave_up lines read input_or_gate_not_ready.
# LANE HEARTBEAT (1.0.1, card X-T2, lane infra): every armed status line heartbeats its lane in $Q/LANES.json
# (lanes.py heartbeat --source $VERSION; a receipt=<path> in the line becomes the lane's last receipt + sha256).  c1 / c2
# lines go to L-TOOL (plan 7.1: the C1 / C2 rescores are L-TOOL's), except deferred / gave_up ones: a rescore that only
# waits for its input must not keep L-TOOL fresh, since finalize / export need an agent and a stale L-TOOL is what lets
# the other agent take it over (plan 7.4.5).  Everything else goes to L-D1 (GPU0 / GPU4 after the seal).  Never fatal
# (20 s cap), silent with no registry.  `post_d1_queue.sh --lane-of "<status line>"` prints the routing and exits.
set -u
VERSION=post_d1_queue-1.3.0
OC=${POSTD1_OC:-/home/cereal/omni-campaign}
Q=${POSTD1_Q:-$OC/distill/queues}
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
TC=$OC/distill/toolcall
C1_CORPUS=${POSTD1_C1_CORPUS:-$TC/rescore/toolcall-q38-2026-10-14.safetensors}
C2_DIR=${POSTD1_C2_DIR:-$OC/distill/corpus/bfcl_native_v1}
C2_CORPUS=${POSTD1_C2_CORPUS:-}   # empty: the single bfcl-native-*.safetensors in $C2_DIR (c2_corpus)
QTC=/mnt/lexar/omni-campaign/teacher_cache/qtext
STATUS=${POSTD1_STATUS:-$Q/post_d1.status}
R=$Q/post_d1
ALLOC=${POSTD1_ALLOC:-$OC/composite/receipts/allocator}
NVSMI=${POSTD1_NVSMI:-nvidia-smi}
POLL_S=${POSTD1_POLL_S:-300}
DEFER_WAIT_MAX_S=${POSTD1_DEFER_WAIT_MAX_S:-172800}
P2_OC=${POSTD1_P2_OC:-$OC}     # 1.2.0 test hook: the root d2_plan reads the P2 summaries under (P2_OC/distill/tps/p2-v133)
P2_RUN=${POSTD1_P2_RUN:-}      # 1.2.0 test hook: replaces `bash $T/run_p2_v133.sh 0` (never set in production)
LANES_PY=${LANES_PY:-$T/lanes.py}
LANES_FILE=${LANES_FILE:-$Q/LANES.json}
ALL_JOBS="p2 c1 c2 j6 h3 p11"
J6_RAM_GIB=${POSTD1_J6_RAM_GIB:-70}
K3_PGREP=${POSTD1_K3_PGREP:-k3_teacher_cache}
RESCORE_RAM_GIB=${POSTD1_RESCORE_RAM_GIB:-70}   # 1.3.0: the C1 / C2 build-text gate (test hook; 70 in production)
MODE=armed
if [ "${1:-}" = "--dry-run" ]; then MODE=dry; shift; fi
if [ "${1:-}" = "--lane-of" ]; then MODE=laneof; LANE_LINE=${2:-}; set --; fi
JOBS="${*:-$ALL_JOBS}"
for j in $JOBS; do case " $ALL_JOBS " in *" $j "*) ;; *) echo "unknown job $j (jobs: $ALL_JOBS)"; exit 64 ;; esac; done

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(stamp)] $*"; }
lane_of() {  # lane_of <status line>: the LANES.json lane it heartbeats (see LANE HEARTBEAT above)
  case "$1" in
    *state=deferred*|*state=gave_up*) echo L-D1 ;;
    "job=c1 "*|"job=c2 "*) echo L-TOOL ;;
    *) echo L-D1 ;;
  esac
}
lane_hb() {  # lane_hb <status line>: armed only, never fatal, at most 20 s, silent when there is no registry
  [ "$MODE" = armed ] && [ -f "$LANES_FILE" ] && [ -f "$LANES_PY" ] || return 0
  local lane st rcp extra=()
  lane=$(lane_of "$1")
  case "$lane:$1" in
    L-TOOL:*) st="C1/C2 rescore in post_d1_queue on GPU0 ($VERSION)" ;;
    *:state=end*) st="post-D1 queue ended ($VERSION)" ;;
    *) st="post-D1 queue running on GPU0/GPU4 ($VERSION)" ;;
  esac
  rcp=$(echo "$1" | grep -o 'receipt=[^ ]*' | tail -1 | cut -d= -f2-)
  [ -n "$rcp" ] && [ -f "$rcp" ] && extra=(--receipt "$rcp")
  timeout 20 python3 "$LANES_PY" --file "$LANES_FILE" heartbeat --lane "$lane" --agent claude --source "$VERSION" \
    --status "$st" --detail "${1:0:400}" ${extra[@]+"${extra[@]}"} > /dev/null 2>&1 || say "lane heartbeat $lane failed (ignored)"
  return 0
}
status() { say "STATUS $*"; [ "$MODE" = armed ] && { echo "$(stamp) post_d1 $*" >> "$STATUS"; lane_hb "$*"; }; return 0; }
[ "$MODE" = laneof ] && { lane_of "$LANE_LINE"; exit 0; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
mem_gib() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
used_mib() { $NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$1" '$1==g{print $2}'; }
lock_free() { [ -e "$1" ] || return 0; ( exec 3< "$1" && flock -n 3 ) 2>/dev/null; }
job_done() { [ -f "$R/$1.json" ] && "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("rc") == 0 else 1)' "$R/$1.json" 2>/dev/null; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID

receipt() {  # receipt <job> <rc> <log> [path...]: $R/<job>.json with every path's sha256
  local job=$1 rc=$2 log=$3; shift 3
  "$PY" - "$R/$job.json" "$job" "$rc" "$log" "$VERSION" "${STARTED:-}" "$@" <<'EOF'
import hashlib, json, os, sys, time
out, job, rc, log, ver, started, *paths = sys.argv[1:]
def sha(p):
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
doc = {"job": job, "rc": int(rc), "queue": ver, "started_utc": started or None,
       "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": log, "log_sha256": sha(log),
       "outputs": {p: sha(p) for p in paths}}
tmp = out + ".tmp"
open(tmp, "w").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
os.replace(tmp, out)
EOF
}

wait_mem() {  # wait_mem <GiB> <job> [max s, default 6 h]: 1 at the cap
  local held=0
  while [ "$(mem_gib)" -lt "$1" ]; do
    [ $held -ge "${3:-21600}" ] && return 1
    [ $((held % 1800)) -eq 0 ] && status "job=$2 state=waiting reason=MemAvailable_$(mem_gib)GiB_lt_$1"
    sleep 60; held=$((held + 60))
  done
}
take_card() {  # take_card <gpu> <job>: flock $Q/gpu<N>.lock on fd 7 (waits), then the card must be < 2,000 MiB
  local held=0 u
  exec 7>> "$Q/gpu$1.lock"
  until flock -n 7; do
    [ $((held % 1800)) -eq 0 ] && status "job=$2 state=waiting reason=gpu$1.lock_held"
    sleep 60; held=$((held + 60))
  done
  held=0; u=$(used_mib "$1")
  while [ "${u:-99999}" -ge 2000 ]; do
    [ $held -ge 21600 ] && { exec 7>&-; return 1; }
    [ $((held % 1800)) -eq 0 ] && status "job=$2 state=waiting reason=GPU$1_${u}MiB"
    sleep 60; held=$((held + 60)); u=$(used_mib "$1")
  done
}
drop_card() { exec 7>&-; }
self_locking() {  # self_locking <gpu> <job> <log> cmd...: a runner that takes gpu<N>.lock itself (flock -n) and refuses
  # when it is held; wait for the lock to look free, run, and retry (3x) a refusal that was only the lock race
  local g=$1 job=$2 log=$3 rc n held; shift 3
  for n in 1 2 3; do
    held=0
    until lock_free "$Q/gpu$g.lock"; do
      [ $((held % 1800)) -eq 0 ] && status "job=$job state=waiting reason=gpu$g.lock_held"
      sleep 60; held=$((held + 60))
    done
    "$@" >> "$log" 2>&1; rc=$?
    tail -5 "$log" | grep -q "another job holds .*gpu$g.lock" || return $rc
    status "job=$job event=lock_race attempt=$n"; sleep 120
  done
  return $rc
}

# ------------------------------------------------------------------------------------------------ jobs
job_p2() {
  local log=$R/p2.log want
  want=$(sha "$T/streamed_kd_trainer.py")
  status "job=p2 state=started trainer=$(grep -m1 '^TOOL_VERSION' $T/streamed_kd_trainer.py | cut -d'"' -f2) sha=${want:0:16}"
  if [ -n "$P2_RUN" ]; then WANT_T=$want self_locking 0 p2 "$log" bash -c "$P2_RUN"
  else WANT_T=$want self_locking 0 p2 "$log" bash "$T/run_p2_v133.sh" 0; fi
  local rc=$? vrc S=$P2_OC/distill/tps/p2-v133/p2-summary-v133-gpu0.json
  "$PY" "$T/d1_seal_lib.py" p2-verdict --summary "$S" --out "$R/p2.reading" >> "$log" 2>&1   # diagnostic reading only
  # $Q/p2.verdict in the format its consumer reads (d2_launch -> d2_plan verdict): d2_plan's own G5-1 writer
  env CUDA_VISIBLE_DEVICES= PYTHONPATH=$T nice -n 15 "$PY" "$T/d2_plan.py" --oc "$P2_OC" verdict --verdict "$Q/p2.verdict" --write >> "$log" 2>&1
  vrc=$?
  receipt p2 $rc "$log" "$S" "$Q/p2.verdict" "$R/p2.reading"
  status "job=p2 state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc verdict_rc=$vrc verdict=\"$(grep -E '^(engine|micro_batch|accumulation|best_tok_per_s)=' "$Q/p2.verdict" 2>/dev/null | tr '\n' ' ')\" receipt=$R/p2.json"
  return $rc
}
job_p11() {
  local log=$R/p11.log
  status "job=p11 state=started engine=$(sha $T/p11_needle_engine.py | cut -c1-16)"
  self_locking 4 p11 "$log" bash "$T/run_p11_omni1_v110.sh" 4
  local rc=$?
  receipt p11 $rc "$log" $(ls "$OC"/distill/tps/p11-v110/*.json 2>/dev/null)
  status "job=p11 state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc $(grep 'step=read' "$Q/p11.status" 2>/dev/null | tail -1 | cut -c1-300) receipt=$R/p11.json"
  return $rc
}
corpus_ready() {  # corpus_ready <corpus>: a sidecar (<stem>.rows.json, else rows.json beside it) names this file's sha256
  local side="${1%.safetensors}.rows.json" digest
  [ -f "$side" ] || side=$(dirname "$1")/rows.json
  [ -f "$1" ] && [ -f "$side" ] || return 1
  digest=$(sha "$1")
  [ -n "$digest" ] && grep -q "$digest" "$side"
}
rescore_waits() {  # prints why a C1 / C2 build-text would only wait now (empty: it can run); never takes a lock (1.3.0)
  local why=""
  [ "$(mem_gib)" -ge "$RESCORE_RAM_GIB" ] || why="$why MemAvailable_$(mem_gib)GiB_lt_$RESCORE_RAM_GIB"
  [ "$(free_gb /)" -ge 60 ] || why="$why root_$(free_gb /)GB_lt_60"
  [ "$(free_gb /mnt/lexar)" -ge 20 ] || why="$why lexar_$(free_gb /mnt/lexar)GB_lt_20"
  echo "${why# }"
}
rescore_defer() {  # rescore_defer <job> <why>: one status line per new reason, numbers ignored (routes to L-D1: a waiting
  # rescore must keep L-TOOL stale); returns 75
  local last=LAST_WHY_$1 key=${2//[0-9]/}
  [ "$key" != "${!last:-}" ] && status "job=$1 state=deferred reason=\"$2\" (gpu0.lock not taken)"
  printf -v "$last" '%s' "$key"
  return 75
}
rescore_gate() {  # rescore_gate <job>: 0 when the gates are clear, else the deferral (75) -- before any started line
  local why
  why=$(rescore_waits)
  [ -z "$why" ] || rescore_defer "$1" "$why"
}
rescore() {  # rescore <job> <corpus> <purpose>
  local job=$1 corpus=$2 purpose=$3 log=$R/$1.log head rc cache attempt why
  head=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["head_precision"])' "$Q/gpu0.head_decision.json" 2>/dev/null || echo fp32)
  rescore_gate "$job" || return 75   # (job_c1 / job_c2 already checked; kept so rescore alone never waits under the lock)
  take_card 0 "$job" || { status "job=$job state=failed reason=GPU0_never_free"; return 1; }
  for attempt in 1 2; do
    wait_mem "$RESCORE_RAM_GIB" "$job" 1800 || { drop_card; rescore_defer "$job" "MemAvailable_$(mem_gib)GiB_lt_${RESCORE_RAM_GIB}_under_lock"; return 75; }
    why=$(rescore_waits)
    [ -n "$why" ] && { drop_card; rescore_defer "$job" "${why}_under_lock"; return 75; }
    say "$job attempt $attempt" >> "$log"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$T OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false \
      "$PY" "$T/qtext_teacher_cache.py" build-text --model-dir "$M27" --device cuda:0 --dtype bfloat16 \
      --attn-implementation sdpa --linear-attn-kernel auto --digest-threads 8 --teacher q38-27bu --corpus "$corpus" \
      --out-root "$QTC" --micro-batch 8 --max-batch-tokens 32768 --batch-order length --rows-per-shard 256 \
      --head-precision "$head" --purpose "$purpose" --allocator-receipt "$ALLOC/post_d1-$job.json" >> "$log" 2>&1
    rc=$?
    say "$job attempt $attempt exit $rc" >> "$log"
    [ $rc -eq 0 ] || [ $rc -eq 2 ] && break
    sleep 60
  done
  cache=$(grep -o "SEALED [^ ]*manifest.json" "$log" | tail -1 | sed 's/^SEALED //; s#/manifest.json##')
  if [ $rc -eq 0 ] && [ -n "$cache" ]; then
    CUDA_VISIBLE_DEVICES= "$PY" "$T/qtext_teacher_cache.py" verify --cache-dir "$cache" >> "$log" 2>&1; rc=$?
  fi
  drop_card
  receipt "$job" $rc "$log" "$corpus" "${corpus%.safetensors}.rows.json" "$(dirname "$corpus")/rows.json" ${cache:+"$cache/manifest.json"}
  status "job=$job state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc cache=${cache:-none} manifest_sha=$(sha "${cache:-/nonexistent}/manifest.json" | cut -c1-16) receipt=$R/$job.json"
  return $rc
}
job_c1() {
  corpus_ready "$C1_CORPUS" && grep -q "RELEASE GPU1,GPU2 owner=claude lane=toolcall" "$Q/p40.claims" 2>/dev/null || return 75
  rescore_gate c1 || return 75
  status "job=c1 state=started corpus=$C1_CORPUS sha=$(sha "$C1_CORPUS" | cut -c1-16) qtext=$(grep -m1 '^TOOL_VERSION' $T/qtext_teacher_cache.py | cut -d'"' -f2)"
  rescore c1 "$C1_CORPUS" "post_d1 C1: Q38 bf16 rescore of the tool-call self-gen (toolcall_selfgen export, route tool_native; assistant positions via target_mask)"
}
c2_corpus() {  # the C2 corpus: $POSTD1_C2_CORPUS, else the ONE bfcl-native-*.safetensors in $C2_DIR (none / two: empty)
  [ -n "$C2_CORPUS" ] && { echo "$C2_CORPUS"; return 0; }
  local found=("$C2_DIR"/bfcl-native-*.safetensors)
  [ ${#found[@]} -eq 1 ] && [ -f "${found[0]}" ] && echo "${found[0]}"
}
j6_waits() {  # prints why j6_vl.sh would only sleep now (empty: it can run); never takes a lock
  local why=""
  pgrep -f "$K3_PGREP" > /dev/null 2>&1 && why="$why K3_process_live($K3_PGREP)"
  [ "$(mem_gib)" -ge "$J6_RAM_GIB" ] || why="$why MemAvailable_$(mem_gib)GiB_lt_$J6_RAM_GIB"
  echo "${why# }"
}
job_c2() {
  local c2
  c2=$(c2_corpus)
  [ -n "$c2" ] && corpus_ready "$c2" || return 75
  rescore_gate c2 || return 75
  status "job=c2 state=started corpus=$c2 sha=$(sha "$c2" | cut -c1-16)"
  rescore c2 "$c2" "post_d1 C2: Q38 bf16 rescore of Codex's BFCL-native train rows (card D2-M2 copy, BFCL-62 twins dropped; assistant positions via target_mask)"
}
job_j6() {
  local log=$R/j6.log rc
  [ -e "$Q/j6.done" ] && { receipt j6 0 "$log" "$Q/j6.done"; status "job=j6 state=skipped reason=j6.done_present"; return 0; }
  local why
  why=$(j6_waits)
  if [ -n "$why" ]; then
    [ "$why" != "${J6_LAST_WHY:-}" ] && status "job=j6 state=waiting reason=\"$why\" (gpu0.lock not taken)"
    J6_LAST_WHY=$why
    return 75
  fi
  take_card 0 j6 || { status "job=j6 state=failed reason=GPU0_never_free"; return 1; }
  status "job=j6 state=started runner=$(grep -m1 '^J6_VERSION' $Q/j6_vl.sh | cut -d'"' -f2)"
  J6_GPU=0 bash "$Q/j6_vl.sh" 0 >> "$log" 2>&1; rc=$?
  drop_card
  receipt j6 $rc "$log" "$Q/j6.done" "$Q/j6.status"
  status "job=j6 state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc $(tail -1 "$Q/j6.status" 2>/dev/null | cut -c1-300) receipt=$R/j6.json"
  return $rc
}
job_h3() {
  local log=$R/h3.log note
  if [ -s "$Q/h3_3090_path.proven" ]; then
    note="h3_3090_path.proven present ($(sha "$Q/h3_3090_path.proven" | cut -c1-16)) but no 3090 H3 runner is wired in $VERSION: H3 stays in the gpu6 queue; wire it in the next version"
  else
    note="no proven 3090 H3 path (no $Q/h3_3090_path.proven with int8-convrot loadability + parity PASS receipts): H3 stays in the gpu6 queue (queue_gpu6 1.3.0: after the K3 passes)"
  fi
  say "$note" >> "$log"
  receipt h3 0 "$log" "$Q/h3_3090_path.proven"
  status "job=h3 state=skipped reason=\"$note\" receipt=$R/h3.json"
  return 0
}

# ------------------------------------------------------------------------------------------------ dry run
if [ "$MODE" = dry ]; then
  say "$VERSION dry run; jobs: $JOBS"
  say "start gate: $Q/d1.sealed $([ -s "$Q/d1.sealed" ] && head -c 200 "$Q/d1.sealed" || echo absent); gpu0.lock $(lock_free "$Q/gpu0.lock" && echo free || echo held); gpu4.lock $(lock_free "$Q/gpu4.lock" && echo free || echo held); post_d1.lock $(lock_free "$Q/post_d1.lock" && echo free || echo held)"
  say "host: MemAvailable $(mem_gib) GiB; / $(free_gb /) GB; /mnt/lexar $(free_gb /mnt/lexar) GB; cards $($NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
  for j in $JOBS; do say "job $j: $(job_done $j && echo "done ($(sha "$R/$j.json" | cut -c1-16))" || echo pending)"; done
  say "p2: WANT_T=$(sha $T/streamed_kd_trainer.py | cut -c1-16) ($(grep -m1 '^TOOL_VERSION' $T/streamed_kd_trainer.py | cut -d'"' -f2)) bash $T/run_p2_v133.sh 0 (pins guard $(sha $T/p2p11_card_guard.sh | cut -c1-16), summary $(sha $T/p2_v133_summary.py | cut -c1-16)) -> d2_plan.py verdict --write -> $Q/p2.verdict ($([ -f "$Q/p2.verdict" ] && echo present || echo absent)); d1_seal_lib reading -> $R/p2.reading"
  say "p11: bash $T/run_p11_omni1_v110.sh 4 (engine $(sha $T/p11_needle_engine.py | cut -c1-16))"
  say "c1: corpus $C1_CORPUS $( corpus_ready "$C1_CORPUS" && echo ready || echo NOT-READY ); toolcall release line $(grep -c 'RELEASE GPU1,GPU2 owner=claude lane=toolcall' "$Q/p40.claims" 2>/dev/null)"
  c2=$(c2_corpus); say "c2: corpus ${c2:-<none: no single bfcl-native-*.safetensors in $C2_DIR>} $( [ -n "$c2" ] && corpus_ready "$c2" && echo ready || echo NOT-READY )"
  say "c1/c2 rescore pre-gate now (1.3.0, checked before gpu0.lock): $(w=$(rescore_waits); echo "${w:-clear}")"
  say "j6: J6_GPU=0 bash $Q/j6_vl.sh 0 (j6.done $([ -e "$Q/j6.done" ] && echo present || echo absent)); pre-gate now: $(w=$(j6_waits); echo "${w:-clear}")"
  say "h3: $Q/h3_3090_path.proven $([ -s "$Q/h3_3090_path.proven" ] && echo present || echo absent)"
  say "DRY RUN: nothing was written"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ armed
mkdir -p "$R" "$ALLOC" "$Q/logs"
exec 9>> "$Q/post_d1.lock"
flock -n 9 || { say "another post_d1_queue holds $Q/post_d1.lock"; exit 5; }
status "state=armed version=$VERSION jobs=\"$JOBS\" pid=$$"
held=0
until [ -s "$Q/d1.sealed" ]; do
  [ $((held % 3600)) -eq 0 ] && status "state=waiting reason=no_d1.sealed"
  sleep "$POLL_S"; held=$((held + POLL_S))
done
status "event=d1_sealed \"$(head -1 "$Q/d1.sealed" | cut -c1-240)\""

DEFERRED=""
run_job() {
  local j=$1 rc
  if job_done "$j"; then status "job=$j state=skipped reason=done_before ($R/$j.json)"; return 0; fi
  STARTED=$(stamp)
  "job_$j"; rc=$?
  if [ $rc -eq 75 ]; then return 75; fi
  [ $rc -eq 0 ] || status "job=$j state=failed rc=$rc (the queue continues)"
  return 0
}
for j in $JOBS; do
  run_job "$j"
  if [ $? -eq 75 ]; then DEFERRED="$DEFERRED $j"; status "job=$j state=deferred reason=input_or_gate_not_ready"; fi
done
held=0
while [ -n "${DEFERRED// /}" ]; do
  left=""
  for j in $DEFERRED; do
    run_job "$j"
    [ $? -eq 75 ] && left="$left $j"
  done
  DEFERRED=$left
  [ -z "${DEFERRED// /}" ] && break
  if [ $held -ge "$DEFER_WAIT_MAX_S" ]; then
    for j in $DEFERRED; do receipt "$j" 75 "$R/$j.log"; status "job=$j state=gave_up reason=input_or_gate_not_ready_after_${held}s"; done
    break
  fi
  [ $((held % 3600)) -eq 0 ] && status "state=waiting deferred=\"$DEFERRED\""
  sleep "$POLL_S"; held=$((held + POLL_S))
done
status "state=end"
exit 0
