#!/usr/bin/env bash
# $OC/distill/queues/post_d1_queue.sh -- GPU0 / GPU4 work after the D1 seal, UNATTENDED.  Lane d1seal, Claude,
# 2026-10-14 (plan STAGE_D_RECONCILED_PLAN_2026-10-14 section 8 rows 8-9, section 3.4 C1 / C2 / C3; WAVE3 section 6
# option 2).  NOT ARMED by its author.
#
#   post_d1_queue.sh --dry-run   every job, its inputs / gates as they stand, its command; writes NOTHING; exit 0
#   post_d1_queue.sh             armed (detached: setsid nohup bash $Q/post_d1_queue.sh > $Q/logs/post_d1.log 2>&1 &)
#   post_d1_queue.sh <job>...    only those jobs, in that order (p2 p11 c1 j6 c2 h3)
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
#              d1_seal_lib p2-verdict -> $Q/p2.verdict (STAGE_D_RULINGS G5-1: >= 151 tok/s streamed, else the measured
#              split cell; < 60 token cut allowed; NO depth cut) -- applied under the standing GO, recorded for review
#   p11  GPU4  OMNI1 256k needle rerun with the fixed scorer: run_p11_omni1_v110.sh 4 (p11_needle_engine 1.1.1 pinned;
#              takes gpu4.lock itself; MemAvailable >= 40 GiB)
#   c1   GPU0  tool-call self-gen bf16 rescore (C1): waits for toolcall_selfgen `export` (the sealed
#              rescore/toolcall-q38-2026-10-14.rows.json naming the corpus sha) and the toolcall RELEASE line; then
#              qtext_teacher_cache build-text (deployed version, Q38 bf16 top-64, head precision of
#              gpu0.head_decision.json) under gpu0.lock, MemAvailable >= 70 GiB, then `verify`
#   j6   GPU0  VL / video answers: J6_GPU=0 $Q/j6_vl.sh 0 under gpu0.lock (its own gates: MemAvailable >= 70 GiB,
#              j6.lock, never twice, j6.done ends it at once)
#   c2   GPU0  BFCL-native 506 rows rescore (C2): waits for toolcall/bfcl-native/bfcl-native-506.safetensors and a
#              sidecar naming its sha256 (bfcl-native-506.rows.json or rows.json; card D2-M2); same build-text + verify
#   h3   --    H3 production caches run here ONLY with a proven 3090 path ($Q/h3_3090_path.proven naming PASS
#              receipts for int8-convrot loadability and parity).  None exists, and no 3090 H3 runner is wired in this
#              version, so the job records that H3 stays in the gpu6 queue (queue_gpu6 1.3.0 order) and ends.
set -u
VERSION=post_d1_queue-1.0.0
OC=${POSTD1_OC:-/home/cereal/omni-campaign}
Q=${POSTD1_Q:-$OC/distill/queues}
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
TC=$OC/distill/toolcall
C1_CORPUS=${POSTD1_C1_CORPUS:-$TC/rescore/toolcall-q38-2026-10-14.safetensors}
C2_CORPUS=${POSTD1_C2_CORPUS:-$TC/bfcl-native/bfcl-native-506.safetensors}
QTC=/mnt/lexar/omni-campaign/teacher_cache/qtext
STATUS=${POSTD1_STATUS:-$Q/post_d1.status}
R=$Q/post_d1
ALLOC=${POSTD1_ALLOC:-$OC/composite/receipts/allocator}
NVSMI=${POSTD1_NVSMI:-nvidia-smi}
POLL_S=${POSTD1_POLL_S:-300}
DEFER_WAIT_MAX_S=${POSTD1_DEFER_WAIT_MAX_S:-172800}
ALL_JOBS="p2 p11 c1 j6 c2 h3"
MODE=armed
if [ "${1:-}" = "--dry-run" ]; then MODE=dry; shift; fi
JOBS="${*:-$ALL_JOBS}"
for j in $JOBS; do case " $ALL_JOBS " in *" $j "*) ;; *) echo "unknown job $j (jobs: $ALL_JOBS)"; exit 64 ;; esac; done

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(stamp)] $*"; }
status() { say "STATUS $*"; [ "$MODE" = armed ] && echo "$(stamp) post_d1 $*" >> "$STATUS"; return 0; }
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

wait_mem() {  # wait_mem <GiB> <job>  (6 h, then the job fails)
  local held=0
  while [ "$(mem_gib)" -lt "$1" ]; do
    [ $held -ge 21600 ] && return 1
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
  WANT_T=$want self_locking 0 p2 "$log" bash "$T/run_p2_v133.sh" 0
  local rc=$? S=$OC/distill/tps/p2-v133/p2-summary-v133-gpu0.json
  "$PY" "$T/d1_seal_lib.py" p2-verdict --summary "$S" --out "$Q/p2.verdict" >> "$log" 2>&1
  receipt p2 $rc "$log" "$S" "$Q/p2.verdict"
  status "job=p2 state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc verdict=\"$(head -1 "$Q/p2.verdict" 2>/dev/null)\" receipt=$R/p2.json"
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
rescore() {  # rescore <job> <corpus> <purpose>
  local job=$1 corpus=$2 purpose=$3 log=$R/$1.log head rc cache attempt
  head=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["head_precision"])' "$Q/gpu0.head_decision.json" 2>/dev/null || echo fp32)
  take_card 0 "$job" || { status "job=$job state=failed reason=GPU0_never_free"; return 1; }
  for attempt in 1 2; do
    wait_mem 70 "$job" || { drop_card; status "job=$job state=failed reason=MemAvailable"; return 1; }
    [ "$(free_gb /)" -ge 60 ] && [ "$(free_gb /mnt/lexar)" -ge 20 ] || { drop_card; status "job=$job state=failed reason=disk"; return 1; }
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
  status "job=c1 state=started corpus=$C1_CORPUS sha=$(sha "$C1_CORPUS" | cut -c1-16) qtext=$(grep -m1 '^TOOL_VERSION' $T/qtext_teacher_cache.py | cut -d'"' -f2)"
  rescore c1 "$C1_CORPUS" "post_d1 C1: Q38 bf16 rescore of the tool-call self-gen (toolcall_selfgen export, route tool_native; assistant positions via target_mask)"
}
job_c2() {
  corpus_ready "$C2_CORPUS" || return 75
  status "job=c2 state=started corpus=$C2_CORPUS sha=$(sha "$C2_CORPUS" | cut -c1-16)"
  rescore c2 "$C2_CORPUS" "post_d1 C2: Q38 bf16 rescore of Codex's 506 BFCL-native train rows (card D2-M2 copy; assistant positions via target_mask)"
}
job_j6() {
  local log=$R/j6.log rc
  [ -e "$Q/j6.done" ] && { receipt j6 0 "$log" "$Q/j6.done"; status "job=j6 state=skipped reason=j6.done_present"; return 0; }
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
  status "job=h3 state=skipped reason=\"$note\""
  return 0
}

# ------------------------------------------------------------------------------------------------ dry run
if [ "$MODE" = dry ]; then
  say "$VERSION dry run; jobs: $JOBS"
  say "start gate: $Q/d1.sealed $([ -s "$Q/d1.sealed" ] && head -c 200 "$Q/d1.sealed" || echo absent); gpu0.lock $(lock_free "$Q/gpu0.lock" && echo free || echo held); gpu4.lock $(lock_free "$Q/gpu4.lock" && echo free || echo held); post_d1.lock $(lock_free "$Q/post_d1.lock" && echo free || echo held)"
  say "host: MemAvailable $(mem_gib) GiB; / $(free_gb /) GB; /mnt/lexar $(free_gb /mnt/lexar) GB; cards $($NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
  for j in $JOBS; do say "job $j: $(job_done $j && echo "done ($(sha "$R/$j.json" | cut -c1-16))" || echo pending)"; done
  say "p2: WANT_T=$(sha $T/streamed_kd_trainer.py | cut -c1-16) ($(grep -m1 '^TOOL_VERSION' $T/streamed_kd_trainer.py | cut -d'"' -f2)) bash $T/run_p2_v133.sh 0 (pins guard $(sha $T/p2p11_card_guard.sh | cut -c1-16), summary $(sha $T/p2_v133_summary.py | cut -c1-16)) -> d1_seal_lib p2-verdict -> $Q/p2.verdict"
  say "p11: bash $T/run_p11_omni1_v110.sh 4 (engine $(sha $T/p11_needle_engine.py | cut -c1-16))"
  say "c1: corpus $C1_CORPUS $( corpus_ready "$C1_CORPUS" && echo ready || echo NOT-READY ); toolcall release line $(grep -c 'RELEASE GPU1,GPU2 owner=claude lane=toolcall' "$Q/p40.claims" 2>/dev/null)"
  say "c2: corpus $C2_CORPUS $( corpus_ready "$C2_CORPUS" && echo ready || echo NOT-READY )"
  say "j6: J6_GPU=0 bash $Q/j6_vl.sh 0 (j6.done $([ -e "$Q/j6.done" ] && echo present || echo absent))"
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
  if [ $? -eq 75 ]; then DEFERRED="$DEFERRED $j"; status "job=$j state=deferred reason=input_not_ready"; fi
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
    for j in $DEFERRED; do receipt "$j" 75 "$R/$j.log"; status "job=$j state=gave_up reason=input_not_ready_after_${held}s"; done
    break
  fi
  [ $((held % 3600)) -eq 0 ] && status "state=waiting deferred=\"$DEFERRED\""
  sleep "$POLL_S"; held=$((held + POLL_S))
done
status "state=end"
exit 0
