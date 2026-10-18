#!/usr/bin/env bash
# Toolcall lane, step 3 (refill rank 1 of $Q/D2_DECISIONS_2026-10-15.json, rule D2-X1-refill):
# raise the tool-call self-gen from 1.5M to 3.0M tokens and stop sampling by 2026-10-15T22:00Z,
# so GPU1 / GPU2 are free for the D1 bank v1 review.  Replaces the tail of
# run_toolcall_selfgen_2026-10-14.sh (whose shell was stopped at 02:19:45Z; its run kept going).
#
# The 1.5M run's token target is part of its locked run_config, so it is never resumed with a
# larger one.  Instead (toolcall_selfgen >= 1.2.0):
#   1. wait for the running 1.5M run (pid $1) to end on its own; if it ended short, resume it once
#      (same arguments, --stop-by).  Then finalize it: run 1 sealed as its own manifest.
#   2. continue-plan: the plan items after run 1's cutoff (same schemas, same scenarios and screens,
#      ids unchanged) -> plan-2026-10-15-cont.jsonl.
#   3. run 2 on that plan under base seed 20261015 (a new seed range), token target 3,000,000 minus
#      run 1's sealed tokens, --stop-by 21:00Z (graceful: no new item / attempt); a hard stop
#      (TERM, then KILL) at 21:40Z if it is still alive.
#   4. finalize run 2 (--prior-manifest run 1: ids disjoint, cross-run duplicate first user messages
#      screened out; --accept-short --cut-at-gap when it did not reach its target).
#   5. export BOTH runs into the ONE corpus post_d1_queue's C1 reads
#      ($TC/rescore/toolcall-q38-2026-10-14.safetensors); the sidecar names both manifests and their
#      row ranges (run 1 = A2 2(a), run 2 = the D2-X1 refill).  If run 2 cannot be sealed, run 1
#      alone is exported (D2 never waits on a refill source).
#   6. stop both servers through their receipts, write the toolcall RELEASE line (C1's gate and the
#      D1 review's card check), remove the transient Q4 only after a sealed export.
# Servers are stopped before finalize when run 2 ends at or after the deadline, and ALWAYS by the
# end of this script, also on failure (the GPUs must be free by 22:00Z); the Q4 is then kept.
#
#   setsid nohup bash run_toolcall_extend_2026-10-15.sh 73011 > $OC/distill/toolcall/logs/extend.log 2>&1 < /dev/null &
set -u
VERSION=toolcall_extend-1.0.0
RUN1_PID=${1:?usage: run_toolcall_extend_2026-10-15.sh <pid of the running 1.5M toolcall_selfgen.py run>}
# TOOLCALL_EXT_* are test hooks (test_run_toolcall_extend.py); none is set in production.
OC=/home/cereal/omni-campaign
PY=${TOOLCALL_EXT_PY:-/home/cereal/model-surgery-lab/venv/bin/python}
T=${TOOLCALL_EXT_T:-$OC/distill/tools}
Q=${TOOLCALL_EXT_Q:-$OC/distill/queues}
TC=${TOOLCALL_EXT_TC:-$OC/distill/toolcall}
SCRATCH=${TOOLCALL_EXT_SCRATCH:-/mnt/lexar/omni-campaign/scratch/toolcall-q38-2026-10-14}
CHECK=$TC/receipts/export_check_Q38-27BU-parent_toolcall.json
BANK=${TOOLCALL_EXT_BANK:-$OC/calibration/banks/omni_toolcall_bank_v1.json}
PLAN=$TC/plan-2026-10-14.jsonl
PLAN2=$TC/plan-2026-10-15-cont.jsonl
RUN1=$TC/run-q38-q4km-2026-10-14
RUN2=$TC/run-q38-q4km-2026-10-15-cont
DEST=$TC/rescore/toolcall-q38-2026-10-14.safetensors
S1=${TOOLCALL_EXT_S1:-http://127.0.0.1:18321}
S2=${TOOLCALL_EXT_S2:-http://127.0.0.1:18322}
TARGET1=${TOOLCALL_EXT_TARGET1:-1500000}
TOTAL=${TOOLCALL_EXT_TOTAL:-3000000}
BASE_SEED2=20261015
STOP_BY=${TOOLCALL_STOP_BY:-2026-10-15T21:00:00Z}
HARD_STOP=${TOOLCALL_HARD_STOP:-2026-10-15T21:40:00Z}
STATUS=$TC/logs/extend.status
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
mkdir -p "$TC/logs" "$TC/rescore"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
status() {  # one line to the status file (the L-TOOL probe's detail) + a lane heartbeat (never fatal)
  stamp "STATUS $*"; echo "$(date -u +%FT%TZ) $VERSION $*" >> "$STATUS"
  timeout 20 python3 "$T/lanes.py" --file "$Q/LANES.json" heartbeat --lane L-TOOL --agent claude --source "$VERSION" \
    --status "tool-call self-gen extend to 3.0M ($VERSION): $1" --detail "${*:0:400}" > /dev/null 2>&1 || stamp "lane heartbeat failed (ignored)"
}
epoch() { date -u -d "$1" +%s; }
now() { date -u +%s; }
jget() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {"d": d}))' "$1" "$2"; }

STOPPED=0
stop_servers() {  # stop_servers <outcome>: both llama-servers through their launch receipts + the RELEASE line, once
  [ "$STOPPED" = 1 ] && return 0
  local g pid
  for g in 1 2; do
    "$PY" "$T/selfgen_sampler.py" stop --launch-receipt "$TC/receipts/serve-toolcall-q38-q4km-gpu$g.json" --outcome "$1" \
      || stamp "stop GPU$g through its receipt returned non-zero"
    pid=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$TC/receipts/serve-toolcall-q38-q4km-gpu$g.json")
    if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "llama-server .*--port 1832$g"; then
      stamp "GPU$g llama-server $pid survived TERM: KILL"; kill -KILL "$pid"
    fi
  done
  echo "$(date -u +%FT%TZ) RELEASE GPU1,GPU2 owner=claude lane=toolcall ($VERSION: $1)" >> "$Q/p40.claims"
  STOPPED=1
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2
}
fail() {  # fail <rc> <why>: keep the Q4 and every run dir; the servers stay up for inspection / resume until
  # the hard stop, then are stopped (the deadline holds on failure too).  An agent taking over kills this
  # driver and owns the deadline.
  status "failed" "rc=$1 $2 (servers stopped at $HARD_STOP unless this driver is killed)"
  while [ "$STOPPED" = 0 ] && [ "$(now)" -lt "$(epoch "$HARD_STOP")" ]; do sleep 60; done
  stop_servers "toolcall extend FAILED rc=$1: $2 (Q4 kept for inspection; runs under $TC)"
  exit "$1"
}

stamp "driver start $VERSION; toolcall_selfgen $(sha256sum $T/toolcall_selfgen.py | cut -c1-16) ($("$PY" $T/toolcall_selfgen.py --version 2>&1))"
"$PY" "$T/toolcall_selfgen.py" --version 2>&1 | grep -q " 1\.2\." || { stamp "STOP: toolcall_selfgen is not 1.2.x"; exit 3; }
Q4=$("$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); assert r['parent_unchanged']; print(r['q4']['path'])" "$CHECK") || { stamp "STOP: export-check receipt unreadable"; exit 3; }
Q4SHA=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['q4']['sha256'])" "$CHECK")
stamp "Q4 $Q4 sha $Q4SHA; stop-by $STOP_BY, hard stop $HARD_STOP"
echo "$(date -u +%FT%TZ) CLAIM GPU1,GPU2 owner=claude lane=toolcall job=toolcall-extend-3.0M pid=$$ log=$TC/logs/extend.log (D2_DECISIONS refill rank 1: run 1 to 1.5M, then continuation run 2 to 3.0M total; sampling stops by $STOP_BY, servers stopped by $HARD_STOP at the latest; release=this driver)" >> "$Q/p40.claims"

# ---- 1. run 1 ends on its own, then is sealed
if kill -0 "$RUN1_PID" 2> /dev/null; then
  tr '\0' ' ' < "/proc/$RUN1_PID/cmdline" | grep -q "toolcall_selfgen.py run .*--out-dir $RUN1 " || { stamp "STOP: pid $RUN1_PID is not the run in $RUN1"; exit 3; }
  status "waiting for run 1" "pid $RUN1_PID $(tr -d '\n ' < $RUN1/progress.json | cut -c1-200)"
  while kill -0 "$RUN1_PID" 2> /dev/null; do sleep 60; done
fi
acc1=$(jget "$RUN1/progress.json" 'd["accepted_tokens"]')
stamp "run 1 process gone; progress accepted_tokens=$acc1"
if [ "$acc1" -lt "$TARGET1" ] && [ "$(now)" -lt "$(epoch "$STOP_BY")" ]; then
  stamp "run 1 short: resuming it once (same locked arguments, --stop-by $STOP_BY)"
  "$PY" "$T/toolcall_selfgen.py" run --plan "$PLAN" --server $S1 --server $S2 --out-dir "$RUN1" --token-target $TARGET1 \
    --gguf "$Q4" --gguf-sha256 "$Q4SHA" --stop-by "$STOP_BY"
  stamp "run 1 resume rc=$?"
  acc1=$(jget "$RUN1/progress.json" 'd["accepted_tokens"]')
fi
if [ ! -s "$RUN1/manifest.json" ]; then
  if [ "$acc1" -ge "$TARGET1" ]; then
    "$PY" "$T/toolcall_selfgen.py" finalize --out-dir "$RUN1" --bank "$BANK" || fail 11 "run 1 finalize failed"
  else
    "$PY" "$T/toolcall_selfgen.py" finalize --out-dir "$RUN1" --bank "$BANK" --accept-short --cut-at-gap || fail 11 "run 1 finalize (short) failed"
  fi
fi
sha256sum "$RUN1/manifest.json" "$RUN1/conversations.jsonl"
tok1=$(jget "$RUN1/manifest.json" 'd["tokens"]')
status "run 1 sealed" "manifest $(sha256sum $RUN1/manifest.json | cut -c1-16) conversations=$(jget $RUN1/manifest.json 'd["conversations"]') tokens=$tok1 screened_out=$(jget $RUN1/manifest.json 'd["screened_out"]') cutoff=$(jget $RUN1/manifest.json 'd["cutoff_plan_index"]')"

# ---- 2.-4. the continuation run
RUN2_SEALED=0
TARGET2=$((TOTAL - tok1))
if [ "$(now)" -ge "$(epoch "$STOP_BY")" ]; then
  stamp "past $STOP_BY: no continuation run"
elif ! curl -sf "$S1/health" > /dev/null && ! curl -sf "$S2/health" > /dev/null; then
  stamp "no sampler server answers /health: no continuation run"
else
  if [ ! -s "$PLAN2" ]; then
    "$PY" "$T/toolcall_selfgen.py" continue-plan --plan "$PLAN" --after-manifest "$RUN1/manifest.json" --out "$PLAN2" || fail 7 "continue-plan failed"
  fi
  sha256sum "$PLAN2" "${PLAN2%.jsonl}.manifest.json"
  status "run 2 sampling" "target $TARGET2 tokens (3.0M - run 1's $tok1), plan $(sha256sum $PLAN2 | cut -c1-16) items $(wc -l < $PLAN2), base seed $BASE_SEED2, stop by $STOP_BY"
  "$PY" "$T/toolcall_selfgen.py" run --plan "$PLAN2" --server $S1 --server $S2 --out-dir "$RUN2" --token-target $TARGET2 \
    --base-seed $BASE_SEED2 --gguf "$Q4" --gguf-sha256 "$Q4SHA" --stop-by "$STOP_BY" &
  RPID=$!
  KILLED=0; last_beat=0
  while kill -0 $RPID 2> /dev/null; do
    if [ "$(now)" -ge "$(epoch "$HARD_STOP")" ]; then
      stamp "hard stop $HARD_STOP: TERM run 2 (pid $RPID)"; kill -TERM $RPID; KILLED=1
      for _ in $(seq 30); do kill -0 $RPID 2> /dev/null || break; sleep 2; done
      kill -0 $RPID 2> /dev/null && { stamp "run 2 still alive: KILL"; kill -KILL $RPID; }
      break
    fi
    if [ $(( $(now) - last_beat )) -ge 600 ] && [ -s "$RUN2/progress.json" ]; then
      echo "$(date -u +%FT%TZ) $VERSION run2 $(tr -d '\n ' < $RUN2/progress.json | cut -c1-260)" >> "$STATUS"; last_beat=$(now)
    fi
    sleep 20
  done
  wait $RPID; rc2=$?
  [ "$KILLED" = 1 ] && rc2=137
  stamp "run 2 rc=$rc2"
  if [ "$KILLED" = 1 ] || [ "$(now)" -ge "$(epoch "$STOP_BY")" ]; then
    stop_servers "toolcall run 2 ended at/after $STOP_BY or the hard stop (rc=$rc2); finalize/export follow"
  fi
  case $rc2 in
    0) fin=() ;;
    2|3|4|137) fin=(--accept-short --cut-at-gap) ;;
    *) fin=(); stamp "run 2 rc=$rc2 is a functional failure: run 2 is NOT sealed; run 1 alone is exported" ;;
  esac
  if [ "$rc2" = 0 ] || [ ${#fin[@]} -gt 0 ]; then
    if [ -s "$RUN2/journal.jsonl" ] && "$PY" "$T/toolcall_selfgen.py" finalize --out-dir "$RUN2" --bank "$BANK" \
        --prior-manifest "$RUN1/manifest.json" ${fin[@]+"${fin[@]}"}; then
      sha256sum "$RUN2/manifest.json" "$RUN2/conversations.jsonl"
      [ "$(jget "$RUN2/manifest.json" 'd["conversations"]')" -gt 0 ] && RUN2_SEALED=1
      status "run 2 sealed" "manifest $(sha256sum $RUN2/manifest.json | cut -c1-16) conversations=$(jget $RUN2/manifest.json 'd["conversations"]') tokens=$(jget $RUN2/manifest.json 'd["tokens"]') short=$(jget $RUN2/manifest.json 'd["short_of_target"]') cut_at_gap=$(jget $RUN2/manifest.json 'd["cut_at_gap"]')"
    else
      stamp "run 2 not sealed (no journal or finalize failed): run 1 alone is exported"
    fi
  fi
fi

# ---- 5. one corpus for C1
if [ -e "$DEST" ]; then
  mv -- "$DEST" "$DEST.before-extend-$(date -u +%Y%m%dT%H%M%SZ)"; stamp "moved an existing $DEST aside"
fi
if [ "$RUN2_SEALED" = 1 ]; then
  if ! "$PY" "$T/toolcall_selfgen.py" export --out-dir "$RUN1" --out-dir "$RUN2" --dest "$DEST"; then
    stamp "export of run 1 + run 2 failed: exporting run 1 alone"; RUN2_SEALED=0
    "$PY" "$T/toolcall_selfgen.py" export --out-dir "$RUN1" --dest "$DEST" || fail 12 "export of run 1 failed"
  fi
else
  "$PY" "$T/toolcall_selfgen.py" export --out-dir "$RUN1" --dest "$DEST" || fail 12 "export of run 1 failed"
fi
CUDA_VISIBLE_DEVICES= "$PY" - "$DEST" <<'EOF' || fail 12 "the export does not load as C1 will read it"
import json, sys
from pathlib import Path
import qtext_teacher_cache as qt
dest = Path(sys.argv[1])
side = json.load(open(dest.with_suffix(".rows.json")))
loaded = qt.load_text_corpus(dest)
assert loaded["rows"] == side["rows"] and int(loaded["store_mask"].sum()) == side["supervised_tokens"], (loaded["rows"], side["rows"])
print("export check: rows", side["rows"], "tokens", side["tokens"], "supervised", side["supervised_tokens"],
      "runs", [(s["manifest_sha256"][:16], s["row_range"], s["tokens"]) for s in side["source_manifests"]])
EOF
sha256sum "$DEST" "${DEST%.safetensors}.rows.json"
outcome="toolcall self-gen extend complete: run 1 $(sha256sum $RUN1/manifest.json | cut -c1-16)$([ $RUN2_SEALED = 1 ] && echo " + run 2 $(sha256sum $RUN2/manifest.json | cut -c1-16)"); export $DEST sha $(sha256sum $DEST | cut -c1-16) tokens $(jget ${DEST%.safetensors}.rows.json 'd["tokens"]')"

# ---- 6. free the cards, drop the transient Q4
stop_servers "$outcome"
case "$Q4" in
  "$SCRATCH"/*.gguf) rm -f -- "$Q4" && stamp "removed transient $Q4" ;;
  *) stamp "refusing to remove $Q4: not in the declared scratch" ;;
esac
status "done" "$outcome"
df -BG / /mnt/lexar
stamp "done"
