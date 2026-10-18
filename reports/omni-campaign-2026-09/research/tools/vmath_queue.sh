#!/usr/bin/env bash
# vmath_queue.sh -- the verified-math pilot (research lane, Claude, 2026-10-15) as an unattended GUEST on the P40s.
#
# It starts only when cards are free and gives them back to anyone who needs them:
#   gate   vmath_pilot.py gate: GPU1/GPU2 (VMATH_GPUS) with no other lane's live p40.claims line, gpuN.lock free,
#          < 2,000 MiB and no compute app; MemAvailable >= 70 GiB (K3's identity-gate floor) + 9.5 GiB per card (the
#          measured RssAnon of one toolcall llama-server, 9,044,744 / 9,315,580 kB on 2026-10-15); no YIELD file; NOT
#          d1.sealed-without-d1.reviewed (D1's bank review may need P40 1+2+3: v1 first).  Polls every 5 min.
#   run    flock gpuN.lock, one CLAIM line, selfgen_sampler.py serve per card (the parent's own Q4_K_M, byte-identical
#          copy in declared scratch, sha 76c1e4c6...), eog-probe once, selfgen_sampler.py run (resumable journal) toward
#          VMATH_TOKEN_TARGET response tokens.
#   yield  every 60 s vmath_pilot.py yield-check (YIELD file / D1 review pending / another lane claimed a held card /
#          MemAvailable < 70 GiB twice in a row) and a session cap (VMATH_SESSION_MAX_S, 12 h): SIGTERM the sampler (its
#          journal keeps every finished sample), stop the servers, RELEASE line, drop the locks, back off 30 min, re-gate.
#   done   the sampler sealed manifest.json -> vmath_pilot.py verify (summary + intake batch) -> stop servers, RELEASE,
#          delete the Q4 copy from declared scratch, write $P/DONE.  A functional failure (sampler exit 1) stops the queue
#          with the cards released and nothing deleted.
# Writes only under $P (/mnt/lexar/omni-campaign/research/data/vmath-pilot-2026-10-15) and its scratch Q4 dir, plus the
# claim lines.  Never touches GPU0/4/5/6, the sealed corpora or another lane's files.
#
#   setsid nohup bash vmath_queue.sh > $P/logs/queue.log 2>&1 < /dev/null &
#   bash vmath_queue.sh --gate-once     print the gate line and exit (0 open / 10 wait)
#   touch $P/YIELD                      ask a running session to give the cards back (rm it to let the queue re-gate)
set -u
VERSION="vmath_queue 1.0.1"
OC=/home/cereal/omni-campaign
# VMATH_* overrides exist for test_vmath_queue.sh (stub servers, a fake queues dir); production uses the defaults.
Q=${VMATH_Q:-$OC/distill/queues}
T=${VMATH_T:-$OC/distill/tools}
R=/mnt/lexar/omni-campaign/research
RT=${VMATH_RT:-$R/tools}
PY=${VMATH_PY:-/home/cereal/model-surgery-lab/venv/bin/python}
LANE=research-vmath
P=${VMATH_DIR:-$R/data/vmath-pilot-2026-10-15}
PROMPTS=$P/prompts
RUN=$P/run
V=$P/verify
Q4DIR=${VMATH_Q4DIR:-/mnt/lexar/omni-campaign/scratch/research-vmath-2026-10-15}
GPUS_WANT=${VMATH_GPUS:-1,2}
TARGET=${VMATH_TOKEN_TARGET:-1200000}
SESSION_MAX_S=${VMATH_SESSION_MAX_S:-43200}
BACKOFF_S=${VMATH_BACKOFF_S:-1800}
POLL_S=${VMATH_POLL_S:-300}
CHECK_S=${VMATH_CHECK_S:-60}
GATE_EXTRA=${VMATH_GATE_EXTRA:-}
YIELD=$P/YIELD
STATUS=$P/queue.status
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$T
mkdir -p "$P/logs" "$P/receipts"

stamp() { date -u +%FT%TZ; }
say() { echo "[$(stamp)] $*"; }
status() { echo "$(stamp) $VERSION pid=$$ $*" >> "$STATUS"; say "$*"; }

gate_line() {
  "$PY" "$RT/vmath_pilot.py" gate --claims "$Q/p40.claims" --queues-dir "$Q" --gpus "$GPUS_WANT" \
    --yield-file "$YIELD" --lane "$LANE" --seal-lib "$T/d1_seal_lib.py" $GATE_EXTRA
}

if [ "${1:-}" = "--gate-once" ]; then gate_line; exit $?; fi

exec 9>> "$P/queue.lock"
flock -n 9 || { say "another vmath_queue holds $P/queue.lock"; exit 5; }

[ -s "$PROMPTS/manifest.json" ] || { status "state=refused reason=no-prompt-set ($PROMPTS/manifest.json)"; exit 3; }

HELD=""
SERVERS=()
CLAIMED=0
release_all() {
  local g
  for g in ${HELD//,/ }; do
    if [ -s "$P/receipts/serve-vmath-q38-gpu$g.json" ]; then
      "$PY" "$T/selfgen_sampler.py" stop --launch-receipt "$P/receipts/serve-vmath-q38-gpu$g.json" --outcome "${1:-released}" \
        >> "$P/logs/serve-stop.log" 2>&1 || say "stop gpu$g returned non-zero (see serve-stop.log)"
      mv -f "$P/receipts/serve-vmath-q38-gpu$g.json" "$P/receipts/serve-vmath-q38-gpu$g.$(date -u +%Y%m%dT%H%M%SZ).json"
    fi
  done
  if [ "$CLAIMED" = 1 ]; then
    echo "$(stamp) RELEASE ${HELD_CLAIM} owner=claude lane=$LANE job=vmath-pilot released (${1:-released})" >> "$Q/p40.claims"
    CLAIMED=0
  fi
  exec 3>&- 4>&- 5>&-
  HELD=""
}
SAMPLER_PID=""
# 1.0.1: a SIGTERMed sampler only exits once its workers' HTTP requests return (request timeout 7,200 s; one thinking
# sample can run ~20 min), so the cards are given back FIRST (release_all stops the servers, which fails those requests
# at once) and the sampler is reaped afterwards with a bounded wait.  Its journal is fsynced per sample: nothing is lost.
reap_sampler() {
  local pid=$SAMPLER_PID n=0
  [ -n "$pid" ] || return 0
  while kill -0 "$pid" 2>/dev/null && [ $n -lt 180 ]; do sleep 2; n=$((n + 2)); done
  if kill -0 "$pid" 2>/dev/null; then say "sampler $pid alive 180 s after its servers stopped: SIGKILL"; kill -KILL "$pid" 2>/dev/null; fi
  wait "$pid" 2>/dev/null
  SAMPLER_PID=""
}
trap '[ -n "$SAMPLER_PID" ] && kill -TERM "$SAMPLER_PID" 2>/dev/null; release_all "queue exit"; reap_sampler; exit 0' TERM INT
trap 'release_all "queue exit"' EXIT

wait_q4() {
  while [ ! -s "$Q4DIR/q4_copy_receipt.json" ]; do
    if ! pgrep -f vmath_q4_copy.sh > /dev/null; then
      status "state=refused reason=no-q4-copy (vmath_q4_copy.sh is not running and $Q4DIR/q4_copy_receipt.json is absent)"
      exit 4
    fi
    sleep 120
  done
  Q4=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['q4']['path'])" "$Q4DIR/q4_copy_receipt.json")
  Q4SHA=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['q4']['sha256'])" "$Q4DIR/q4_copy_receipt.json")
}

take_cards() {  # sets HELD / HELD_CLAIM and the claim, or returns 1
  local line gpus g fd=3
  line=$(gate_line); local rc=$?
  [ $rc -eq 0 ] || { LAST_GATE="$line"; return 1; }
  gpus=$(echo "$line" | sed -n 's/^GATE OPEN //p')
  [ -n "$gpus" ] || return 1
  HELD=""
  for g in ${gpus//,/ }; do
    eval "exec $fd>> \"$Q/gpu$g.lock\""
    if flock -n $fd; then HELD="${HELD:+$HELD,}$g"; fd=$((fd + 1)); else eval "exec $fd>&-"; fi
  done
  [ -n "$HELD" ] || return 1
  HELD_CLAIM=$(echo "$HELD" | sed 's/\([0-9]\)/GPU\1/g')
  CLAIM_LINE_NO=$(wc -l < "$Q/p40.claims")
  echo "$(stamp) CLAIM $HELD_CLAIM owner=claude lane=$LANE job=vmath-pilot pid=$$ log=$P/logs/queue.log (research guest; yields to v1 lanes; ask with: touch $YIELD)" >> "$Q/p40.claims"
  CLAIMED=1
  return 0
}

serve_cards() {
  local g port urls=()
  : > "$P/servers.txt"
  for g in ${HELD//,/ }; do
    port=$((18330 + g))
    "$PY" "$T/selfgen_sampler.py" serve --gguf "$Q4" --gpu-index "$g" --allowed-gpus 1,2,3 --port "$port" \
      --name "vmath-q38-gpu$g" --slots 4 --ctx-per-slot 8192 --log-dir "$P/logs" --receipt-dir "$P/receipts" \
      --purpose "research lane verified-math pilot (guest): parent Q4_K_M llama-server, claim in queues/p40.claims" \
      >> "$P/logs/serve.log" 2>&1 || { status "state=serve-failed gpu=$g (see $P/logs/serve.log)"; return 1; }
    echo "http://127.0.0.1:$port" >> "$P/servers.txt"
  done
  if [ ! -s "$P/receipts/eog_probe.json" ]; then
    local args=()
    while read -r u; do args+=(--server "$u"); done < "$P/servers.txt"
    "$PY" "$T/selfgen_sampler.py" eog-probe "${args[@]}" --prompts "$PROMPTS/prompts.jsonl" --prompts-manifest "$PROMPTS/manifest.json" \
      --out "$P/receipts/eog_probe.json" >> "$P/logs/serve.log" 2>&1 || { status "state=eog-probe-failed"; return 1; }
  fi
  return 0
}

session() {  # one claimed session; returns the sampler's exit code, or 20 when it yielded
  local args=() u started rc low=0 why
  while read -r u; do args+=(--server "$u"); done < "$P/servers.txt"
  nice -n 15 "$PY" "$T/selfgen_sampler.py" run "${args[@]}" --prompts "$PROMPTS/prompts.jsonl" \
    --prompts-manifest "$PROMPTS/manifest.json" --out-dir "$RUN" --gguf "$Q4" --gguf-sha256 "$Q4SHA" \
    --token-target "$TARGET" --base-seed 20261015 --slots-per-server 4 --request-timeout 7200 --rows-per-shard 64 \
    >> "$P/logs/run.log" 2>&1 &
  local pid=$!
  SAMPLER_PID=$pid
  started=$(date +%s)
  status "state=running gpus=$HELD sampler_pid=$pid target=$TARGET"
  while kill -0 $pid 2>/dev/null; do
    sleep "$CHECK_S"
    why=$("$PY" "$RT/vmath_pilot.py" yield-check --claims "$Q/p40.claims" --queues-dir "$Q" --held "$HELD" \
          --since-line "$((CLAIM_LINE_NO + 1))" --yield-file "$YIELD" --lane "$LANE" --mem-floor-gib 70 $GATE_EXTRA)
    rc=$?
    if [ $rc -eq 20 ] && echo "$why" | grep -q MemAvailable; then low=$((low + 1)); [ $low -lt 2 ] && rc=0; else [ $rc -eq 0 ] && low=0; fi
    if [ $rc -eq 20 ] || [ $(( $(date +%s) - started )) -ge "$SESSION_MAX_S" ]; then
      [ $rc -eq 20 ] || why="session cap ${SESSION_MAX_S} s"
      status "state=yielding reason=\"$why\""
      kill -TERM $pid 2>/dev/null
      return 20                     # the caller stops the servers, then reap_sampler
    fi
  done
  wait $pid
  rc=$?
  SAMPLER_PID=""
  return $rc
}

status "state=start target=$TARGET gpus_wanted=$GPUS_WANT prompts=$(sha256sum "$PROMPTS/prompts.jsonl" | cut -c1-16)"
while :; do
  [ -e "$P/DONE" ] && { status "state=done (DONE present)"; exit 0; }
  wait_q4
  if ! take_cards; then
    status "state=waiting $(echo "${LAST_GATE:-}" | cut -c1-400)"
    sleep "$POLL_S"; continue
  fi
  if ! serve_cards; then release_all "serve failed"; sleep "$BACKOFF_S"; continue; fi
  session; rc=$?
  if [ $rc -eq 0 ] && [ -s "$RUN/manifest.json" ]; then
    "$PY" "$RT/vmath_pilot.py" verify --run-dir "$RUN" --prompts-dir "$PROMPTS" --out-dir "$V" >> "$P/logs/verify.log" 2>&1
    vrc=$?
    release_all "vmath pilot complete: run manifest $(sha256sum "$RUN/manifest.json" | cut -c1-16); verify rc=$vrc"
    if [ $vrc -eq 0 ]; then
      case "$Q4" in
        "$Q4DIR"/*.gguf) rm -f -- "$Q4" && status "removed transient $Q4" ;;
        *) status "refusing to remove $Q4: not in $Q4DIR" ;;
      esac
      echo "$(stamp) $(sha256sum "$V/summary.json")" > "$P/DONE"
      status "state=done summary=$V/summary.json"
      exit 0
    fi
    status "state=verify-failed rc=$vrc (see $P/logs/verify.log); cards released, Q4 kept"
    exit 6
  fi
  release_all "session ended rc=$rc"
  reap_sampler
  if [ $rc -eq 1 ]; then status "state=stopped reason=sampler-functional-failure (see $P/logs/run.log); cards released"; exit 7; fi
  status "state=backoff rc=$rc seconds=$BACKOFF_S"
  sleep "$BACKOFF_S"
done
