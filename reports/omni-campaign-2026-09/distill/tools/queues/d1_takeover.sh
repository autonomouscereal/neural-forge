#!/bin/bash
# $OC/distill/queues/d1_takeover.sh -- hand D1 from Codex's allocator attempts to the hardened trainer.  Lane d1,
# Claude, 2026-10-14 (state doc STATE_2026-10-14_CODEX_HANDOFF.md step 4c, "if r4 dies").  NOT ARMED by its author:
# the Arm phase starts it after review.  It never stops, signals, execs into or writes anything of Codex's D1 (its
# containers, lease dirs, rung dirs, the Windows monitor): it only reads (docker ps / inspect, pgrep, /proc, the
# attempts' receipts / status / log / rung files).
#
# 1.1.0 (2026-10-14 16:10Z): r4 is gone (log stops 14:50:31Z after replaying L00-L03, container removed); Codex
# launched r5 (15:29Z, revoked ~15:50Z) and r6 (container up 15:55:30Z), each re-replaying L00-L10.  So this script
# follows EVERY Codex D1 attempt (containers codex-d1-native-01a10f78-*, run dirs
# run-OMNI1D1-native-01a10f78-20261014-c3*-recovery-r*), not r4 by name.
# 1.2.0 (2026-10-14 review): the launch goes through d1_hardened_launch 1.2.0 -- a system-manager unit (sudo -n
# systemd-run --uid, system.slice) instead of systemd --user, so systemd-oomd's pressure kill on user@1000.service
# cannot take D1; the entry 1.2.0 re-checks MemAvailable >= 100 GiB on EVERY start (systemd restarts too, not only
# this script's launch) and its ExecStopPost writes $RUN/ALERTS.jsonl when systemd will not restart D1.  Gate (2)
# looks for the unit in both managers.
#
#   d1_takeover.sh --check   read-only: every gate once, the Codex attempts, the canonical prefix (verified), the
#                            planned launch; exit 0 when a launch would happen now, 1 otherwise.  Writes nothing.
#   d1_takeover.sh           (detached: setsid nohup bash d1_takeover.sh > log 2>&1 < /dev/null &) poll every 60 s
#                            until Codex's D1 is over and every gate holds; record every attempt's end state; sync
#                            the attempts' committed rungs into the canonical dir in attempt order (verified; a
#                            refused attempt ends the sync, the canonical prefix stays self-consistent); launch the
#                            hardened D1 on GPU0 + GPU4 through d1_hardened_launch.sh; wait until it holds its cards;
#                            write the allocator claims; exit.  If an attempt finished D1 itself (completed, exit 0,
#                            delta in its receipt) nothing is launched.
#
# Gates:
#  (1) Codex's D1 is over: no running codex-d1-native-01a10f78-* container holds a trainer process
#      (d1_allocated_entry.py --run inside the container's cgroup), and a running container WITHOUT one has been
#      trainer-less for CODEX_GRACE_S seconds (default 900): Codex's D1 containers outlive their trainer (the v31
#      container kept `sleep 259200` after v35 died), so "container exited" alone could idle both 3090s for days.
#      CODEX_REQUIRE_CONTAINER_EXIT=1 restores the strict reading (no such container running at all).
#  (2) no other D1 interpreter on the host (d1_trunk_recovery / d1_allocated_entry / d1_hardened_entry) and the
#      claude-d1-hardened unit is not active (system manager, or a 1.1.0 launch in the user manager);
#  (3) GPU0 and GPU4: no compute process, < 2000 MiB used (the entry re-checks its cards by UUID before starting);
#  (4) $Q/gpu0.lock and $Q/gpu4.lock not held (read from /proc/locks; the trainer process takes them itself);
#  (5) MemAvailable >= 100 GiB (D1 holds ~85 GiB of host streams; a K3 pass ~72 GB: the later starter waits);
#  (6) free disk: root >= 60 GiB; Lexar >= 60 + 170 GiB (rungs 8.6 GB, delta 8.6 GB, two 82.3 GB boundaries during
#      a swap).
# The launched run: d1_trunk_recovery 1.3.1 under d1_hardened_entry 1.2.0 (system-manager unit as cereal in
# system.slice, MemoryMax 128G, MemAvailable >= 100 GiB on every start, Restart=on-failure,
# RestartPreventExitStatus=3, ALERTS.jsonl), r4's exact data / prereg / hyperparameters, the canonical prefix
# L00..L{N-1} imported (never retrained), the boundary cache under /mnt/lexar/omni-campaign/scratch/d1_boundary/
# (recomputed once through the prefix when absent, then after every D1_BOUNDARY_EVERY-th new layer, default 1).
set -u
VERSION=d1_takeover-1.2.0
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
CODEX_FILTER=codex-d1-native-01a10f78
D1_ROOT=/mnt/lexar/omni-campaign/distill/d1
ATTEMPT_PREFIX=run-OMNI1D1-native-01a10f78-20261014-c3     # c36-recovery-r2, c37-recovery-r3, -r4, -r5, -r6, ...
V35_TRAINER=$OC/distill/d1/run-OMNI1D1-native-01a10f78-20261014/trainer-v35
CODEX_GRACE_S=${CODEX_GRACE_S:-${R4_GRACE_S:-900}}
CODEX_REQUIRE_CONTAINER_EXIT=${CODEX_REQUIRE_CONTAINER_EXIT:-0}
# boundary after every K-th new layer.  1 = the state doc's rule.  Measured 2026-10-14: one boundary is 82.3 GB
# (P, C 41.06 GB each + B 0.17 GB); the exFAT Lexar wrote 108-139 MB/s and read 123-145 MB/s through the trainer's
# own functions.  The writes run on threads that overlap compute (P during the layer's ~17 min child pass, C during
# the next ~8 min parent forward), so K=1 should cost no wall time; a restart loads a boundary in ~10 min.
BOUNDARY_EVERY=${D1_BOUNDARY_EVERY:-1}
CAN=$D1_ROOT/OMNI1D1-rungs-canonical
TAG=claude-20261014
RUN=$D1_ROOT/run-OMNI1D1-hardened-$TAG
OUT=$RUN/trainer
DELTA=/mnt/lexar/omni-campaign/deltas/OC02-D-Q38-27BU-OMNI1D1-hardened-$TAG
BOUND=/mnt/lexar/omni-campaign/scratch/d1_boundary/OMNI1D1-hardened-$TAG
UNIT=claude-d1-hardened
GPU_UUIDS=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561,GPU-2ce6a95c-7b1f-644a-3808-d781732a0582   # host GPU0, GPU4
CHILD=/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1
PARENT=/mnt/ai_data/omni-campaign/s0-probe-sep04
SU=$OC/distill/codex-preparation-01a1326f-20261013/d1-source-successor-v2
MANIFEST_SHA=a488e75b46a667ebe91c23a08e5ce50a07a49be8208520f9af2d71bc060bbf88
VALIDATOR_SHA=0f2092d06f3232e62dd2657a7985b9b3c69da9c0cd07231fbf2924e98d479a86
STATUS=$Q/d1_takeover.status
REC=$Q/d1_takeover
GONE_FILE=$REC/.codex_trainer_gone_since   # first poll that saw a trainer-less Codex D1 container (gates: subshell)
export CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1
say() { echo "$(date -u +%FT%TZ) $*"; }
status() { say "$*"; [ "$CHECK" = 1 ] || echo "$(date -u +%FT%TZ) d1_takeover $*" >> "$STATUS"; }

# r4's exact recipe (Codex's resource policy / d1_allocated_contract required dict) on this host's paths + 1.3.0 flags
d1_args() {   # $1 = prefix layers, $2 = SHA256SUMS pin
  echo --child "$CHILD" --parent "$PARENT" \
    --preregistration "$SU/PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md" \
    --ids "$SU/d1_source_ids.safetensors" --seq-len 4096 --lengths-key lengths --held-rows 62 \
    --source-manifest "$SU/manifest.json" --source-manifest-sha256 $MANIFEST_SHA --source-validator-sha256 $VALIDATOR_SHA \
    --devices cuda:0,cuda:1 --dtype bfloat16 --train-dtype float32 --master-dtype float32 \
    --epochs 2 --tokens-per-step 32768 --micro-tokens 8192 --lr 2e-5 --warmup-steps 10 --min-lr-frac 0.1 \
    --clip 1.0 --seed 0 --loss-site residual --base-digest full --min-free-gib 60 --resume --no-pin --layers all \
    --out "$OUT" --delta-out "$DELTA" \
    --prefix-dir "$CAN" --prefix-sums-sha256 "$2" --prefix-layers "$1" --boundary-dir "$BOUND" \
    --boundary-every "$BOUNDARY_EVERY"
}

codex_running() { docker ps --filter "name=^$CODEX_FILTER" --filter status=running --format '{{.Names}}' | tr '\n' ' '; }
codex_trainer_pids() {   # trainer processes inside running Codex D1 containers (cgroup match, not command-line text)
  local n cid p
  for n in $(codex_running); do
    cid=$(docker inspect -f '{{.Id}}' "$n" 2>/dev/null) || continue
    for p in $(pgrep -f 'd1_allocated_entry\.py --run'); do
      grep -qs "$cid" /proc/$p/cgroup && printf '%s(%s) ' "$p" "${n##*-}"
    done
  done
}
other_d1_pids() {   # any D1 interpreter on the host (shell command lines that merely mention one are not)
  pgrep -f 'python[0-9.]* [^ ]*(d1_trunk_recovery|d1_allocated_entry|d1_hardened_entry)\.py' | tr '\n' ' '
}
attempt_trainers() {   # every attempt's trainer dir holding a receipt, in attempt order (v35 first; idempotent)
  echo "$V35_TRAINER"
  ls -d $D1_ROOT/${ATTEMPT_PREFIX}*/trainer-* 2>/dev/null | sort -V | while read -r d; do [ -f "$d/d1_receipt.json" ] && echo "$d"; done
}
lock_holders() {   # card locks someone holds now, from /proc/locks (takes nothing)
  local f ino held=""
  for f in gpu0 gpu4; do
    [ -e "$Q/$f.lock" ] || continue
    ino=$(stat -c %i "$Q/$f.lock")
    awk -v i="$ino" '{n = split($6, a, ":"); if (a[n] == i) f = 1} END {exit !f}' /proc/locks && held="$held $f.lock"
  done
  echo "$held"
}
codex_completed_d1() {   # prints the attempt and returns 0 when one finished D1 itself (completed, exit 0, delta)
  local d
  for d in $(attempt_trainers); do
    "$PY" - "$(dirname "$d")/runtime-status.json" "$d/d1_receipt.json" <<'EOF' && { echo "$d"; return 0; }
import json, sys
try:
    st = json.load(open(sys.argv[1])); rec = json.load(open(sys.argv[2]))
except Exception:
    sys.exit(1)
sys.exit(0 if st.get("completed") is True and st.get("exit_code") == 0 and rec.get("delta") else 1)
EOF
  done
  return 1
}

codex_over() {   # echo the reason Codex's D1 is not over (empty: over)
  local running pids now since
  running=$(codex_running)
  if [ "$CODEX_REQUIRE_CONTAINER_EXIT" = 1 ]; then
    [ -n "${running// /}" ] && echo " codex-d1-container-running(${running% })"
    return
  fi
  [ -n "${running// /}" ] || { [ "$CHECK" = 1 ] || rm -f "$GONE_FILE"; return; }
  pids=$(codex_trainer_pids)
  now=$(date +%s)
  if [ -n "${pids// /}" ]; then
    [ "$CHECK" = 1 ] || rm -f "$GONE_FILE"
    echo " codex-d1-trainer-alive(${pids% })"
    return
  fi
  if [ "$CHECK" = 1 ]; then echo " codex-d1-container-without-trainer(${running% }; grace not tracked by --check)"; return; fi
  [ -s "$GONE_FILE" ] || echo "$now" > "$GONE_FILE"
  since=$(cat "$GONE_FILE")
  [ $((now - since)) -ge "$CODEX_GRACE_S" ] || echo " codex-d1-trainer-gone-$((now - since))s-of-${CODEX_GRACE_S}s(${running% } still running)"
}
gates() {
  local why="" g used apps avail root lex st
  why="$why$(codex_over)"
  [ -n "$(other_d1_pids | tr -d ' ')" ] && why="$why d1-process-live($(other_d1_pids))"
  for st in "$(systemctl show -p ActiveState --value $UNIT.service 2>/dev/null)" \
            "$(systemctl --user show -p ActiveState --value $UNIT.service 2>/dev/null)"; do
    case "$st" in active|activating|deactivating|reloading) why="$why unit-$UNIT-$st" ;; esac
  done
  for g in 0 4; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
    { [ -n "$used" ] && [ "$used" -lt 2000 ]; } || why="$why gpu$g-${used:-?}MiB"
    apps=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
    [ "$apps" -eq 0 ] || why="$why gpu$g-$apps-compute-procs"
  done
  [ -n "$(lock_holders)" ] && why="$why locks-held:$(lock_holders)"
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] || why="$why memavail-${avail}GiB"
  root=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  lex=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
  [ "$root" -ge 60 ] || why="$why root-${root}GiB"
  [ "$lex" -ge 230 ] || why="$why lexar-${lex}GiB"
  echo "$why"
}

record_codex_exit() {   # -> $REC/codex_d1_exit_<utc>.json (reads only)
  mkdir -p "$REC"
  local f=$REC/codex_d1_exit_$(date -u +%Y%m%dT%H%M%SZ).json
  docker ps -a --filter "name=^$CODEX_FILTER" --format '{{.Names}}' | while read -r n; do
    docker inspect --format "{\"name\": \"$n\", \"state\": {{json .State}}}" "$n" 2>/dev/null
  done > "$REC/.codex_states.jsonl"
  "$PY" - "$f" "$REC/.codex_states.jsonl" "$(cat "$GONE_FILE" 2>/dev/null)" $(attempt_trainers) <<'EOF'
import hashlib, json, sys, time
from pathlib import Path
out, states, gone, *trainers = sys.argv[1:]
def read(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        return {"unreadable": repr(e)}
attempts = []
for t in trainers:
    t = Path(t); run = t.parent
    rec = read(t / "d1_receipt.json")
    layers = rec.get("layers", []) if isinstance(rec, dict) else []
    trained = [l["text_layer"] for l in layers if l.get("rung")]
    log = run / "runtime-log.txt"
    attempts.append({"trainer_dir": str(t), "runtime_status": read(run / "runtime-status.json"),
                     "receipt_sha256": hashlib.sha256((t / "d1_receipt.json").read_bytes()).hexdigest(),
                     "receipt_layers": len(layers), "layers_trained_here": trained,
                     "last_committed_rung": max(trained) if trained else None,
                     "rung_files": sorted(p.name for p in (t / "rungs").glob("L*")) if (t / "rungs").is_dir() else [],
                     "receipt_verdict": rec.get("verdict") if isinstance(rec, dict) else None,
                     "runtime_log_tail": log.read_text(errors="replace").splitlines()[-25:] if log.is_file() else None})
containers = [json.loads(l) for l in Path(states).read_text().splitlines() if l.strip()]
doc = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "codex_d1_containers": containers,
       "trainer_gone_since_epoch": int(gone) if gone else None, "attempts": attempts}
Path(out).write_text(json.dumps(doc, indent=1))
last = [a for a in attempts if a["last_committed_rung"] is not None]
print(f"Codex D1 end state recorded: {out}: {len(containers)} containers, {len(attempts)} attempts; "
      f"last committed rung {max(a['last_committed_rung'] for a in last) if last else None}")
EOF
  rm -f "$REC/.codex_states.jsonl"
}

sync_attempts() {   # attempt order; a refusal ends the sync and leaves the (self-consistent) canonical prefix as is
  local d out rc
  for d in $(attempt_trainers); do
    out=$("$PY" $T/d1_canonical_rungs.py sync --canonical "$CAN" --receipt "$d/d1_receipt.json" --rungs-dir "$d/rungs" \
          --label "$(basename "$(dirname "$d")")/$(basename "$d")" 2>&1)
    rc=$?
    echo "$out" | grep -v "cp: warning"
    if [ $rc -ne 0 ]; then
      status "canonical sync REFUSED (rc $rc) for $d; later attempts not synced; continuing from the canonical prefix as is"
      return 1
    fi
  done
  return 0
}

# ---------------------------------------------------------------- --check: read-only
if [ "$CHECK" = 1 ]; then
  say "$VERSION --check"
  say "Codex D1 containers running: $(codex_running); their trainers: $(codex_trainer_pids); D1 interpreters on host: $(other_d1_pids)"
  say "attempts: $(attempt_trainers | tr '\n' ' ')"
  say "trainer: $(grep -m1 '^TOOL_VERSION' $T/d1_trunk_recovery.py | cut -c1-30) sha $(sha256sum $T/d1_trunk_recovery.py | cut -c1-16); entry $(grep -m1 '^TOOL_VERSION' $T/d1_hardened_entry.py | cut -c1-30)"
  VJ=$("$PY" $T/d1_canonical_rungs.py verify --canonical "$CAN" --json 2>&1 | tail -1)
  N=$(echo "$VJ" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["layers"])' 2>/dev/null)
  S=$(echo "$VJ" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["sha256sums_sha256"])' 2>/dev/null)
  say "canonical: ${N:-?} layers, SHA256SUMS ${S:-?}"
  say "planned: d1_hardened_launch.sh --unit $UNIT --run-dir $RUN --manager system --memory-max 128G --min-mem-available-gib 100 --cuda-visible $GPU_UUIDS --lock $Q/gpu0.lock --lock $Q/gpu4.lock -- $(d1_args "${N:-?}" "${S:-?}")"
  sudo -n true 2>/dev/null && say "sudo -n: ok (system-manager launch possible)" || say "sudo -n: FAILED (the launch would exit 2)"
  DONE=$(codex_completed_d1) && say "a Codex attempt finished D1 itself ($DONE): nothing would be launched"
  WHY=$(gates)
  say "gates:${WHY:- all hold}"
  [ -z "$WHY" ] && exit 0
  exit 1
fi

# ---------------------------------------------------------------- armed: wait, record, sync, launch
mkdir -p "$REC" "$RUN"
exec 9>> "$Q/d1_takeover.lock"
flock -n 9 || { say "another d1_takeover holds $Q/d1_takeover.lock"; exit 5; }
status "armed ($VERSION; grace ${CODEX_GRACE_S}s, strict-container ${CODEX_REQUIRE_CONTAINER_EXIT}, boundary every $BOUNDARY_EVERY); waiting for Codex's D1 to be over"
LAST=""
while true; do
  WHY=$(gates)
  [ -z "$WHY" ] && break
  [ "$WHY" != "$LAST" ] && status "blocked:$WHY"
  LAST=$WHY
  sleep 60
done
record_codex_exit >> "$STATUS" 2>&1
if DONE=$(codex_completed_d1); then
  status "Codex attempt $DONE finished D1 itself (completed, exit 0, delta in its receipt): nothing launched; seal from its delta"
  exit 0
fi
sync_attempts >> "$STATUS" 2>&1
VJ=$("$PY" $T/d1_canonical_rungs.py verify --canonical "$CAN" --json 2>>"$STATUS" | tail -1)
N=$(echo "$VJ" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["layers"])') || { status "FAIL: canonical verify; nothing launched"; exit 2; }
S=$(echo "$VJ" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["sha256sums_sha256"])')
status "canonical L00..L$(printf %02d $((N - 1))) verified ($N layers), SHA256SUMS $S"
[ "$N" -ge 64 ] && { status "all 64 rungs are canonical: launch skipped; build the delta from the canonical rungs instead"; exit 0; }
WHY=$(gates)    # once more, right before the cards are loaded
[ -z "$WHY" ] || { status "re-gate failed:$WHY; nothing launched (re-arm)"; exit 3; }
mkdir -p "$(dirname "$BOUND")"
# shellcheck disable=SC2046
bash $T/d1_hardened_launch.sh --unit $UNIT --run-dir "$RUN" --manager system --memory-max 128G --restart-sec 300 \
  --cuda-visible "$GPU_UUIDS" --lock "$Q/gpu0.lock" --lock "$Q/gpu4.lock" --lock-wait-s 1800 --threads 16 \
  --min-mem-available-gib 100 --mem-wait-s 21600 \
  -- $(d1_args "$N" "$S") >> "$STATUS" 2>&1 || { status "FAIL: d1_hardened_launch.sh"; exit 2; }
ST=""
for i in $(seq 1 60); do
  ST=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("state"), d.get("pid"))' "$RUN/runtime-status.json" 2>/dev/null)
  case "$ST" in running*|succeeded*|refused*|failed*|lock_timeout*|gpu_busy*|memory_low*|killed*) break ;; esac
  sleep 10
done
status "hardened D1 launched: unit $UNIT (system manager), state ${ST:-unknown}, prefix $N layers, boundary $BOUND, status $RUN/runtime-status.json, log $RUN/runtime-log.txt, alerts $RUN/ALERTS.jsonl"
START=$(date -u +%FT%TZ)
for g in 0 4; do
  printf '{"lane":"D1 OMNI1D1 hardened (d1_takeover)","gpu":%s,"card":"%s","by":"claude lane d1","started_utc":"%s","unit":"%s","status":"%s","log":"%s"}\n' \
    $g "$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)" "$START" "$UNIT" "$RUN/runtime-status.json" "$RUN/runtime-log.txt" \
    > $OC/composite/receipts/allocator/d1-omni1d1-gpu$g-claude.json
done
exit 0
