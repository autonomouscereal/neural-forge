#!/usr/bin/env bash
# run_d2_arm_2026-10-15.sh -- arm the Stage D2 queue (queues/d2_launch.sh run) unattended, at the first safe moment.
# Lane d2ready (L-TRAIN), Claude lead, 2026-10-15 (STATE_2026-10-15_D2_READINESS.md, gate "D2 queue armed").
#
# WHY.  d2_launch.sh (1.3.x) is installed but nothing in the D1 -> D2 chain starts it: d1_seal_queue and post_d1_queue
#   end without arming it, so D2 would sit unstarted after every other gate opened until an agent woke up.
# WHEN (both):
#   1. $Q/post_d1/p2.json exists (post_d1's P2 job finished, any rc).  Arming earlier is unsafe: in run mode the D2
#      gates call `d2_plan verdict --write`, which writes $Q/p2.verdict from whatever P2 cells have completed so far,
#      and the first writer wins (cmd_verdict returns an existing file), so a half-finished P2 could fix the engine.
#   2. $Q/post_d1/k3decide.json exists (the K3-first / D2-first decision was taken, or gave up after 48 h), OR no
#      post_d1_queue.sh process lives (nothing will decide; D2-first by default).  Arming later than this costs
#      nothing: D2 cannot start before C1 and C2 seal, and post_d1 runs them after k3decide.
# THEN:  `d2_launch.sh preflight` (D2_PREFLIGHT_DATA_CHECK=0, read-only).  rc 2 (a pin or a file refused) -> not armed,
#   status REFUSED, re-checked every REFUSED_RETRY_S (a fix may land).  Otherwise
#     cd /tmp; setsid nohup bash $Q/d2_launch.sh run >> $LEX/distill/d2/logs/queue.log 2>&1 < /dev/null &
#   with this watcher's lock fd closed, and SETTLE_S later the queue must hold $Q/d2.lock (its single-instance lock).
#   From then on d2_launch owns everything: it waits for every gate (never competes for a card), finalizes A2 once
#   C1/C2 seal, applies D2_DECISIONS, trains, and parks before registration while d2_delta_recipe.py is not built.
# NEVER: arms a second queue ($Q/d2.lock held, or a `<launch> run` process alive -> exit 0), touches a card, a lock of
#   another lane, post_d1, d1_seal, or any D2 input.  CPU only.
#
#   setsid nohup bash $T/run_d2_arm_2026-10-15.sh > $Q/logs/d2_arm.log 2>&1 < /dev/null &
#   bash $T/run_d2_arm_2026-10-15.sh --dry-run     # the two conditions as they stand; writes and starts nothing
# Exit: 0 armed (or a D2 queue already holds $Q/d2.lock; --dry-run); 1 the armed queue did not take $Q/d2.lock;
#   4 the give-up time passed unarmed; 5 another arm watcher holds $Q/d2_arm.lock.
set -u
VERSION=d2_arm-1.0.0
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
# D2ARM_* are test hooks (test_run_d2_arm.sh); none is set in production.
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
Q=${D2ARM_Q:-$OC/distill/queues}
LAUNCH=${D2ARM_LAUNCH:-$Q/d2_launch.sh}
D2LOG=${D2ARM_D2LOG:-/mnt/lexar/omni-campaign/distill/d2/logs}
POSTD1_MATCH=${D2ARM_POSTD1_MATCH:-$Q/post_d1_queue.sh}
GIVE_UP=${D2ARM_GIVE_UP:-2026-10-26T00:00:00Z}
POLL_S=${D2ARM_POLL_S:-300}
REFUSED_RETRY_S=${D2ARM_REFUSED_RETRY_S:-1800}
SETTLE_S=${D2ARM_SETTLE_S:-30}
STATUS=$Q/d2_arm.status
P2R=$Q/post_d1/p2.json
K3R=$Q/post_d1/k3decide.json

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
status() {  # $1 short state, rest detail: one line to $STATUS; an L-TRAIN heartbeat (never fatal, silent with no
  # registry).  "waiting" sets the detail only (the lane status stays the D2 lane's); every final state sets both.
  stamp "STATUS $*"; echo "$(date -u +%FT%TZ) $VERSION $*" >> "$STATUS"
  [ -f "$Q/LANES.json" ] || return 0
  local d="$*" st=()
  [ "$1" = waiting ] || st=(--status "D2 queue ($VERSION): $1")
  timeout 20 python3 "$T/lanes.py" --file "$Q/LANES.json" heartbeat --lane L-TRAIN --agent claude --source "$VERSION" \
    "${st[@]}" --detail "${d:0:400}" > /dev/null 2>&1 || stamp "lane heartbeat failed (ignored)"
}
sha16() { sha256sum "$1" 2>/dev/null | cut -c1-16; }
d2_queue_live() {  # a D2 queue holds its single-instance lock, or a `<launch> run` process lives
  if [ -e "$Q/d2.lock" ] && ! ( flock -n 8 ) 8< "$Q/d2.lock"; then echo "$Q/d2.lock held"; return 0; fi
  local p; p=$(pgrep -f -- "$LAUNCH run" | head -1)
  [ -n "$p" ] && { echo "pid $p ($LAUNCH run)"; return 0; }
  return 1
}
postd1_live() { pgrep -f -- "$POSTD1_MATCH" > /dev/null; }
conditions() {  # prints why not yet, or nothing when both conditions hold
  [ -f "$P2R" ] || { echo "P2 not finished (no $P2R)"; return; }
  [ -f "$K3R" ] && return
  postd1_live && { echo "K3-first / D2-first not decided yet (no $K3R; post_d1 alive)"; return; }
  return 0
}

stamp "watcher start $VERSION; launch $LAUNCH $(sha16 "$LAUNCH") ($(grep -m1 '^VERSION=' "$LAUNCH" 2>/dev/null | cut -d'"' -f2)); give-up $GIVE_UP"
if [ $DRY -eq 1 ]; then
  live=$(d2_queue_live) && stamp "DRY: a D2 queue is already live: $live"
  why=$(conditions)
  stamp "DRY: p2 receipt $([ -f "$P2R" ] && echo present || echo absent); k3decide receipt $([ -f "$K3R" ] && echo present || echo absent); post_d1 $(postd1_live && echo alive || echo 'not alive')"
  stamp "DRY: $([ -z "$why" ] && echo "would run preflight and arm now" || echo "would wait: $why"); nothing was written"
  exit 0
fi
exec 9>> "$Q/d2_arm.lock"
flock -n 9 || { stamp "another arm watcher holds $Q/d2_arm.lock"; exit 5; }
last=""; status "waiting" "armed watcher pid $$: arms d2_launch run once P2 has finished and K3-first/D2-first is decided"
while :; do
  if live=$(d2_queue_live); then status "already armed" "a D2 queue is live ($live): nothing to do"; exit 0; fi
  why=$(conditions)
  if [ "$(date -u +%s)" -ge "$(date -u -d "$GIVE_UP" +%s)" ]; then status "gave up" "not armed by $GIVE_UP: ${why:-preflight refused}"; exit 4; fi
  if [ -z "$why" ]; then
    pf=$Q/logs/d2_arm.preflight.log; mkdir -p "$Q/logs"
    ( cd /tmp && D2_PREFLIGHT_DATA_CHECK=0 timeout 1800 bash "$LAUNCH" preflight ) > "$pf" 2>&1; prc=$?
    gates=$(grep '^D2 GATES open' "$pf" | tail -1 | cut -c1-300)
    if [ $prc -eq 2 ]; then
      status "REFUSED" "preflight rc 2, not armed: $(grep -E 'REFUSED' "$pf" | head -3 | tr '\n' ' ' | cut -c1-300) (re-checked every ${REFUSED_RETRY_S}s; log $pf)"
      sleep "$REFUSED_RETRY_S"; continue
    fi
    mkdir -p "$D2LOG"
    lsha=$(sha256sum "$LAUNCH" | cut -c1-64); lver=$(grep -m1 '^VERSION=' "$LAUNCH" | cut -d'"' -f2)
    stamp "arming: preflight rc $prc ($gates); launch $lver $lsha"
    ( cd /tmp && exec setsid nohup bash "$LAUNCH" run >> "$D2LOG/queue.log" 2>&1 < /dev/null 9>&- ) &
    sleep "$SETTLE_S"
    if live=$(d2_queue_live) && [ -e "$Q/d2.lock" ] && ! ( flock -n 8 ) 8< "$Q/d2.lock"; then
      status "armed" "d2_launch run ($lver ${lsha:0:16}) armed by $VERSION: $live; preflight rc $prc: ${gates#D2 GATES }; log $D2LOG/queue.log; status $Q/d2.status"
      exit 0
    fi
    status "FAILED" "the launched d2_launch run did not take $Q/d2.lock within ${SETTLE_S}s: $(tail -3 "$D2LOG/queue.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    exit 1
  fi
  [ "$why" != "$last" ] && { stamp "STATUS waiting: $why"; echo "$(date -u +%FT%TZ) $VERSION waiting: $why" >> "$STATUS"; last=$why; }
  sleep "$POLL_S"
done
