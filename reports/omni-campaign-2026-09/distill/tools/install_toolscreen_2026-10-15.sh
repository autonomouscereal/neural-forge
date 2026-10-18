#!/bin/bash
# install_toolscreen_2026-10-15.sh -- install post_d1_queue.sh 1.5.0 (C1 order: D2-X1 held-out screen -> rescore) and
# re-arm post_d1 (lane toolscreen, Claude, 2026-10-15).  Safe only BEFORE the D1 seal: the armed post_d1 1.4.0 only
# sleeps until $Q/d1.sealed exists, so stop-waiter, install, restart loses nothing (done jobs are skipped; none has run).
# Same procedure as install_sealdecide_2026-10-15.sh (1.3.0 -> 1.4.0).
#   bash install_toolscreen_2026-10-15.sh <staging dir> <receipt dir>
# Refuses (exit 2) when: $Q/d1.sealed exists; the watcher pid is not post_d1_queue.sh; its last status line is not the
# no_d1.sealed wait; the live post_d1_queue.sh is not the 1.4.0 it replaces; a staged file's sha differs from the pin;
# the screen tool it calls is not deployed at its pin; any post_d1 job receipt exists.  Stops ONLY that bash and its
# `sleep` children.
set -u
D=${1:?staging dir}; R=${2:?receipt dir}
OC=/home/cereal/omni-campaign; Q=$OC/distill/queues; T=$OC/distill/tools
OLD_PID=${OLD_PID:-1920111}
OLD_SHA=d24649b77d815b39e3cb531c3a57add547420ae2f90dcdb347e3e63a29573410   # post_d1_queue.sh 1.4.0 (lane sealdecide)
WANT_POST=${WANT_POST:?sha of post_d1_queue.sh 1.5.0}; WANT_TPOST=${WANT_TPOST:?}; WANT_SCREEN=${WANT_SCREEN:?sha of the deployed toolcall_heldout_screen.py}
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R" "$Q/backups" "$T/_superseded"
echo "$(stamp) install start (old pid $OLD_PID)"
# -- refusals
[ -e "$Q/d1.sealed" ] && { echo "REFUSE: $Q/d1.sealed exists: post_d1 may be past its wait"; exit 2; }
tr '\0' ' ' < /proc/$OLD_PID/cmdline 2>/dev/null | grep -q "bash $Q/post_d1_queue.sh" || { echo "REFUSE: pid $OLD_PID is not post_d1_queue.sh"; exit 2; }
tail -1 "$Q/post_d1.status" | grep -q "post_d1 state=waiting reason=no_d1.sealed" || { echo "REFUSE: post_d1 is not in its d1.sealed wait: $(tail -1 "$Q/post_d1.status")"; exit 2; }
[ "$(sha256sum < "$Q/post_d1_queue.sh" | cut -c1-64)" = "$OLD_SHA" ] || { echo "REFUSE: live post_d1_queue.sh is not 1.4.0 ($OLD_SHA)"; exit 2; }
for p in "post_d1_queue.sh $WANT_POST" "test_post_d1_queue.sh $WANT_TPOST"; do
  set -- $p; [ "$(sha256sum < "$D/$1" | cut -c1-64)" = "$2" ] || { echo "REFUSE: staged $1 sha drift"; exit 2; }
done
[ "$(sha256sum < "$T/toolcall_heldout_screen.py" | cut -c1-64)" = "$WANT_SCREEN" ] || { echo "REFUSE: $T/toolcall_heldout_screen.py is not $WANT_SCREEN"; exit 2; }
ls "$Q/post_d1"/*.json > /dev/null 2>&1 && { echo "REFUSE: post_d1 job receipts exist ($(ls "$Q/post_d1"))"; exit 2; }
tr '\0' '\n' < /proc/$OLD_PID/environ | grep '^POSTD1_' > "$R/post_d1.env.before"
ENVV=$(grep '^POSTD1_C2_CORPUS=' "$R/post_d1.env.before" | cut -d= -f2-)
[ -n "$ENVV" ] || { echo "REFUSE: POSTD1_C2_CORPUS not in the old watcher's environment"; exit 2; }
[ "$(grep -c '^POSTD1_' "$R/post_d1.env.before")" = 1 ] || { echo "REFUSE: the old watcher carries other POSTD1_ variables: $(cat "$R/post_d1.env.before")"; exit 2; }
echo "env: POSTD1_C2_CORPUS=$ENVV"
# -- backups (live 1.4.0 by sha in $Q/backups and tools/_superseded, as 1.3.0 was)
cp -p "$Q/post_d1_queue.sh" "$Q/backups/post_d1_queue.sh.1.4.0-live-$TS"
cp -p "$Q/test_post_d1_queue.sh" "$Q/backups/test_post_d1_queue.sh.1.4.0-$TS"
cp -p "$Q/post_d1_queue.sh" "$T/_superseded/post_d1_queue_1.4.0.sh"
cp -p "$Q/test_post_d1_queue.sh" "$T/_superseded/test_post_d1_queue_1.4.0.sh"
cp -p "$Q/post_d1.status" "$R/post_d1.status.before"
sha256sum "$Q/backups/post_d1_queue.sh.1.4.0-live-$TS" "$Q/backups/test_post_d1_queue.sh.1.4.0-$TS" "$T/_superseded/post_d1_queue_1.4.0.sh" "$T/_superseded/test_post_d1_queue_1.4.0.sh"
# -- stop the waiter (its bash and its sleep children only; the sleep holds fd 9 = post_d1.lock)
KIDS=$(ps -o pid=,args= --ppid $OLD_PID | awk '$2 == "sleep" {print $1}' | tr '\n' ' ')
ps -o pid,ppid,etime,args --ppid $OLD_PID > "$R/old_children.txt"
echo "$(stamp) stop: kill -TERM $OLD_PID; sleep children: ${KIDS:-none}"
kill -TERM $OLD_PID
for k in $KIDS; do kill -TERM "$k" 2>/dev/null; done
for i in $(seq 1 30); do [ -d /proc/$OLD_PID ] || break; sleep 1; done
[ -d /proc/$OLD_PID ] && { echo "FAIL: pid $OLD_PID still alive after 30 s"; exit 3; }
for i in $(seq 1 30); do ( exec 3< "$Q/post_d1.lock" && flock -n 3 ) 2>/dev/null && break; sleep 1; done
( exec 3< "$Q/post_d1.lock" && flock -n 3 ) 2>/dev/null || { echo "FAIL: post_d1.lock still held: $(lsof "$Q/post_d1.lock" 2>/dev/null | tail -3)"; exit 3; }
echo "$(stamp) $OLD_PID stopped, post_d1.lock free"
echo "$(stamp) post_d1 state=stopped version=post_d1_queue-1.4.0 pid=$OLD_PID by=install_toolscreen_2026-10-15 (re-armed as 1.5.0: c1 = heldout screen -> rescore)" >> "$Q/post_d1.status"
# -- install
cp "$D/post_d1_queue.sh" "$Q/post_d1_queue.sh"; cp "$D/test_post_d1_queue.sh" "$Q/test_post_d1_queue.sh"
sha256sum "$Q/post_d1_queue.sh" "$Q/test_post_d1_queue.sh" | tee "$R/installed.sha256"
[ "$(sha256sum < "$Q/post_d1_queue.sh" | cut -c1-64)" = "$WANT_POST" ] || { echo "FAIL: installed sha drift"; exit 3; }
# -- gate check before arming
bash "$Q/post_d1_queue.sh" --dry-run > "$R/post_dry.log" 2>&1; echo "post_d1 --dry-run rc=$?"; grep -E "c1|DRY RUN" "$R/post_dry.log" | tail -4
# -- re-arm, same command as install_sealdecide (log appended, not truncated)
echo "$(stamp) arming post_d1_queue 1.5.0 (POSTD1_C2_CORPUS=$ENVV)"
POSTD1_C2_CORPUS=$ENVV setsid nohup bash "$Q/post_d1_queue.sh" >> "$Q/logs/post_d1.log" 2>&1 < /dev/null &
NEW=$!
sleep 8
echo "new pid $NEW: $(tr '\0' ' ' < /proc/$NEW/cmdline 2>/dev/null)"
echo "env: $(tr '\0' '\n' < /proc/$NEW/environ 2>/dev/null | grep '^POSTD1_')"
echo "$NEW" > "$R/new_pid"
tail -3 "$Q/post_d1.status"
tail -2 "$Q/post_d1.status" | grep -q "state=armed version=post_d1_queue-1.5.0 jobs=\"p2 k3decide c1 c2 j6 h3 p11\" pid=$NEW" \
  && tail -1 "$Q/post_d1.status" | grep -q "state=waiting reason=no_d1.sealed" && echo "ARMED OK: pid $NEW waits for d1.sealed" || { echo "FAIL: not in its wait state"; exit 4; }
echo "$(stamp) install done"
