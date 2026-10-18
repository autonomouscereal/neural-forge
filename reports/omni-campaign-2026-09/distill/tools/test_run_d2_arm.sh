#!/usr/bin/env bash
# test_run_d2_arm.sh -- run_d2_arm_2026-10-15.sh against a stub d2_launch.sh and a stand-in post_d1, every path under
# declared scratch (lane d2ready, 2026-10-15).  No card, no live lock, no LANES.json (heartbeats silent).
#   bash test_run_d2_arm.sh [script under test, default $T/run_d2_arm_2026-10-15.sh]
#   waits     waits while P2 is unfinished, then while post_d1 lives with no k3decide receipt; arms once both exist
#   p2gate    a k3decide receipt alone (no P2 receipt) never arms (the verdict first-writer race)
#   postd1    P2 finished and post_d1 dead (no k3decide receipt): arms (nothing would decide)
#   refused   preflight rc 2: not armed, status REFUSED, re-checked; arms when preflight stops refusing
#   already   a queue holds d2.lock: rc 0, nothing launched
#   dryrun    conditions met: rc 0, nothing launched, no status written
#   giveup    the give-up time passes unarmed: rc 4, nothing launched
#   died      the launched queue never takes d2.lock: rc 1, FAILED
#   single    a second watcher: rc 5
#   fds       the armed queue does not inherit the watcher's d2_arm.lock
VERSION="test_run_d2_arm 1.0.0"
S=${1:-/home/cereal/omni-campaign/distill/tools/run_d2_arm_2026-10-15.sh}
TD=${D2ARM_TEST_DIR:-/mnt/lexar/omni-campaign/scratch/d2arm-test-$$}
rm -rf "$TD"; mkdir -p "$TD/q/post_d1" "$TD/q/logs" "$TD/d2log"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }
L=$TD/q/d2_launch.sh
cat > "$L" <<EOF
#!/usr/bin/env bash
VERSION="d2_launch-9.9.9-stub"
case \$1 in
  preflight)
    n=\$(cat "$TD/pf_count" 2>/dev/null || echo 0); n=\$((n + 1)); echo \$n > "$TD/pf_count"
    rc=\${STUB_PF_RC:-10}; [ -n "\${STUB_PF_RC_FIRST:-}" ] && [ \$n -eq 1 ] && rc=\$STUB_PF_RC_FIRST
    [ \$rc -eq 2 ] && echo "  tools        REFUSED  trainer 0000 (want 1111)"
    echo "D2 GATES open 2 wait 7 refused \$([ \$rc -eq 2 ] && echo 1 || echo 0): stub"; exit \$rc ;;
  run)
    echo run >> "$TD/calls"
    [ -n "\${STUB_DIE:-}" ] && exit 2
    exec 3> "$TD/q/d2.lock"; flock -n 3 || exit 5
    sleep 31337 3>&- ;;
esac
EOF
cat > "$TD/post_d1_queue.sh" <<'EOF'
sleep 600
EOF
export D2ARM_Q=$TD/q D2ARM_LAUNCH=$L D2ARM_D2LOG=$TD/d2log D2ARM_POSTD1_MATCH="$TD/post_d1_queue.sh" \
       D2ARM_POLL_S=1 D2ARM_REFUSED_RETRY_S=1 D2ARM_SETTLE_S=2
reset() { pkill -f "$L run" 2>/dev/null; pkill -f "$TD/post_d1_queue.sh" 2>/dev/null; pkill -x -f "sleep 31337" 2>/dev/null; sleep 0.3
          rm -f "$TD/calls" "$TD/pf_count" "$TD/q/d2_arm.status" "$TD/q/post_d1/"*.json; }
runs() { grep -c run "$TD/calls" 2>/dev/null || echo 0; }

# waits
reset; bash "$TD/post_d1_queue.sh" & PD=$!
( sleep 3; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"; sleep 3; echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json" ) &
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$(runs)" = 1 ] && grep -q "waiting: P2 not finished" "$TD/q/d2_arm.status" \
  && grep -q "waiting: K3-first / D2-first not decided yet" "$TD/q/d2_arm.status" && grep -q "d2_arm-1.0.0 armed d2_launch run (d2_launch-9.9.9-stub" "$TD/q/d2_arm.status" \
  && ! ( flock -n 8 ) 8< "$TD/q/d2.lock"
check "waits: P2, then the decision; arms once, and the queue holds d2.lock" $? "rc=$rc runs=$(runs) $(cat "$TD/q/d2_arm.status") $OUT"
# fds: the armed stub must not hold the watcher's lock
( flock -n 8 ) 8< "$TD/q/d2_arm.lock"; check "fds: the armed queue does not inherit d2_arm.lock" $? "d2_arm.lock still held after the watcher exited"
kill $PD 2>/dev/null; wait $PD 2>/dev/null

# p2gate
reset; bash "$TD/post_d1_queue.sh" & PD=$!
echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json"
OUT=$(D2ARM_GIVE_UP=$(date -u -d '+4 seconds' +%FT%TZ) timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 4 ] && [ "$(runs)" = 0 ] && grep -q "gave up .*P2 not finished" "$TD/q/d2_arm.status"
check "p2gate: a decision receipt without the P2 receipt never arms" $? "rc=$rc runs=$(runs) $OUT"
kill $PD 2>/dev/null; wait $PD 2>/dev/null

# postd1
reset; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$(runs)" = 1 ]; check "postd1: P2 done, post_d1 dead, no decision -> arms" $? "rc=$rc runs=$(runs) $OUT"

# refused
reset; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"; echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json"
OUT=$(STUB_PF_RC_FIRST=2 timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$(runs)" = 1 ] && [ "$(cat "$TD/pf_count")" = 2 ] && grep -q "REFUSED preflight rc 2, not armed: .*tools *REFUSED" "$TD/q/d2_arm.status"
check "refused: preflight rc 2 -> not armed, REFUSED; armed on the re-check" $? "rc=$rc runs=$(runs) pf=$(cat "$TD/pf_count") $(cat "$TD/q/d2_arm.status")"
reset; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"; echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json"
OUT=$(STUB_PF_RC=2 D2ARM_GIVE_UP=$(date -u -d '+4 seconds' +%FT%TZ) timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 4 ] && [ "$(runs)" = 0 ] && [ "$(cat "$TD/pf_count")" -ge 2 ] && grep -q "gave up not armed by .*preflight refused" "$TD/q/d2_arm.status"
check "refused: a preflight that keeps refusing never arms, then gives up (rc 4)" $? "rc=$rc runs=$(runs) pf=$(cat "$TD/pf_count" 2>/dev/null) $(tail -1 "$TD/q/d2_arm.status")"

# already
reset; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"; echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json"
( exec 8> "$TD/q/d2.lock"; flock -n 8; exec sleep 3134 ) & H=$!; sleep 0.5
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
kill $H 2>/dev/null; wait $H 2>/dev/null
[ $rc -eq 0 ] && [ "$(runs)" = 0 ] && grep -q "already armed .*d2.lock held" "$TD/q/d2_arm.status"
check "already: d2.lock held -> rc 0, nothing launched" $? "rc=$rc runs=$(runs) $OUT"

# dryrun
reset; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"; echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json"
OUT=$(timeout 60 bash "$S" --dry-run 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$(runs)" = 0 ] && [ ! -e "$TD/q/d2_arm.status" ] && [ ! -e "$TD/pf_count" ] && echo "$OUT" | grep -q "would run preflight and arm now"
check "dryrun: rc 0, nothing launched or written" $? "rc=$rc runs=$(runs) $OUT"

# giveup
reset
OUT=$(D2ARM_GIVE_UP=2000-01-01T00:00:00Z timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 4 ] && [ "$(runs)" = 0 ]; check "giveup: rc 4, nothing launched" $? "rc=$rc $OUT"

# died
reset; echo '{"rc": 0}' > "$TD/q/post_d1/p2.json"; echo '{"rc": 0}' > "$TD/q/post_d1/k3decide.json"
OUT=$(STUB_DIE=1 timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 1 ] && [ "$(runs)" = 1 ] && grep -q "FAILED" "$TD/q/d2_arm.status"
check "died: the queue never takes d2.lock -> rc 1, FAILED" $? "rc=$rc runs=$(runs) $OUT"

# single
reset
( exec 9>> "$TD/q/d2_arm.lock"; flock -n 9; exec sleep 3135 ) & H=$!; sleep 0.5
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
kill $H 2>/dev/null; wait $H 2>/dev/null
[ $rc -eq 5 ] && [ "$(runs)" = 0 ]; check "single: a second watcher -> rc 5" $? "rc=$rc $OUT"

reset; rm -rf "$TD"
echo "D2 ARM WATCHER TEST: $pass passed, $fail failed"
[ $fail -eq 0 ]
