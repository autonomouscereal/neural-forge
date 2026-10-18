#!/usr/bin/env bash
# test_x3_v1_yield_watchdog.sh -- x3_v1_yield_watchdog.py against dummy v1 queues (gpu6, post_d1, d1_seal sleepers
# with the real names), a fake /proc/meminfo, and a REAL x3_queue.sh (temp copy) whose stages are a TERM-aware stub.
# Everything lives in a temp dir: no real claims file, lock, status file or card is touched, and the production fold
# queue cannot match (the watchdog under test matches the TEMP queue path and the TEMP $X only).
# Part P: every pressure signal, positive and negative (--probe).  Part E: yield and resume end to end.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D1LIB_DIR=${D1LIB_DIR:-/home/cereal/omni-campaign/distill/tools}
D=$(mktemp -d /tmp/x3wd-test.XXXXXX)
WD=""
cleanup() { [ -n "$WD" ] && kill $WD 2>/dev/null; pkill -f "$D/" 2>/dev/null; sleep 1; pkill -9 -f "$D/" 2>/dev/null; rm -rf $D; }
trap cleanup EXIT
fail=0
ok() { echo "ok   [$1]"; }
bad() { echo "FAIL [$1] $2"; fail=1; }
mkdir -p $D/tools $D/q/logs/gpu6 $D/x/children/QF08-C2K896 $D/xl $D/legacy
cp $HERE/x3_queue.sh $HERE/x3_v1_yield_watchdog.py $D/tools/
touch $D/x/children/QF08-C2K896/qf-build-receipt.json   # build done: the queue's first stage is eval-parent (the stub)
cat > $D/stubpy <<'EOF'
#!/bin/bash
echo "$(date -u +%FT%TZ) stub start $$ $*"
trap 'echo "$(date -u +%FT%TZ) stub TERM $$"; exit 143' TERM
for i in $(seq 1 900); do sleep 1; done
EOF
printf '#!/bin/sh\necho "3, 100"\n' > $D/nvsmi
printf '#!/bin/bash\nsleep 3000\n' > $D/q/gpu6.sh
cp $D/q/gpu6.sh $D/q/d1_seal_queue.sh; cp $D/q/gpu6.sh $D/q/post_d1_queue.sh; cp $D/q/gpu6.sh $D/legacy/x3_queue.sh
chmod +x $D/stubpy $D/nvsmi
: > $D/q/p40.claims
export X3Q_ROOT=$D/x CLAIMS=$D/q/p40.claims LOCK3=$D/q/gpu3.lock NVSMI=$D/nvsmi X3Q_PY=$D/stubpy NEED_GIB=0 POLL_S=1
mem() {  # mem <MemAvailable GiB> [swap used %]
  printf 'MemTotal: 263975028 kB\nMemAvailable: %d kB\nSwapTotal: 8388604 kB\nSwapFree: %d kB\n' \
    $(( $1 * 1048576 )) $(( 8388604 * (100 - ${2:-0}) / 100 )) > $D/meminfo
}
WARGS=(--x $D/x --queue-script $D/tools/x3_queue.sh --q $D/q --d1-runtime $D/runtime-status.json --meminfo $D/meminfo
       --lanes "" --poll 1 --resume-hold-s 4)
probe() { python3 $D/tools/x3_v1_yield_watchdog.py "${WARGS[@]}" --probe \
          | python3 -c 'import json,sys; print(",".join(sorted(json.load(sys.stdin)["pressure"])) or "none")'; }
expect() { local got; got=$(probe); [ "$got" = "$1" ] && ok "$2 -> $got" || bad "$2" "want '$1' got '$got'"; }
now() { date -u +%FT%TZ; }
wait_until() { local end=$((SECONDS + $1)); shift; while [ $SECONDS -lt $end ]; do "$@" && return 0; sleep 0.5; done; return 1; }
queue_alive() { pgrep -f "^bash $D/tools/x3_queue.sh" > /dev/null; }
queue_dead() { ! queue_alive; }
yielded() { [ -f $D/x/V1_YIELD.json ]; }
lock_state() { ( exec 7>>$D/q/gpu3.lock; flock -n 7 ) 2>/dev/null && echo free || echo busy; }
lock_busy() { [ "$(lock_state)" = busy ]; }
lock_free() { [ "$(lock_state)" = free ]; }
seal_reads() { python3 -c "import sys, pathlib; sys.path.insert(0, '$D1LIB_DIR'); import d1_seal_lib as L
print('held' if L.p40_free(pathlib.Path('$CLAIMS'), [3])[0] else 'free')"; }
fold_last() { grep "lane=fold-research" $CLAIMS | tail -1; }
starts() { grep -c "stub start" $D/x/logs/eval-parent.log 2>/dev/null || echo 0; }
running() { queue_alive && [ "$(starts)" -ge "$1" ] && grep -q "CLAIM GPU3 owner=claude lane=fold-research" $CLAIMS && lock_busy; }

# ============================================================== P: signals (--probe), no fold job
mem 200; : > $D/runtime-status.json
expect none "idle host, nothing waiting"
mem 85 95; expect swap_full "swap 95 % used and MemAvailable 85 < 90"
mem 95 95; expect none "swap 95 % but MemAvailable 95 >= 90"
mem 85 50; expect none "MemAvailable 85 but swap 50 %"
mem 200
bash $D/q/gpu6.sh >> $D/q/logs/gpu6/queue-1.3.0.log 2>&1 & G6=$!
sleep 0.3
echo "$(now) MemAvailable 60 GiB < 90; waiting" >> $D/q/logs/gpu6/queue-1.3.0.log
expect gpu6_wait_ram "gpu6 queue wait_ram line (read through /proc/<pid>/fd/1)"
echo "$(now) MemAvailable 95 GiB" >> $D/q/logs/gpu6/queue-1.3.0.log
expect none "gpu6 wait_ram ended"
echo "$(now) k3-pass-0 WAIT S8/GPU6, MemAvailable >= 90 GiB (wait, never kill)" >> $D/q/gpu6.status
mem 70; expect k3_queued_low_mem "gpu6 job WAIT and MemAvailable 70 < 75"
mem 80; expect none "gpu6 job WAIT and MemAvailable 80"
echo "$(now) k3-pass-0 START pass 0" >> $D/q/gpu6.status
mem 70; expect none "gpu6 job running (START), MemAvailable 70"
mem 200
echo '{"state": "waiting_for_memory", "waiting_on": "MemAvailable 50.3 GiB < 70 GiB"}' > $D/runtime-status.json
expect d1_wait_mem "D1 runtime-status waiting_on MemAvailable"
echo '{"state": "running", "waiting_on": null}' > $D/runtime-status.json
expect none "D1 running"
bash $D/q/post_d1_queue.sh & PD=$!
echo "$(now) post_d1 state=armed version=post_d1_queue-1.3.0 jobs=\"p2 c1\" pid=$PD" >> $D/q/post_d1.status
echo "$(now) post_d1 job=c1 state=deferred reason=\"MemAvailable_60GiB_lt_70\" (gpu0.lock not taken)" >> $D/q/post_d1.status
expect post_d1_wait_mem "post_d1 deferred on MemAvailable (armed pid alive)"
kill $PD; wait $PD 2>/dev/null
expect none "post_d1 line unchanged but its queue is gone"
bash $D/q/d1_seal_queue.sh & SQ=$!
echo "$(now) d1_seal state=armed version=d1_seal_queue-1.2.0 phase=all pid=$SQ" >> $D/q/d1_seal.status
echo "$(now) d1_seal step=A8-s7 state=waiting reason=MemAvailable_30GiB_lt_40" >> $D/q/d1_seal.status
expect d1_seal_wait_mem "d1_seal wait_mem"
echo "$(now) d1_seal step=A9-sealed state=done marker=x" >> $D/q/d1_seal.status
expect none "d1_seal phase A done"
echo "$(now) d1_seal step=B1-cards state=waiting attempt=1/3 s8(gpu6/4/0 locks + cards) p40(P40 HELD GPU3: x)" >> $D/q/d1_seal.status
expect gpu3_wait "d1_seal B1-cards waiting (review needs the P40s)"
echo "$(now) d1_seal step=B1-cards state=held attempt=1/3 placement=p40 gpus=1,2,3 dtype=float16" >> $D/q/d1_seal.status
expect gpu3_wait "d1_seal holds GPU1,2,3 for the review"
echo "$(now) d1_seal step=B event=released placement=p40 gpus=1,2,3" >> $D/q/d1_seal.status
expect none "d1_seal released the P40s"
echo "$(now) d1_seal step=B1-cards state=retrying attempt=2/3 in 300s" >> $D/q/d1_seal.status
expect gpu3_wait "d1_seal retrying after a failed review"
echo "$(now) d1_seal step=B1-cards state=held attempt=2/3 placement=s8 gpus=6,4,0" >> $D/q/d1_seal.status
expect none "d1_seal review on s8 (GPU6,4,0): GPU3 not needed"
kill $SQ; wait $SQ 2>/dev/null

# ============================================================== E1: legacy 1.0.0 claim line, restated + gpu3.lock
bash $D/legacy/x3_queue.sh & LQ=$!
echo "$(now) GPU3 lane=fold-research owner=claude job=x3-08b pid=$LQ log=x preemptible: kill -TERM $LQ (trainer checkpoints, claim released within ~3 min, then re-queues); stop for good: touch X/STOP_QUEUE" >> $CLAIMS
[ "$(seal_reads)" = free ] && ok "E1 before: the seal queue's parser reads the legacy claim as FREE (the bug)" || bad "E1 before" "$(seal_reads)"
python3 $D/tools/x3_v1_yield_watchdog.py --x $D/xl --queue-script $D/legacy/x3_queue.sh --q $D/q --d1-runtime $D/runtime-status.json \
  --meminfo $D/meminfo --lanes "" --poll 1 > $D/wd-legacy.out 2>&1 & WD=$!
if wait_until 8 grep -q "restated by x3_v1_yield_watchdog" $CLAIMS; then ok "E1 restated CLAIM line appended"; else bad "E1 restate" "$(tail -3 $D/wd-legacy.out)"; fi
[ "$(seal_reads)" = held ] && ok "E1 the seal queue's parser now reads GPU3 HELD" || bad "E1 seal" "$(seal_reads)"
wait_until 5 lock_busy && ok "E1 gpu3.lock held by the watchdog for the live legacy claim" || bad "E1 lock" "$(lock_state)"
echo "$(now) GPU3 lane=fold-research job=x3-08b released (test)" >> $CLAIMS
wait_until 6 lock_free && ok "E1 lock released after the fold lane's release line" || bad "E1 lock release" "$(lock_state)"
[ "$(seal_reads)" = free ] && ok "E1 seal reads free after the release" || bad "E1 seal after" "$(seal_reads)"
[ ! -e $D/xl/V1_YIELD.json ] && ok "E1 no yield without pressure" || bad "E1 yield" "$(cat $D/xl/V1_YIELD.json)"
kill $WD; wait $WD 2>/dev/null; WD=""; kill $LQ; wait $LQ 2>/dev/null

# ============================================================== E2: RAM pressure (gpu6 wait_ram) yields the real queue
: > $CLAIMS; mem 200
setsid bash $D/tools/x3_queue.sh >> $D/x/queue.log 2>&1 < /dev/null &
wait_until 15 running 1 && ok "E2 x3_queue.sh 1.1.0 claimed GPU3 (CLAIM line, gpu3.lock) and runs the stub stage" \
  || bad "E2 start" "$(tail -3 $D/x/queue.status)"
[ "$(seal_reads)" = held ] && ok "E2 the seal queue's parser reads the 1.1.0 claim as HELD" || bad "E2 seal" "$(seal_reads)"
python3 $D/tools/x3_v1_yield_watchdog.py "${WARGS[@]}" > $D/wd.out 2>&1 & WD=$!
sleep 3
queue_alive && ! yielded && ok "E2 no pressure: the watchdog leaves the job alone" || bad "E2 idle" "$(tail -3 $D/x/v1_yield_watchdog.log)"
T0=$SECONDS
echo "$(now) MemAvailable 60 GiB < 90; waiting" >> $D/q/logs/gpu6/queue-1.3.0.log
wait_until 20 queue_dead && ok "E2 gpu6 wait_ram -> queue stopped in $((SECONDS - T0)) s" || bad "E2 yield" "$(tail -5 $D/x/v1_yield_watchdog.log)"
grep -q "stub TERM" $D/x/logs/eval-parent.log && ok "E2 the stage got SIGTERM" || bad "E2 stage TERM" "$(tail -2 $D/x/logs/eval-parent.log)"
yielded && grep -q "^v1-yield-watchdog" $D/x/STOP_QUEUE && ok "E2 V1_YIELD.json + own STOP_QUEUE written" || bad "E2 marker" "$(ls $D/x)"
fold_last | grep -q "RELEASE GPU3 owner=claude lane=fold-research job=x3-08b released (preempted)" \
  && ok "E2 queue wrote its RELEASE line" || bad "E2 release line" "$(fold_last)"
wait_until 5 lock_free && ok "E2 gpu3.lock free after the yield" || bad "E2 lock" "$(lock_state)"

# ============================================================== E3: resume gate (110 GiB for the hold window, dips reset it)
echo "$(now) MemAvailable 95 GiB" >> $D/q/logs/gpu6/queue-1.3.0.log
mem 100; sleep 7
queue_dead && yielded && ok "E3 MemAvailable 100 < 110: still held after 7 s" || bad "E3 held" "$(tail -3 $D/x/v1_yield_watchdog.log)"
mem 120; sleep 2; mem 100; sleep 2; mem 120; T0=$SECONDS
wait_until 15 queue_alive; EL=$((SECONDS - T0))
queue_alive && [ $EL -ge 4 ] && ok "E3 relaunched ${EL} s after the last dip (window 4 s; the dip reset it)" \
  || bad "E3 resume" "alive=$(queue_alive && echo y || echo n) after ${EL}s; $(tail -4 $D/x/v1_yield_watchdog.log)"
wait_until 15 running 2 && ok "E3 relaunched queue claimed GPU3 again and runs the stage (start #2)" || bad "E3 run" "$(tail -3 $D/x/queue.status)"
[ ! -e $D/x/STOP_QUEUE ] && ! yielded && ls $D/x/V1_YIELD.json.done-* > /dev/null 2>&1 \
  && ok "E3 own STOP_QUEUE removed, yield marker archived" || bad "E3 cleanup" "$(ls $D/x)"

# ============================================================== E4: swap > 90 % with MemAvailable < 90
mem 95 95; sleep 3
queue_alive && ok "E4 swap 95 % but MemAvailable 95: no yield" || bad "E4 neg" "$(tail -3 $D/x/v1_yield_watchdog.log)"
mem 85 95; T0=$SECONDS
wait_until 20 queue_dead && ok "E4 swap 95 % + MemAvailable 85 -> queue stopped in $((SECONDS - T0)) s" || bad "E4 yield" "$(tail -4 $D/x/v1_yield_watchdog.log)"
mem 95 95; sleep 6
queue_dead && ok "E4 MemAvailable 95 (< 110): held" || bad "E4 held" "relaunched early"
mem 120 95
wait_until 15 running 3 && ok "E4 MemAvailable 120 for the window -> relaunched (start #3)" || bad "E4 resume" "$(tail -4 $D/x/v1_yield_watchdog.log)"

# ============================================================== E5: the D1 seal review needs GPU3
mem 200
bash $D/q/d1_seal_queue.sh & SQ=$!
echo "$(now) d1_seal state=armed version=d1_seal_queue-1.2.0 phase=all pid=$SQ" >> $D/q/d1_seal.status
echo "$(now) d1_seal step=B1-cards state=waiting attempt=1/3 s8(gpu6/4/0 locks + cards) p40(P40 HELD GPU3: fold)" >> $D/q/d1_seal.status
T0=$SECONDS
wait_until 20 queue_dead && ok "E5 seal B1 waiting for cards -> fold job stopped in $((SECONDS - T0)) s" || bad "E5 yield" "$(tail -4 $D/x/v1_yield_watchdog.log)"
wait_until 5 lock_free && [ "$(seal_reads)" = free ] && ok "E5 GPU3 free for the seal (lock free, claims read free)" || bad "E5 free" "$(lock_state) $(seal_reads)"
echo "$(now) CLAIM GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review pid=$SQ log=x (release by d1_seal_queue-1.2.0)" >> $CLAIMS
echo "$(now) d1_seal step=B1-cards state=held attempt=1/3 placement=p40 gpus=1,2,3" >> $D/q/d1_seal.status
echo "$(now) d1_seal step=B2-thinkoff state=started attempt=1/3 placement=p40 gpus=1,2,3" >> $D/q/d1_seal.status
sleep 8
queue_dead && ok "E5 review running on the P40s, MemAvailable 200: held for 8 s (> window)" || bad "E5 held" "relaunched during the review"
echo "$(now) d1_seal step=B2-thinkoff state=done dossier=x" >> $D/q/d1_seal.status
echo "$(now) RELEASE GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review released" >> $CLAIMS
echo "$(now) d1_seal step=B event=released placement=p40 gpus=1,2,3" >> $D/q/d1_seal.status
T0=$SECONDS
wait_until 15 running 4 && ok "E5 review released GPU3 -> relaunched after $((SECONDS - T0)) s and claimed GPU3 (start #4)" \
  || bad "E5 resume" "$(tail -4 $D/x/v1_yield_watchdog.log)"
kill $SQ; wait $SQ 2>/dev/null

# ============================================================== E6: a foreign STOP_QUEUE is never overridden
echo "operator: stop for good" > $D/x/STOP_QUEUE
mem 85 95
wait_until 20 queue_dead && ok "E6 pressure with a foreign STOP_QUEUE: job stopped" || bad "E6 yield" "$(tail -3 $D/x/v1_yield_watchdog.log)"
mem 200; sleep 7
queue_dead && grep -q "not this watchdog's" $D/x/v1_yield_watchdog.log && ok "E6 no relaunch over a foreign STOP_QUEUE" \
  || bad "E6 relaunch" "$(tail -3 $D/x/v1_yield_watchdog.log)"
rm -f $D/x/STOP_QUEUE
wait_until 5 grep -q "yield marker VOID" $D/x/v1_yield_watchdog.log && ! yielded && sleep 3 && queue_dead \
  && ok "E6 STOP_QUEUE removed by its owner: marker voided, nothing relaunched by the watchdog" || bad "E6 void" "$(tail -3 $D/x/v1_yield_watchdog.log)"

kill -TERM $WD; wait $WD; rc=$?; WD=""
[ $rc = 0 ] && grep -q "EXIT pid" $D/x/v1_yield_watchdog.log && ok "watchdog exits 0 on SIGTERM" || bad "exit" "rc $rc"
kill $G6 2>/dev/null
echo "--- watchdog log (E2-E6) ---"; cat $D/x/v1_yield_watchdog.log
[ $fail = 0 ] && echo "ALL OK" || { echo "FAILURES"; exit 1; }
