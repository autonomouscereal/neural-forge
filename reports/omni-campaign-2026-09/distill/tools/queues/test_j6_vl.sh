#!/usr/bin/env bash
# test_j6_vl.sh -- j6_vl.sh's calling convention, lock, done file and live-build-vl guard, on throwaway lock/done/
# status files (no GPU work; reads nvidia-smi only).  Written 2026-10-14 by Claude, lane vl, for j6_vl.sh 1.1.0.
#   bash test_j6_vl.sh [script under test, default $Q/j6_vl.sh]
# Every case that does not stop at the lock selftest is made unable to start the teacher: a fake
# `qtext_teacher_cache-j6.py build-vl` process is live, so a runner that fell past the case's own exit refuses.
#   q6-shape      queue_gpu6.sh 1.2.0 :692/:704 exactly: caller holds the lock on fd 8, J6_GPU=6, no argument
#                 -> rc 0, "held by the caller (inherited fd 8)", and the caller still holds the lock afterwards
#   usage         no argument and no J6_GPU -> 64; J6_GPU=4 / J6_GPU=5 / argument 4 -> 64; argument beats J6_GPU
#   own-lock      lock free -> "acquired (fd 9)", released at exit
#   contended     a third process holds the lock -> 5, with or without the caller's fd 8 open (unlocked) on it
#   build-vl      a live frozen-copy build-vl is seen by the lock selftest and refuses a real run (rc 3)
#   done          J6_SELFTEST=done writes the done file; a run with it present skips (rc 0, skipped line)
S=${1:-/home/cereal/omni-campaign/distill/queues/j6_vl.sh}
TD=${J6_TEST_DIR:-/mnt/lexar/omni-campaign/scratch/j6-vl-test-$$}
mkdir -p "$TD"
export J6_LOCK_FILE=$TD/j6.lock J6_DONE_FILE=$TD/j6.done J6_STATUS_FILE=$TD/j6.status J6_WAIT_MAX_S=0
UUID6=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 6)
pass=0; fail=0
check() {  # check <name> <condition result 0/1> <detail>
  if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi
}
fake_build_vl() { bash -c 'exec -a "python /x/queues/frozen/qtext_teacher_cache-j6.py build-vl" sleep 60' & FAKE=$!; sleep 0.5; }
stop_fake() { kill $FAKE 2>/dev/null; wait $FAKE 2>/dev/null; }

# q6-shape: the caller's lines, verbatim apart from the log path and the selftest switch
OUT=$( (
  exec 8> "$J6_LOCK_FILE"
  flock -n 8 || { echo "caller could not lock"; exit 99; }
  J6_SELFTEST=lock J6_GPU=6 CUDA_VISIBLE_DEVICES=$UUID6 bash "$S" < /dev/null 2>&1; echo "rc=$?"
  flock -n "$J6_LOCK_FILE" true && echo "caller-holds=no" || echo "caller-holds=yes"
  exec 8>&-
) )
echo "$OUT" | grep -q "^rc=0$" && echo "$OUT" | grep -q "J6 lock: held by the caller (inherited fd 8); gpu 6"
check "q6-shape: J6_GPU=6, caller holds fd 8 -> rc 0, inherited" $? "$OUT"
echo "$OUT" | grep -q "^caller-holds=yes$"; check "q6-shape: caller still holds the lock after the call" $? "$OUT"
flock -n "$J6_LOCK_FILE" true; check "q6-shape: lock free once the caller closed fd 8" $? "still held"

# usage
for case in "none::" "env4:4:" "env5:5:" "arg4::4"; do
  IFS=: read -r name env arg <<< "$case"
  OUT=$(J6_SELFTEST=lock J6_GPU=$env bash "$S" $arg < /dev/null 2>&1); rc=$?
  [ $rc -eq 64 ]; check "usage $name -> 64" $? "rc=$rc $OUT"
done
OUT=$(J6_SELFTEST=lock J6_GPU=4 bash "$S" 6 < /dev/null 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$OUT" | grep -q "; gpu 6;"; check "usage: argument 6 beats J6_GPU=4" $? "rc=$rc $OUT"

# own lock
OUT=$(J6_SELFTEST=lock bash "$S" 6 < /dev/null 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$OUT" | grep -q "J6 lock: acquired (fd 9)"; check "own-lock: free lock -> acquired (fd 9)" $? "rc=$rc $OUT"
flock -n "$J6_LOCK_FILE" true; check "own-lock: released at exit" $? "still held"

# contended
# the holder is one process (flock(1)'s own child would keep the lock after flock is killed)
bash -c 'exec 7> "$1"; flock 7; exec sleep 30' _ "$J6_LOCK_FILE" & HOLDER=$!; sleep 0.5
OUT=$(J6_SELFTEST=lock J6_GPU=6 bash "$S" < /dev/null 2>&1); rc=$?
[ $rc -eq 5 ]; check "contended: third holder -> 5" $? "rc=$rc $OUT"
OUT=$( ( exec 8> "$J6_LOCK_FILE"; J6_SELFTEST=lock J6_GPU=6 bash "$S" < /dev/null 2>&1; echo "rc=$?" ) )
echo "$OUT" | grep -q "^rc=5$"; check "contended: caller fd 8 open but unlocked, third holder -> 5" $? "$OUT"
kill $HOLDER 2>/dev/null; wait $HOLDER 2>/dev/null

# build-vl guard
REAL=$(pgrep -af 'qtext_teacher_cache[^ /]*\.py build-vl' | head -1)
if [ -z "$REAL" ]; then
  OUT=$(J6_SELFTEST=lock bash "$S" 6 < /dev/null 2>&1)
  echo "$OUT" | grep -q "other build-vl: none;"; check "build-vl: none live -> none" $? "$OUT"
else
  echo "skip build-vl none-case: a real build-vl is live ($REAL)"
fi
fake_build_vl
OUT=$(J6_SELFTEST=lock bash "$S" 6 < /dev/null 2>&1)
echo "$OUT" | grep -q "other build-vl: $FAKE python /x/queues/frozen/qtext_teacher_cache-j6.py build-vl"
GUARD_SEEN=$?; check "build-vl: the frozen copy's build-vl is seen" $GUARD_SEEN "$OUT"
if [ $GUARD_SEEN -eq 0 ]; then
  rm -f "$J6_DONE_FILE"
  OUT=$(J6_GPU=6 timeout 60 bash "$S" < /dev/null 2>&1); rc=$?
  [ $rc -eq 3 ] && grep -q "state=refused reason=a qtext build-vl is already running: $FAKE " "$J6_STATUS_FILE"
  check "build-vl: a real run refuses (rc 3, refused line)" $? "rc=$rc $OUT"
else
  check "build-vl: a real run refuses (not run: the guard did not see the fake)" 1 ""
fi

# done (the fake build-vl is still live: a runner that missed the done file refuses instead of starting)
OUT=$(J6_SELFTEST=done J6_GPU=6 bash "$S" < /dev/null 2>&1); rc=$?
[ $rc -eq 0 ] && grep -q "j6 done on GPU6 by j6_vl.sh" "$J6_DONE_FILE"; check "done: J6_SELFTEST=done writes the done file" $? "rc=$rc $OUT"
if [ -e "$J6_DONE_FILE" ]; then
  OUT=$( ( exec 8> "$J6_LOCK_FILE"; flock -n 8; J6_GPU=6 timeout 60 bash "$S" < /dev/null 2>&1; echo "rc=$?" ) )
  echo "$OUT" | grep -q "^rc=0$" && grep -q "state=skipped reason=already_done" "$J6_STATUS_FILE"
  check "done: present -> rc 0 and a skipped line" $? "$OUT"
else
  check "done: present -> skip (not run: no done file)" 1 ""
fi
stop_fake

echo "J6 SELFTEST: $pass passed, $fail failed (script $S sha256 $(sha256sum "$S" | cut -c1-64); scratch $TD)"
[ $fail -eq 0 ]
