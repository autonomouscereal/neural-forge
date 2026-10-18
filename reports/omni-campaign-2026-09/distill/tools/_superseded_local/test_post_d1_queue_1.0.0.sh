#!/usr/bin/env bash
# test_post_d1_queue.sh -- post_d1_queue.sh's modes and job bookkeeping on a throwaway queue dir under declared
# scratch.  Lane d1seal, Claude, 2026-10-14.  Only the h3 job (which never launches anything) and jobs whose inputs
# are absent run armed, so no case can reach a GPU.
#   bash test_post_d1_queue.sh [script under test, default $Q/post_d1_queue.sh]
#   live-dry    the real paths, --dry-run: exit 0, "DRY RUN", the queue dir and status file unchanged
#   ready       --dry-run with a fixture corpus whose sidecar names its sha256 -> "ready"; a wrong sha -> NOT-READY
#   usage       an unknown job -> 64
#   wait        armed without d1.sealed: waits (status line), nothing runs
#   h3          armed h3 with d1.sealed: receipt rc 0 + "skipped" (no proven 3090 path); a re-run skips it as done
#   defer       armed c2 with no corpus: deferred, then given up at the cap with a receipt rc 75
#   single      a second instance (post_d1.lock held): exit 5
S=${1:-/home/cereal/omni-campaign/distill/queues/post_d1_queue.sh}
TD=/mnt/lexar/omni-campaign/scratch/postd1-test-$$
mkdir -p "$TD/q" "$TD/alloc" "$TD/in"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }

# live-dry -----------------------------------------------------------------------------------------------------
Q=/home/cereal/omni-campaign/distill/queues
before=$(ls -la --time-style=full-iso "$Q" | md5sum)
OUT=$(bash "$S" --dry-run 2>&1); rc=$?
after=$(ls -la --time-style=full-iso "$Q" | md5sum)
[ $rc -eq 0 ] && echo "$OUT" | grep -q "DRY RUN: nothing was written" && echo "$OUT" | grep -q "^.* p2: WANT_T="
check "live-dry: exit 0 with every job's command" $? "rc=$rc $(echo "$OUT" | tail -3)"
[ "$before" = "$after" ] && [ ! -e "$Q/post_d1" ]; check "live-dry: the queue dir is untouched" $? "changed"
echo "$OUT" | grep -E "start gate|c1:|c2:|h3:" | sed 's/^/      /'

export POSTD1_Q=$TD/q POSTD1_STATUS=$TD/q/post_d1.status POSTD1_ALLOC=$TD/alloc POSTD1_POLL_S=1 POSTD1_DEFER_WAIT_MAX_S=2 \
       POSTD1_C1_CORPUS=$TD/in/c1.safetensors POSTD1_C2_CORPUS=$TD/in/c2.safetensors

# ready --------------------------------------------------------------------------------------------------------
head -c 4096 /dev/urandom > "$TD/in/c2.safetensors"
echo "{\"corpus\": {\"sha256\": \"$(sha256sum "$TD/in/c2.safetensors" | cut -c1-64)\"}}" > "$TD/in/c2.rows.json"
bash "$S" --dry-run c2 2>&1 | grep -q "c2: corpus .* ready"; check "ready: a sidecar naming the corpus sha -> ready" $? "$(bash "$S" --dry-run c2 2>&1 | grep c2:)"
echo '{"corpus": {"sha256": "0000"}}' > "$TD/in/c2.rows.json"
bash "$S" --dry-run c2 2>&1 | grep -q "c2: corpus .* NOT-READY"; check "ready: a wrong sha -> NOT-READY" $? "$(bash "$S" --dry-run c2 2>&1 | grep c2:)"
rm -f "$TD/in/c2.safetensors" "$TD/in/c2.rows.json"

# usage --------------------------------------------------------------------------------------------------------
bash "$S" bogus > /dev/null 2>&1; check "usage: unknown job -> 64" $([ $? -eq 64 ]; echo $?) "rc"

# wait ---------------------------------------------------------------------------------------------------------
timeout 4 bash "$S" h3 > /dev/null 2>&1; rc=$?
[ $rc -eq 124 ] && grep -q "state=waiting reason=no_d1.sealed" "$TD/q/post_d1.status" && [ ! -e "$TD/q/post_d1/h3.json" ]
check "wait: no d1.sealed -> waits, runs nothing" $? "rc=$rc $(tail -2 "$TD/q/post_d1.status")"

# h3 -----------------------------------------------------------------------------------------------------------
echo "2026-10-15T20:00:00Z D1 sealed (test)" > "$TD/q/d1.sealed"
timeout 60 bash "$S" h3 > /dev/null 2>&1; rc=$?
[ $rc -eq 0 ] && grep -q '"rc": 0' "$TD/q/post_d1/h3.json" && grep -q "job=h3 state=skipped reason=\"no proven 3090 H3 path" "$TD/q/post_d1.status"
check "h3: no proven 3090 path -> skipped with a receipt" $? "rc=$rc $(tail -2 "$TD/q/post_d1.status")"
timeout 60 bash "$S" h3 > /dev/null 2>&1
grep -q "job=h3 state=skipped reason=done_before" "$TD/q/post_d1.status"; check "h3: a re-run skips a done job" $? "$(tail -1 "$TD/q/post_d1.status")"

# defer --------------------------------------------------------------------------------------------------------
timeout 60 bash "$S" c2 > /dev/null 2>&1; rc=$?
[ $rc -eq 0 ] && grep -q "job=c2 state=deferred" "$TD/q/post_d1.status" && grep -q "job=c2 state=gave_up" "$TD/q/post_d1.status" \
  && grep -q '"rc": 75' "$TD/q/post_d1/c2.json"
check "defer: a missing input defers, then gives up with rc 75" $? "rc=$rc $(tail -3 "$TD/q/post_d1.status")"

# single -------------------------------------------------------------------------------------------------------
flock "$TD/q/post_d1.lock" sleep 20 & H=$!; sleep 1
OUT=$(timeout 30 bash "$S" h3 2>&1); rc=$?
kill $H 2>/dev/null; wait $H 2>/dev/null
[ $rc -eq 5 ]; check "single: a second instance exits 5" $? "rc=$rc $OUT"

rm -rf "$TD"
echo "POST D1 QUEUE SELFTEST: $pass passed, $fail failed"
[ $fail -eq 0 ]
