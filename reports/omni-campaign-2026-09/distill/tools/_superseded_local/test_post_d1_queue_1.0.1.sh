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
#   lanes       (1.0.1) --lane-of routing; no registry -> silent; armed h3 / c2 against a temp LANES.json: L-D1
#               heartbeated with source = VERSION and the h3 receipt + sha, L-TOOL untouched by a deferred c2;
#               a corrupt registry never stops the queue
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

# lanes (1.0.1, card X-T2) -------------------------------------------------------------------------------------
# routing; then armed h3 + c2 against a TEMP registry ($TD/q/LANES.json, never the live $Q/LANES.json)
flock -w 60 "$TD/q/post_d1.lock" true   # single's "sleep 20" child still holds it after its flock parent is killed
r=""; for l in "job=c1 state=started corpus=x" "job=c2 state=done rc=0 receipt=x" "job=c1 state=waiting reason=gpu0.lock_held" \
               "job=c2 state=deferred reason=input_not_ready" "job=c1 state=gave_up reason=x" "job=p2 state=started" \
               "job=j6 state=done" "state=end"; do r="$r $(bash "$S" --lane-of "$l")"; done
[ "$r" = " L-TOOL L-TOOL L-TOOL L-D1 L-D1 L-D1 L-D1 L-D1" ]
check "lanes: c1/c2 -> L-TOOL; deferred / gave_up and every other line -> L-D1" $? "got$r"
OUT=$(LANES_FILE=$TD/none/LANES.json timeout 60 bash "$S" h3 2>&1); rc=$?
[ $rc -eq 0 ] && ! echo "$OUT" | grep -q "lane heartbeat"; check "lanes: no registry -> silent, rc 0" $? "rc=$rc $(echo "$OUT" | grep 'lane heartbeat')"
LP=${LANES_PY:-/home/cereal/omni-campaign/distill/tools/lanes.py}
echo '{"created_by": "test_post_d1_queue", "lanes": [
 {"lane": "L-D1", "owner": "claude", "scope": "t", "lock_paths": [], "status": "s", "transferable": false, "transferable_reason": "t"},
 {"lane": "L-TOOL", "owner": "claude", "scope": "t", "lock_paths": [], "status": "s", "transferable": true, "transferable_reason": "t"}]}' > "$TD/seed.json"
python3 "$LP" --file "$TD/q/LANES.json" init --seed "$TD/seed.json" > /dev/null
rm -f "$TD/q/post_d1/h3.json" "$TD/q/post_d1/c2.json"
timeout 60 bash "$S" h3 > /dev/null 2>&1; rc1=$?
timeout 60 bash "$S" c2 > /dev/null 2>&1; rc2=$?
SV=$(sed -n 's/^VERSION=//p' "$S")
GOT=$(python3 -c 'import json, sys
l = json.load(open(sys.argv[1]))["lanes"]; d, t = l["L-D1"], l["L-TOOL"]; r = d.get("last_receipt") or {}
print(d["heartbeat_source"], d["status"].split(" (")[0], r.get("path"), r.get("sha256"), t["heartbeat_utc"])' "$TD/q/LANES.json" 2>&1)
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] \
  && [ "$GOT" = "$SV post-D1 queue ended $TD/q/post_d1/h3.json $(sha256sum "$TD/q/post_d1/h3.json" | cut -c1-64) None" ]
check "lanes: armed lines heartbeat L-D1 (source = VERSION, h3 receipt + sha); a deferred c2 leaves L-TOOL alone" $? "rc=$rc1/$rc2 got: $GOT"
echo "not json" > "$TD/q/LANES.json"
OUT=$(timeout 60 bash "$S" h3 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$OUT" | grep -q "lane heartbeat L-D1 failed (ignored)"; check "lanes: a corrupt registry never stops the queue" $? "rc=$rc"

rm -rf "$TD"
echo "POST D1 QUEUE SELFTEST: $pass passed, $fail failed"
[ $fail -eq 0 ]
