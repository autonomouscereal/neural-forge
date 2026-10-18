#!/usr/bin/env bash
# test_d1_seal_queue.sh -- d1_seal_queue.sh's modes and gates on throwaway queue / run / seal dirs under declared scratch
# and a fake nvidia-smi.  Lane d1seal, Claude, 2026-10-14.  Nothing here can load a model, take a real card lock or
# write a real receipt: every armed phase-A case stops at A0 or A1 (before any tool runs); the 1.2.0 phase-B cases run
# a stub review runner, stop before B4 (D1SEAL_STOP_AFTER_REVIEW) or at B5's dry run against a refusing stub ledger.
#   bash test_d1_seal_queue.sh [script under test, default $Q/d1_seal_queue.sh]
#   live-dry      the real paths, --dry-run: exit 0, "DRY RUN", the queue dir / status / seal dir unchanged
#   gate          --gate-once on a fixture run: running 10, succeeded-not-finalized 10, finalized 0, alert 2
#   down          armed on an alerted D1: exit 2, "down for good", no card lock file made, no step marker
#   locks         armed on a finished D1 with gpu0.lock held elsewhere: waits, then exit 4 at the cap, no A2 marker
#   single        a second instance (d1_seal.lock held): exit 5
#   cards         --cards-probe: free -> s8; gpu6.lock held -> p40; a live toolcall claim / a busy P40 -> none
#   usage         an unknown argument -> 64
#   retry         (1.2.0) phase B, stub runner: an orphan dossier is moved aside; attempt 1 fails, attempt 2 passes
#   fallback      (1.2.0) every attempt fails: 3 attempts, dossiers kept aside, the clone --notes name the real cause
#   window        (1.2.0) no cards in a 0 s window: cause = no cards, no review ran
#   b3only        (1.2.0) B2 done, B3 pending, no cards: B3 skipped at once (no 72 h wait, no unreviewed fallback)
#   lanes         (1.0.1) armed on an alerted D1 against a temp LANES.json: L-D1 heartbeated (source = VERSION,
#                 status STOPPED); no registry -> silent; a corrupt registry -> still exit 2 (never fatal)
S=${1:-/home/cereal/omni-campaign/distill/queues/d1_seal_queue.sh}
TD=/mnt/lexar/omni-campaign/scratch/d1seal-test-$$
mkdir -p "$TD/q" "$TD/run" "$TD/alloc"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }
cat > "$TD/nvsmi" <<'EOF'
#!/bin/bash
case "$*" in *query-compute-apps*) exit 0 ;; esac
for kv in ${FAKE_USED//,/ }; do echo "${kv%%:*}, ${kv##*:}"; done
EOF
chmod +x "$TD/nvsmi"
export FAKE_USED="0:5,1:5,2:5,3:5,4:5,5:5,6:5"
fx() {  # fx <status json>
  echo '{"launch_id": "L1", "unit": "x.service"}' > "$TD/run/launch.json"
  echo "$1" > "$TD/run/runtime-status.json"
}
DONE='{"launch_id": "L1", "attempt": 1, "completed": true, "state": "succeeded", "exit_code": 0, "systemd": {"service_result": "success"}}'

# live-dry -----------------------------------------------------------------------------------------------------
Q=/home/cereal/omni-campaign/distill/queues
before=$(ls -la --time-style=full-iso "$Q" | md5sum); sb=$(md5sum < "$Q/d1_seal.status" 2>/dev/null)
OUT=$(bash "$S" --dry-run 2>&1); rc=$?
after=$(ls -la --time-style=full-iso "$Q" | md5sum); sa=$(md5sum < "$Q/d1_seal.status" 2>/dev/null)
[ $rc -eq 0 ] && echo "$OUT" | grep -q "DRY RUN: nothing was written" && echo "$OUT" | grep -q "step B5-clone"
check "live-dry: exit 0, plan with every step" $? "rc=$rc $(echo "$OUT" | tail -3)"
[ "$before" = "$after" ] && [ "$sb" = "$sa" ] && [ ! -e /mnt/lexar/omni-campaign/distill/d1/seal-OMNI1D1-hardened-claude-20261014 ]
check "live-dry: the queue dir, the status file and the seal dir are untouched" $? "queue dir changed or seal dir created"
echo "$OUT" | grep -E "D1 unit|review cards now|step A2-sync" | sed 's/^/      /'

export D1SEAL_Q=$TD/q D1SEAL_STATUS=$TD/q/d1_seal.status D1SEAL_SEAL=$TD/seal D1SEAL_RV=$TD/rv D1SEAL_ALLOC=$TD/alloc \
       D1SEAL_UNIT=d1seal-test-no-such-unit D1SEAL_NVSMI=$TD/nvsmi D1SEAL_RUN=$TD/run D1SEAL_POLL_S=1 D1SEAL_LOCK_POLL_S=1

# gate ---------------------------------------------------------------------------------------------------------
fx '{"launch_id": "L1", "completed": false, "state": "running"}'
bash "$S" --gate-once > /dev/null 2>&1; check "gate: running -> 10" $([ $? -eq 10 ]; echo $?) "rc"
fx '{"launch_id": "L1", "completed": true, "state": "succeeded", "exit_code": 0}'
bash "$S" --gate-once > /dev/null 2>&1; check "gate: succeeded but not finalized -> 10" $([ $? -eq 10 ]; echo $?) "rc"
fx "$DONE"
bash "$S" --gate-once > /dev/null 2>&1; check "gate: finalized success -> 0" $? "rc"
fx '{"launch_id": "L1", "completed": true, "state": "refused", "exit_code": 3, "alert": "refused (exit 3)"}'
bash "$S" --gate-once > /dev/null 2>&1; check "gate: alert -> 2" $([ $? -eq 2 ]; echo $?) "rc"

# down ---------------------------------------------------------------------------------------------------------
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 2 ] && grep -q "D1 is down for good" "$TD/q/d1_seal.status"
check "down: an alerted D1 is never sealed (exit 2)" $? "rc=$rc $(tail -2 "$TD/q/d1_seal.status" 2>/dev/null)"
[ ! -e "$TD/q/gpu0.lock" ] && [ ! -e "$TD/q/gpu4.lock" ] && [ -z "$(ls "$TD/seal/steps" 2>/dev/null)" ]
check "down: no card lock opened, no step marker" $? "$(ls "$TD/q" "$TD/seal/steps" 2>&1 | tr '\n' ' ')"

# locks --------------------------------------------------------------------------------------------------------
fx "$DONE"
flock "$TD/q/gpu0.lock" sleep 30 & H=$!; sleep 1
OUT=$(D1SEAL_LOCK_WAIT_MAX_S=3 timeout 60 bash "$S" 2>&1); rc=$?
kill $H 2>/dev/null; wait $H 2>/dev/null
[ $rc -eq 4 ] && grep -q "step=A1-locks state=waiting" "$TD/q/d1_seal.status" && grep -q "locks not obtained" "$TD/q/d1_seal.status"
check "locks: waits for gpu0.lock, gives up at the cap (exit 4)" $? "rc=$rc $(tail -3 "$TD/q/d1_seal.status")"
[ ! -e "$TD/seal/steps/A2-sync.done" ] && ! grep -q "step=A2" "$TD/q/d1_seal.status"
check "locks: nothing past A1 ran" $? "$(ls "$TD/seal/steps" 2>&1)"

# single -------------------------------------------------------------------------------------------------------
flock "$TD/q/d1_seal.lock" sleep 20 & H=$!; sleep 1
OUT=$(timeout 30 bash "$S" 2>&1); rc=$?
kill $H 2>/dev/null; wait $H 2>/dev/null
[ $rc -eq 5 ] && echo "$OUT" | grep -q "another d1_seal_queue"
check "single: a second instance exits 5" $? "rc=$rc $OUT"

# cards --------------------------------------------------------------------------------------------------------
flock -w 60 "$TD/q/gpu0.lock" true   # (1.0.1) locks' "sleep 30" child holds gpu0.lock after its flock parent is killed
: > "$TD/q/p40.claims"
[ "$(bash "$S" --cards-probe 2>/dev/null)" = s8 ]; check "cards: all free -> s8 (S8's own cards first)" $? "$(bash "$S" --cards-probe 2>&1)"
flock "$TD/q/gpu6.lock" sleep 20 & H=$!; sleep 1
[ "$(bash "$S" --cards-probe 2>/dev/null)" = p40 ]; check "cards: gpu6.lock held -> p40" $? "$(bash "$S" --cards-probe 2>&1)"
echo "2026-10-14T14:28:20Z CLAIM GPU1,GPU2 owner=claude lane=toolcall what=x" >> "$TD/q/p40.claims"
OUT=$(bash "$S" --cards-probe 2>&1); rc=$?
[ $rc -ne 0 ] && echo "$OUT" | grep -q "^none:.*p40.claims"; check "cards: a live toolcall claim -> none" $? "$OUT"
echo "2026-10-15T04:00:00Z RELEASE GPU1,GPU2 owner=claude lane=toolcall (done)" >> "$TD/q/p40.claims"
OUT=$(FAKE_USED="0:5,1:5,2:18071,3:5,4:5,5:5,6:5" bash "$S" --cards-probe 2>&1); rc=$?
[ $rc -ne 0 ] && echo "$OUT" | grep -q "GPU2=18071MiB"; check "cards: a busy P40 -> none" $? "$OUT"
kill $H 2>/dev/null; wait $H 2>/dev/null

# usage --------------------------------------------------------------------------------------------------------
bash "$S" --bogus > /dev/null 2>&1; check "usage: unknown argument -> 64" $([ $? -eq 64 ]; echo $?) "rc"

# review retry / fallback (1.2.0, d1seal review) ---------------------------------------------------------------
# phase B on a fake d1.sealed with a STUB review runner (D1SEAL_REVIEW_STUB) and a stub ledger that refuses
# (D1SEAL_LEDGER): nothing loads a model, B4 never runs (it would write to $OC/calibration), B5 stops at its dry run.
flock -w 60 "$TD/q/d1_seal.lock" true   # single's "sleep 20" child still holds it after its flock parent is killed
echo "2026-10-15T20:00:00Z D1 sealed (test)" > "$TD/q/d1.sealed"
cat > "$TD/ledger_stub.py" <<'EOF'
import sys
print("LEDGER STUB", sys.argv[1:])
sys.exit(1)
EOF
FAILSTUB='echo partial > "$1"; echo "{}" > "$1.finite_guard.json"; echo "torch.OutOfMemoryError: CUDA out of memory (stub)"; exit 1'
OKSTUB='c=$(cat calls 2>/dev/null || echo 0); echo $((c + 1)) > calls; if [ "$2" = OC02-OMNI1D1-NATIVE-NOTHINK ] && [ ! -e failed_once ]; then : > failed_once; echo partial > "$1"; echo "OOM (stub)"; exit 1; fi; echo "{}" > "$1"; echo "{}" > "$1.finite_guard.json"'
export D1SEAL_REVIEW_RETRY_S=1 D1SEAL_REVIEW_RAM_GIB=0 D1SEAL_LEDGER=$TD/ledger_stub.py
for g in 0 4 6; do flock -w 60 "$TD/q/gpu$g.lock" true; done   # cards' "sleep 20" children may still hold gpu6.lock
# retry: an orphan dossier from a killed queue + one failed attempt, then success on re-chosen cards
mkdir -p "$TD/rv"; echo stale > "$TD/rv/oc-dossier-OC02-OMNI1D1-NATIVE-NOTHINK.json"
OUT=$(D1SEAL_REVIEW_STUB="$OKSTUB" D1SEAL_STOP_AFTER_REVIEW=1 timeout 120 bash "$S" --phase b 2>&1); rc=$?
ST=$(grep -E "step=B|test_stop" "$TD/q/d1_seal.status")
[ $rc -eq 0 ] && echo "$ST" | grep -q "step=B2-nothink event=moved_aside kind=orphan" \
  && echo "$ST" | grep -q "step=B2-nothink state=failed rc=1 attempt=1/3 placement=s8" \
  && echo "$ST" | grep -q "step=B1-cards state=retrying attempt=2/3" && echo "$ST" | grep -q "step=B1-cards state=held attempt=2/3" \
  && [ -f "$TD/seal/steps/B2-nothink.done" ] && [ -f "$TD/seal/steps/B3-template.done" ] && [ ! -e "$TD/seal/review_pending.cause" ] \
  && [ "$(cat "$TD/seal/calls")" = 3 ] && [ "$(ls "$TD/rv" | grep -c 'NOTHINK.json.orphan-')" = 1 ] \
  && [ "$(ls "$TD/rv" | grep -c 'NOTHINK.json.failed-')" = 1 ] && [ "$(cat "$TD/rv/oc-dossier-OC02-OMNI1D1-NATIVE-NOTHINK.json")" = "{}" ] \
  && ls "$TD/seal/logs" | grep -q "B2-nothink.failed-.*log" && ! grep -q "step=B5" "$TD/q/d1_seal.status"
check "retry: an orphan dossier is moved aside, a failed review is retried on re-chosen cards and passes; no unreviewed clone" $? "rc=$rc $(echo "$ST" | tail -6) rv: $(ls "$TD/rv" | tr '\n' ' ')"
# fallback: every attempt fails -> 3 attempts, then the clone's notes carry the real cause
: > "$TD/q/d1_seal.status"
OUT=$(D1SEAL_SEAL=$TD/seal2 D1SEAL_RV=$TD/rv2 D1SEAL_REVIEW_STUB="$FAILSTUB" timeout 120 bash "$S" --phase b 2>&1); rc=$?
ST=$(cat "$TD/q/d1_seal.status")
[ $rc -eq 3 ] && [ "$(echo "$ST" | grep -c 'step=B2-nothink state=failed rc=1')" = 3 ] && echo "$ST" | grep -q 'state=gave_up cause="the bank v1 review failed 3 time(s)' \
  && [ "$(ls "$TD/rv2" | grep -c 'NOTHINK.json.failed-.*[0-9]Z$')" = 3 ] && [ "$(ls "$TD/rv2" | grep -c 'finite_guard.json$')" = 3 ] \
  && [ ! -e "$TD/rv2/oc-dossier-OC02-OMNI1D1-NATIVE-NOTHINK.json" ] && grep -q "failed 3 time(s)" "$TD/seal2/review_pending.cause" \
  && grep -q -- "--allow-unreviewed-source" "$TD/seal2/logs/B5-clone.log" && grep -q "UNREVIEWED SOURCE.*the bank v1 review failed 3 time(s).*CUDA out of memory" "$TD/seal2/logs/B5-clone.log" \
  && [ ! -e "$TD/seal2/logs/B4-record.log" ] && echo "$ST" | grep -q 'state=stopped step=B5-clone reason="clone DRY RUN refused"'
check "fallback: 3 failed attempts (dossiers kept aside), then the unreviewed clone names the real cause in --notes" $? "rc=$rc $(echo "$ST" | grep -E 'gave_up|stopped|failed' | tail -4 | cut -c1-300) rv2: $(ls "$TD/rv2" | wc -l)"
# window: no cards at all and a 0 s window -> the cause is the missing cards, nothing ran
: > "$TD/q/d1_seal.status"
OUT=$(FAKE_USED="0:9000,1:5,2:9000,3:5,4:5,5:5,6:5" D1SEAL_SEAL=$TD/seal3 D1SEAL_RV=$TD/rv3 D1SEAL_REVIEW_WAIT_MAX_S=0 D1SEAL_REVIEW_STUB="$FAILSTUB" timeout 120 bash "$S" --phase b 2>&1); rc=$?
[ $rc -eq 3 ] && grep -q 'state=gave_up cause="no three free review cards' "$TD/q/d1_seal.status" && ! grep -q "step=B2-nothink state=" "$TD/q/d1_seal.status" \
  && grep -q "UNREVIEWED SOURCE.*no three free review cards" "$TD/seal3/logs/B5-clone.log"
check "window: no cards inside the window -> the notes say so, no review ran" $? "rc=$rc $(tail -3 "$TD/q/d1_seal.status" | cut -c1-300)"
# b3only: B2 done (the review of record), B3 pending, no cards -> one card check, no wait, B4 is reached (stopped by the hook)
: > "$TD/q/d1_seal.status"; mkdir -p "$TD/seal4/steps"; echo "2026-10-16T00:00:00Z abc placement=s8" > "$TD/seal4/steps/B2-nothink.done"
t0=$(date +%s)
OUT=$(FAKE_USED="0:9000,1:5,2:9000,3:5,4:5,5:5,6:5" D1SEAL_SEAL=$TD/seal4 D1SEAL_RV=$TD/rv4 D1SEAL_REVIEW_STUB="$FAILSTUB" D1SEAL_STOP_AFTER_REVIEW=1 timeout 120 bash "$S" --phase b 2>&1); rc=$?
[ $rc -eq 0 ] && [ $(( $(date +%s) - t0 )) -lt 60 ] && grep -q "step=B3-template state=skipped_nonblocking" "$TD/q/d1_seal.status" \
  && ! grep -q "gave_up" "$TD/q/d1_seal.status" && [ ! -e "$TD/seal4/review_pending.cause" ]
check "b3only: B2 done + no cards -> B3 skipped at once, no unreviewed fallback" $? "rc=$rc $(tail -3 "$TD/q/d1_seal.status" | cut -c1-300)"
unset D1SEAL_REVIEW_RETRY_S D1SEAL_REVIEW_RAM_GIB D1SEAL_LEDGER
rm -f "$TD/q/d1.sealed"

# lanes (1.0.1, card X-T2) -------------------------------------------------------------------------------------
fx '{"launch_id": "L1", "completed": true, "state": "refused", "exit_code": 3, "alert": "refused (exit 3)"}'
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 2 ] && ! echo "$OUT" | grep -q "lane heartbeat"; check "lanes: no registry -> silent, still exit 2" $? "rc=$rc"
LP=${LANES_PY:-/home/cereal/omni-campaign/distill/tools/lanes.py}
echo '{"created_by": "test_d1_seal_queue", "lanes": [{"lane": "L-D1", "owner": "claude", "scope": "t", "lock_paths": [],
 "status": "s", "transferable": false, "transferable_reason": "t"}]}' > "$TD/seed.json"
python3 "$LP" --file "$TD/q/LANES.json" init --seed "$TD/seed.json" > /dev/null
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
SV=$(sed -n 's/^VERSION=//p' "$S")
GOT=$(python3 -c 'import json, sys
d = json.load(open(sys.argv[1]))["lanes"]["L-D1"]
print(d["heartbeat_source"], "|", d["status"].split(" (")[0], "|", d["detail"][:13], "|", json.load(open(sys.argv[1]))["revision"])' "$TD/q/LANES.json" 2>&1)
[ $rc -eq 2 ] && [ "$GOT" = "$SV | D1 seal STOPPED, see detail and $TD/q/d1_seal.status | state=stopped | 3" ]
check "lanes: armed lines heartbeat L-D1 (source = VERSION, armed then STOPPED)" $? "rc=$rc got: $GOT"
echo "not json" > "$TD/q/LANES.json"
OUT=$(timeout 60 bash "$S" 2>&1); rc=$?
[ $rc -eq 2 ] && echo "$OUT" | grep -q "lane heartbeat L-D1 failed (ignored)"; check "lanes: a corrupt registry never changes the outcome" $? "rc=$rc"

rm -rf "$TD"
echo "D1 SEAL QUEUE SELFTEST: $pass passed, $fail failed"
[ $fail -eq 0 ]
