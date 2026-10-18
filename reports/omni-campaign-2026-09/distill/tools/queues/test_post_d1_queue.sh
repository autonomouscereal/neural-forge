#!/usr/bin/env bash
# test_post_d1_queue.sh -- post_d1_queue.sh's modes and job bookkeeping on a throwaway queue dir under declared
# scratch.  Lane d1seal, Claude, 2026-10-14.  Only the h3 job (which never launches anything) and jobs whose inputs
# are absent run armed, so no case can reach a GPU.
#   bash test_post_d1_queue.sh [script under test, default $Q/post_d1_queue.sh]
#   live-dry    the real paths, --dry-run: exit 0, "DRY RUN", the queue dir, its post_d1 receipt dir and status file
#               unchanged (1.4.0: compared, no longer required absent -- the armed queue creates $Q/post_d1)
#   ready       --dry-run with a fixture corpus whose sidecar names its sha256 -> "ready"; a wrong sha -> NOT-READY
#   usage       an unknown job -> 64
#   wait        armed without d1.sealed: waits (status line), nothing runs
#   h3          armed h3 with d1.sealed: receipt rc 0 + "skipped" (no proven 3090 path); a re-run skips it as done
#   defer       armed c2 with no corpus: deferred, then given up at the cap with a receipt rc 75
#   single      a second instance (post_d1.lock held): exit 5
#   lanes       (1.0.1) --lane-of routing; no registry -> silent; armed h3 / c2 against a temp LANES.json: L-D1
#               heartbeated with source = VERSION and the h3 receipt + sha, L-TOOL untouched by a deferred c2;
#               a corrupt registry never stops the queue
#   order       (1.1.0) the dry run lists the jobs as p2 c1 c2 j6 h3 p11
#   c2dir       (1.1.0) no override: the ONE bfcl-native-*.safetensors in POSTD1_C2_DIR whose rows.json names its
#               sha -> ready; a second file there -> NOT-READY (ambiguous); an empty dir -> NOT-READY
#   j6gate      (1.1.0) armed j6 while a "K3" process is live: deferred with state=waiting K3_process_live, given up at
#               the cap with rc 75, gpu0.lock never opened (no file made); the same with MemAvailable below the gate
#   rescoregate (1.3.0, d1seal review) a ready c2 with MemAvailable below POSTD1_RESCORE_RAM_GIB: deferred, gpu0.lock never
#               opened, one status line per reason, gave_up rc 75 at the cap; routed to L-D1; the dry run shows the gate
#   k3decide    (1.4.0, lane sealdecide) a stub k3_seal_decide.sh (POSTD1_K3D): gate-once closed -> deferred with one
#               waiting line, the armed script never run, gave_up rc 75 at the cap; gate open -> run with post_d1.lock's fd
#               closed, receipt rc 0 naming the decision, branch in the status line, skipped as done on a re-run; an armed
#               rc 75 -> waiting + deferred; a missing script -> the job fails alone (rc 1 receipt), the queue goes on
#   screen      (1.5.0, lane toolscreen) a stub toolcall_heldout_screen.py (POSTD1_SCREEN) and an unreachable rescore RAM
#               gate (nothing reaches build-text or gpu0.lock): a ready c1 screens FIRST (screening -> screened -> the
#               rescore gate's deferral, in that order) and only once across the deferral passes; the dry run shows
#               "needed" then "valid receipt"; a re-exported corpus is screened again; a refused screen fails c1 alone
#               (rc 1 receipt naming the screen log, no started line); a deferred screen (rc 75) defers c1; screening
#               lines heartbeat L-TOOL
#   p2verdict   (1.2.0, d1seal review) the armed p2 job with a stub runner (POSTD1_P2_RUN) and a fixture summary under
#               POSTD1_P2_OC: $Q/p2.verdict is written by d2_plan and READ BACK by d2_plan's own reader (the D2
#               consumer) as engine streamed / micro 8 / acc 1; a run with no completed cell writes no verdict
S=${1:-/home/cereal/omni-campaign/distill/queues/post_d1_queue.sh}
TD=/mnt/lexar/omni-campaign/scratch/postd1-test-$$
mkdir -p "$TD/q" "$TD/alloc" "$TD/in"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }

# live-dry -----------------------------------------------------------------------------------------------------
Q=/home/cereal/omni-campaign/distill/queues
# 1.4.0: the receipt dir $Q/post_d1 is compared, not required absent -- the ARMED queue creates it (1.3.0's check
# `[ ! -e $Q/post_d1 ]` failed on every run once post_d1 was armed on 2026-10-15)
before=$(ls -la --time-style=full-iso "$Q" | md5sum; ls -la --time-style=full-iso "$Q/post_d1" 2>&1 | md5sum)
OUT=$(bash "$S" --dry-run 2>&1); rc=$?
after=$(ls -la --time-style=full-iso "$Q" | md5sum; ls -la --time-style=full-iso "$Q/post_d1" 2>&1 | md5sum)
[ $rc -eq 0 ] && echo "$OUT" | grep -q "DRY RUN: nothing was written" && echo "$OUT" | grep -q "^.* p2: WANT_T="
check "live-dry: exit 0 with every job's command" $? "rc=$rc $(echo "$OUT" | tail -3)"
[ "$before" = "$after" ]; check "live-dry: the queue dir and its post_d1 receipt dir are untouched" $? "changed"
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

# order (1.1.0) ------------------------------------------------------------------------------------------------
JL=$(bash "$S" --dry-run 2>&1 | grep -o "job [a-z0-9]*:" | awk '{print $2}' | tr -d ':' | tr '\n' ' ')
[ "$JL" = "p2 k3decide c1 c2 j6 h3 p11 " ]; check "order: p2 k3decide c1 c2 j6 h3 p11" $? "got '$JL'"

# k3decide (1.4.0) ---------------------------------------------------------------------------------------------
# the stub's version line is printed, never written as an assignment here: tool_pins.py reads the first VERSION= line
# of THIS file as the test's own version
printf '#!/bin/bash\n%s="k3_seal_decide.sh 9.9.9-stub"\n' VERSION > "$TD/k3d.sh"
cat >> "$TD/k3d.sh" <<'EOF'
if [ "${1:-}" = --gate-once ]; then [ -n "${STUB_GATE:-}" ] && { echo "$STUB_GATE"; exit 75; }; echo ready; exit 0; fi
echo "armed fd9=$([ -e /proc/$$/fd/9 ] && echo open || echo closed)" >> "$STUB_CALLS"
[ "${STUB_RC:-0}" = 75 ] && { echo "[x] STATUS deferred: stub inner gate"; exit 75; }
echo '{"branch": "k3-first", "applied": true}' > "$POSTD1_Q/k3_seal_decision.json"; exit 0
EOF
export STUB_CALLS=$TD/k3d.calls
rm -f "$TD/q/post_d1/k3decide.json" "$STUB_CALLS"
( export POSTD1_K3D=$TD/k3d.sh STUB_GATE="no_d1.sealed D1_unit_active"; timeout 60 bash "$S" k3decide > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && [ "$(grep -c 'job=k3decide state=waiting reason="no_d1.sealed D1_unit_active" (nothing written)' "$TD/q/post_d1.status")" = 1 ] \
  && grep -q "job=k3decide state=gave_up" "$TD/q/post_d1.status" && grep -q '"rc": 75' "$TD/q/post_d1/k3decide.json" && [ ! -e "$STUB_CALLS" ]
check "k3decide: gate-once closed -> one waiting line, the script never run armed, gave_up rc 75" $? "rc=$rc $(grep k3decide "$TD/q/post_d1.status" | tail -3) $(cat "$STUB_CALLS" 2>/dev/null)"
rm -f "$TD/q/post_d1/k3decide.json"
( export POSTD1_K3D=$TD/k3d.sh; timeout 60 bash "$S" k3decide > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && grep -q '"rc": 0' "$TD/q/post_d1/k3decide.json" && grep -q "k3_seal_decision.json" "$TD/q/post_d1/k3decide.json" \
  && grep -q "job=k3decide state=done rc=0 branch=k3-first receipt=" "$TD/q/post_d1.status" && grep -qx "armed fd9=closed" "$STUB_CALLS" \
  && grep -q "job=k3decide state=started script=.* k3_seal_decide.sh 9.9.9-stub" "$TD/q/post_d1.status"
check "k3decide: gate open -> run (post_d1.lock fd closed), receipt names the decision, branch in the status line" $? "rc=$rc $(grep k3decide "$TD/q/post_d1.status" | tail -2) $(cat "$STUB_CALLS" 2>/dev/null)"
( export POSTD1_K3D=$TD/k3d.sh; timeout 60 bash "$S" k3decide > /dev/null 2>&1 )
grep -q "job=k3decide state=skipped reason=done_before" "$TD/q/post_d1.status" && [ "$(grep -c armed "$STUB_CALLS")" = 1 ]
check "k3decide: a re-run skips the done job" $? "$(tail -1 "$TD/q/post_d1.status")"
rm -f "$TD/q/post_d1/k3decide.json" "$STUB_CALLS"
( export POSTD1_K3D=$TD/k3d.sh STUB_RC=75; timeout 60 bash "$S" k3decide > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && grep -q 'job=k3decide state=waiting reason=".*deferred: stub inner gate"' "$TD/q/post_d1.status" && grep -q '"rc": 75' "$TD/q/post_d1/k3decide.json"
check "k3decide: an armed rc 75 -> waiting, deferred, gave_up rc 75 at the cap" $? "rc=$rc $(grep k3decide "$TD/q/post_d1.status" | tail -3)"
rm -f "$TD/q/post_d1/k3decide.json"
( export POSTD1_K3D=$TD/none.sh; timeout 60 bash "$S" k3decide h3 > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && grep -q '"rc": 1' "$TD/q/post_d1/k3decide.json" && grep -q 'job=k3decide state=failed reason=".*none.sh missing' "$TD/q/post_d1.status" \
  && grep -q "job=k3decide state=failed rc=1 (the queue continues)" "$TD/q/post_d1.status"
check "k3decide: a missing script fails the job alone (rc 1 receipt), the queue goes on" $? "rc=$rc $(grep k3decide "$TD/q/post_d1.status" | tail -2)"
OUT=$(POSTD1_K3D=$TD/k3d.sh bash "$S" --dry-run k3decide 2>&1 | grep "k3decide:")
echo "$OUT" | grep -q "k3_seal_decide.sh 9.9.9-stub .* gate-once now: ready"; check "k3decide: the dry run shows the script and its gate" $? "$OUT"
rm -f "$TD/q/post_d1/k3decide.json" "$TD/q/k3_seal_decision.json" "$STUB_CALLS"

# c2dir (1.1.0) ------------------------------------------------------------------------------------------------
mkdir -p "$TD/bn"
head -c 2048 /dev/urandom > "$TD/bn/bfcl-native-500.safetensors"
echo "{\"output\": {\"sha256\": \"$(sha256sum "$TD/bn/bfcl-native-500.safetensors" | cut -c1-64)\"}}" > "$TD/bn/rows.json"
OUT=$(POSTD1_C2_CORPUS= POSTD1_C2_DIR=$TD/bn bash "$S" --dry-run c2 2>&1 | grep "c2:")
echo "$OUT" | grep -q "bfcl-native-500.safetensors ready"; check "c2dir: the single bfcl-native file named by rows.json -> ready" $? "$OUT"
cp "$TD/bn/bfcl-native-500.safetensors" "$TD/bn/bfcl-native-506.safetensors"
OUT=$(POSTD1_C2_CORPUS= POSTD1_C2_DIR=$TD/bn bash "$S" --dry-run c2 2>&1 | grep "c2:")
echo "$OUT" | grep -q "NOT-READY"; check "c2dir: two candidate files -> NOT-READY" $? "$OUT"
rm -f "$TD/bn"/*
OUT=$(POSTD1_C2_CORPUS= POSTD1_C2_DIR=$TD/bn bash "$S" --dry-run c2 2>&1 | grep "c2:")
echo "$OUT" | grep -q "NOT-READY"; check "c2dir: an empty dir -> NOT-READY" $? "$OUT"

# j6gate (1.1.0) -----------------------------------------------------------------------------------------------
sleep 31337 & K=$!
rm -f "$TD/q/post_d1/j6.json" "$TD/q/gpu0.lock"
# the pattern goes in through export, never on a command line: pgrep -f would match `env VAR=...` / `timeout ...` itself
( export POSTD1_K3_PGREP="sleep 31337"; timeout 60 bash "$S" j6 > /dev/null 2>&1 ); rc=$?
kill $K 2>/dev/null; wait $K 2>/dev/null
[ $rc -eq 0 ] && grep -q 'job=j6 state=waiting reason="K3_process_live' "$TD/q/post_d1.status" \
  && grep -q "job=j6 state=gave_up" "$TD/q/post_d1.status" && grep -q '"rc": 75' "$TD/q/post_d1/j6.json" && [ ! -e "$TD/q/gpu0.lock" ]
check "j6gate: a live K3 process defers j6 without opening gpu0.lock" $? "rc=$rc $(grep 'job=j6' "$TD/q/post_d1.status" | tail -2) $(ls "$TD/q")"
OUT=$(export POSTD1_K3_PGREP="sleep 31337"; bash "$S" --dry-run j6 2>&1 | grep "j6:")
! echo "$OUT" | grep -q "K3_process_live"; check "j6gate: control -- with the sleep gone the K3 gate is clear (the pattern matched the process, not a command line)" $? "$OUT"
rm -f "$TD/q/post_d1/j6.json"
( export POSTD1_K3_PGREP="no-such-process-$$" POSTD1_J6_RAM_GIB=100000; timeout 60 bash "$S" j6 > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && grep -q 'job=j6 state=waiting reason="MemAvailable_' "$TD/q/post_d1.status" && [ ! -e "$TD/q/gpu0.lock" ]
check "j6gate: MemAvailable below the gate defers j6 the same way" $? "rc=$rc $(grep 'job=j6' "$TD/q/post_d1.status" | tail -2)"
OUT=$(export POSTD1_K3_PGREP="no-such-process-$$" POSTD1_J6_RAM_GIB=0; bash "$S" --dry-run j6 2>&1 | grep "j6:")
echo "$OUT" | grep -q "pre-gate now: clear"; check "j6gate: the dry run shows a clear pre-gate" $? "$OUT"

# rescoregate (1.3.0) ------------------------------------------------------------------------------------------
# a READY c2 corpus with MemAvailable below the build-text gate: deferred without opening gpu0.lock, given up at the
# cap with rc 75 (1.2.x took the lock, waited 6 h and failed for good); nothing reaches qtext_teacher_cache
head -c 4096 /dev/urandom > "$TD/in/c2.safetensors"
echo "{\"corpus\": {\"sha256\": \"$(sha256sum "$TD/in/c2.safetensors" | cut -c1-64)\"}}" > "$TD/in/c2.rows.json"
rm -f "$TD/q/post_d1/c2.json" "$TD/q/gpu0.lock" "$TD/q/post_d1/c2.log"
( export POSTD1_RESCORE_RAM_GIB=100000; timeout 60 bash "$S" c2 > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && ! grep -q 'job=c2 state=started' "$TD/q/post_d1.status"   && grep -q 'job=c2 state=deferred reason="MemAvailable_[0-9]*GiB_lt_100000" (gpu0.lock not taken)' "$TD/q/post_d1.status"   && grep -q "job=c2 state=gave_up reason=input_or_gate_not_ready" "$TD/q/post_d1.status" && grep -q '"rc": 75' "$TD/q/post_d1/c2.json"   && [ ! -e "$TD/q/gpu0.lock" ] && ! grep -q "attempt" "$TD/q/post_d1/c2.log" 2>/dev/null   && [ "$(grep -c 'job=c2 state=deferred reason="MemAvailable' "$TD/q/post_d1.status")" = 1 ]
check "rescoregate: RAM below the gate defers a ready c2 before any started line, without opening gpu0.lock (one line per reason), rc 75 at the cap" $? "rc=$rc $(grep 'job=c2' "$TD/q/post_d1.status" | tail -3) $(ls "$TD/q")"
[ "$(bash "$S" --lane-of 'job=c2 state=deferred reason="MemAvailable_60GiB_lt_70" (gpu0.lock not taken)')" = L-D1 ]
check "rescoregate: a gate-deferred c2 line heartbeats L-D1, never L-TOOL" $? "$(bash "$S" --lane-of 'job=c2 state=deferred reason=x')"
OUT=$(POSTD1_RESCORE_RAM_GIB=100000 bash "$S" --dry-run c2 2>&1 | grep "rescore pre-gate"); OUT2=$(POSTD1_RESCORE_RAM_GIB=0 bash "$S" --dry-run c2 2>&1 | grep "rescore pre-gate")
echo "$OUT" | grep -q "MemAvailable_" && echo "$OUT2" | grep -q "now (1.3.0, checked before gpu0.lock): clear"
check "rescoregate: the dry run shows the pre-gate (blocked / clear)" $? "$OUT | $OUT2"
rm -f "$TD/in/c2.safetensors" "$TD/in/c2.rows.json" "$TD/q/post_d1/c2.json"

# screen (1.5.0) -----------------------------------------------------------------------------------------------
# the stub: `check` is valid only for the corpus bytes it screened; `screen` records the call and exits STUB_SCREEN_RC
cat > "$TD/screen_stub.py" <<'EOF'
import hashlib, os, sys
cmd, corpus = sys.argv[1], sys.argv[sys.argv.index("--corpus") + 1]
marker = corpus + ".screened"
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if cmd == "check":
    sys.exit(0 if os.path.exists(marker) and open(marker).read() == digest(corpus) else 10)
open(os.environ["STUB_SCREEN_CALLS"], "a").write("screen\n")
rc = int(os.environ.get("STUB_SCREEN_RC", "0"))
if rc == 0:
    open(marker, "w").write(digest(corpus))
    open(os.path.join(os.path.dirname(corpus), "heldout_screen.json"), "w").write('{"hits_in_export": 3}')
print({0: "[x] SCREENED stub", 75: "[x] DEFERRED: stub"}.get(rc, "[x] REFUSED: stub refusal"))
sys.exit(rc)
EOF
export STUB_SCREEN_CALLS=$TD/screen.calls
new_c1() { head -c 4096 /dev/urandom > "$TD/in/c1.safetensors"
  echo "{\"corpus\": {\"sha256\": \"$(sha256sum "$TD/in/c1.safetensors" | cut -c1-64)\"}}" > "$TD/in/c1.rows.json"; }
new_c1
echo "2026-10-15T17:00:00Z RELEASE GPU1,GPU2 owner=claude lane=toolcall (test)" >> "$TD/q/p40.claims"
rm -f "$TD/q/post_d1/c1.json" "$STUB_SCREEN_CALLS" "$TD/q/gpu0.lock"
OUT=$(POSTD1_SCREEN=$TD/screen_stub.py bash "$S" --dry-run c1 2>&1 | grep "c1 heldout screen")
echo "$OUT" | grep -q "needed: $TD/screen_stub.py"; check "screen: the dry run shows the screen needed before the rescore" $? "$OUT"
N0=$(wc -l < "$TD/q/post_d1.status")
( export POSTD1_SCREEN=$TD/screen_stub.py POSTD1_RESCORE_RAM_GIB=100000; timeout 60 bash "$S" c1 > /dev/null 2>&1 ); rc=$?
NEW=$(tail -n +$((N0 + 1)) "$TD/q/post_d1.status")
L1=$(echo "$NEW" | grep -n "job=c1 state=screening corpus_sha=" | head -1 | cut -d: -f1)
L2=$(echo "$NEW" | grep -n "job=c1 event=screened dropped=3 " | head -1 | cut -d: -f1)
L3=$(echo "$NEW" | grep -n 'job=c1 state=deferred reason="MemAvailable_[0-9]*GiB_lt_100000"' | head -1 | cut -d: -f1)
[ $rc -eq 0 ] && [ -n "$L1" ] && [ -n "$L2" ] && [ -n "$L3" ] && [ "$L1" -lt "$L2" ] && [ "$L2" -lt "$L3" ] \
  && [ "$(grep -c screen "$STUB_SCREEN_CALLS")" = 1 ] && ! echo "$NEW" | grep -q "job=c1 state=started" \
  && grep -q '"rc": 75' "$TD/q/post_d1/c1.json" && [ ! -e "$TD/q/gpu0.lock" ]
check "screen: c1 screens first (screening, screened, then the rescore gate), once across the passes; no build-text, no gpu0.lock" $? \
  "rc=$rc L=$L1/$L2/$L3 calls=$(cat "$STUB_SCREEN_CALLS" 2>/dev/null | wc -l) $(echo "$NEW" | grep job=c1 | head -5)"
OUT=$(POSTD1_SCREEN=$TD/screen_stub.py bash "$S" --dry-run c1 2>&1 | grep "c1 heldout screen")
echo "$OUT" | grep -q "valid receipt"; check "screen: the dry run shows a valid receipt once screened" $? "$OUT"
[ "$(bash "$S" --lane-of 'job=c1 state=screening corpus_sha=x')" = L-TOOL ]; check "screen: screening lines heartbeat L-TOOL" $? "routing"
new_c1; rm -f "$TD/q/post_d1/c1.json" "$STUB_SCREEN_CALLS"
( export POSTD1_SCREEN=$TD/screen_stub.py POSTD1_RESCORE_RAM_GIB=100000; timeout 60 bash "$S" c1 > /dev/null 2>&1 )
[ "$(grep -c screen "$STUB_SCREEN_CALLS" 2>/dev/null)" = 1 ]; check "screen: a re-exported corpus (new sha) is screened again" $? "$(cat "$STUB_SCREEN_CALLS" 2>/dev/null)"
new_c1; rm -f "$TD/q/post_d1/c1.json" "$STUB_SCREEN_CALLS"
N0=$(wc -l < "$TD/q/post_d1.status")
( export POSTD1_SCREEN=$TD/screen_stub.py STUB_SCREEN_RC=2 POSTD1_RESCORE_RAM_GIB=100000; timeout 60 bash "$S" c1 > /dev/null 2>&1 ); rc=$?
NEW=$(tail -n +$((N0 + 1)) "$TD/q/post_d1.status")
[ $rc -eq 0 ] && grep -q '"rc": 1' "$TD/q/post_d1/c1.json" && grep -q "c1.screen.log" "$TD/q/post_d1/c1.json" \
  && echo "$NEW" | grep -q 'job=c1 state=failed reason="heldout screen rc=2: REFUSED: stub refusal" (no rescore)' \
  && ! echo "$NEW" | grep -q "job=c1 state=started" && [ ! -e "$TD/q/gpu0.lock" ]
check "screen: a refused screen fails c1 alone (rc 1 receipt with the screen log), no started line" $? "rc=$rc $(echo "$NEW" | grep job=c1 | head -4)"
rm -f "$TD/q/post_d1/c1.json"
N0=$(wc -l < "$TD/q/post_d1.status")
( export POSTD1_SCREEN=$TD/screen_stub.py STUB_SCREEN_RC=75 POSTD1_RESCORE_RAM_GIB=100000; timeout 60 bash "$S" c1 > /dev/null 2>&1 ); rc=$?
NEW=$(tail -n +$((N0 + 1)) "$TD/q/post_d1.status")
[ $rc -eq 0 ] && grep -q '"rc": 75' "$TD/q/post_d1/c1.json" && echo "$NEW" | grep -q 'job=c1 state=deferred reason="heldout_screen_deferred: DEFERRED: stub"' \
  && ! echo "$NEW" | grep -q "job=c1 state=started"
check "screen: a deferred screen (rc 75) defers c1, gave_up rc 75 at the cap, no rescore" $? "rc=$rc $(echo "$NEW" | grep job=c1 | head -4)"
rm -f "$TD/in/c1.safetensors" "$TD/in/c1.rows.json" "$TD/in/c1.safetensors.screened" "$TD/in/heldout_screen.json" "$TD/q/post_d1/c1.json" "$STUB_SCREEN_CALLS"

# p2verdict (1.2.0) --------------------------------------------------------------------------------------------
LABPY=/home/cereal/model-surgery-lab/venv/bin/python
D2P=/home/cereal/omni-campaign/distill/tools/d2_plan.py
mkdir -p "$TD/oc/distill/tps/p2-v133"
SUMF=$TD/oc/distill/tps/p2-v133/p2-summary-v133-gpu0.json
echo '{"gpu_index": 0, "cells": {"L4096-mb8": {"state": "completed", "steady_tok_per_s_mean_steps_2_on": 180.5}, "L4096-mb4": {"state": "completed", "steady_tok_per_s_mean_steps_2_on": 170.0}, "L32768-mb1": {"state": "failed", "reason": "OOM"}}}' > "$SUMF"
rm -f "$TD/q/post_d1/p2.json" "$TD/q/p2.verdict"
( export POSTD1_P2_OC=$TD/oc POSTD1_P2_RUN="echo stub-p2-runner"; timeout 120 bash "$S" p2 > /dev/null 2>&1 ); rc=$?
OUT=$(cd /tmp && "$LABPY" "$D2P" --oc "$TD/oc" verdict --verdict "$TD/q/p2.verdict" 2>&1)
[ $rc -eq 0 ] && grep -q '"rc": 0' "$TD/q/post_d1/p2.json" && echo "$OUT" | grep -q '"engine": "streamed"'   && echo "$OUT" | grep -q '"micro_batch": "8"' && echo "$OUT" | grep -q '"accumulation": "1"' && [ -s "$TD/q/post_d1/p2.reading" ]   && grep -q 'job=p2 state=done rc=0 verdict_rc=0 verdict="engine=streamed' "$TD/q/post_d1.status"
check "p2verdict: p2.verdict is d2_plan's format and d2_plan reads it back (streamed, micro 8, acc 1)" $? "rc=$rc out=$(echo "$OUT" | cut -c1-300) $(grep 'job=p2' "$TD/q/post_d1.status" | tail -1)"
echo '{"gpu_index": 0, "cells": {"L4096-mb8": {"state": "failed", "reason": "OOM"}}}' > "$SUMF"
rm -f "$TD/q/post_d1/p2.json" "$TD/q/p2.verdict"
( export POSTD1_P2_OC=$TD/oc POSTD1_P2_RUN="exit 1"; timeout 120 bash "$S" p2 > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && grep -q '"rc": 1' "$TD/q/post_d1/p2.json" && [ ! -e "$TD/q/p2.verdict" ] && grep -q 'job=p2 state=failed rc=1 verdict_rc=10' "$TD/q/post_d1.status"
check "p2verdict: no completed cell -> no p2.verdict (D2 keeps waiting on the summaries), the job fails alone" $? "rc=$rc $(ls "$TD/q") $(grep 'job=p2' "$TD/q/post_d1.status" | tail -1)"

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
