#!/usr/bin/env bash
# test_x3_queue.sh -- the GPU3 gate of x3_queue.sh against fake claims / nvidia-smi / lock.  No card, no lock, no
# claims file of the real queue is touched (CLAIMS, NVSMI, LOCK3, X3Q_ROOT point into a temp dir).
# 1.1.0 cases: RELEASE lines of other lanes, the CLAIM / RELEASE lines this queue writes as read by the seal queue's
# parser (d1_seal_lib.claim_state) and by lanes.py, and the gpu3.lock taken with the claim.
set -u
D=$(mktemp -d); trap 'rm -rf $D' EXIT
export X3Q_ROOT=$D/x CLAIMS=$D/claims NVSMI=$D/nvsmi LOCK3=$D/gpu3.lock X3Q_SOURCE_ONLY=1 NEED_GIB=0
D1LIB_DIR=${D1LIB_DIR:-/home/cereal/omni-campaign/distill/tools}
fail=0
ok() { echo "ok   [$1]"; }
bad() { echo "FAIL [$1] $2"; fail=1; }
check() { local want=$1 got; got=$(gpu_free); if [[ "$got" == $want ]]; then echo "ok   [$2] -> '${got}'"; else echo "FAIL [$2] want '$want' got '$got'"; fail=1; fi; }
seal_reads() {  # the seal queue's own reading of GPU3: "held" or "free"
  python3 -c "import sys; sys.path.insert(0, '$D1LIB_DIR'); import d1_seal_lib as L, pathlib
print('held' if L.p40_free(pathlib.Path('$CLAIMS'), [3])[0] else 'free')"
}
lanes_reads() {  # lanes.py claims_open for the fold lane: "open" or "closed"
  python3 -c "import sys; sys.path.insert(0, '$D1LIB_DIR'); import lanes
print('open' if any(c['lane'] == 'fold-research' for c in lanes.claims_open('$CLAIMS')) else 'closed')"
}
lock_busy() { ( exec 7>>$LOCK3; flock -n 7 ) && echo free || echo busy; }
printf '#!/bin/sh\necho "3, %s"\n' 100 > $NVSMI; chmod +x $NVSMI
source "$(dirname "$0")/x3_queue.sh"
trap 'rm -rf $D' EXIT
: > $CLAIMS
check "" "empty claims, idle card"
echo "2026-10-14T19:43:02Z GPU3 lane=k3accel owner=claude job=a pid=$$ log=x" >> $CLAIMS
check "claimed by lane k3accel" "live claim (pid alive)"
echo "2026-10-14T19:43:09Z GPU3 lane=k3accel job=a released" >> $CLAIMS
check "" "same lane released"
echo "2026-10-14T20:00:00Z GPU3 lane=k3accel owner=claude job=b pid=999999 log=x" >> $CLAIMS
check "" "stale claim, dead pid"
echo "2026-10-14T14:28:20Z CLAIM GPU1,GPU2 owner=claude lane=toolcall what=x; GPU3 NOT claimed (left for tests)" >> $CLAIMS
check "" "free text 'GPU3 NOT claimed' on a GPU1,GPU2 claim"
echo "2026-10-15T01:00:00Z CLAIM GPU2,GPU3 owner=codex lane=speech what=tts" >> $CLAIMS
check "claimed by lane speech" "multi-card CLAIM line without pid"
echo "2026-10-15T02:00:00Z GPU3 lane=speech job=tts paused (operator)" >> $CLAIMS
check "" "paused frees it"
echo "2026-10-15T03:00:00Z GPU3 lane=fold-research owner=claude job=x3-08b pid=$$ log=x" >> $CLAIMS
check "" "own lane ignored"
printf '#!/bin/sh\necho "3, %s"\n' 16693 > $NVSMI
check "GPU3 holds 16693 MiB" "unannounced process on the card"
printf '#!/bin/sh\necho "3, %s"\n' 100 > $NVSMI
NEED_GIB=100000
check "MemAvailable * GiB < 100000" "RAM gate"
NEED_GIB=0
# ---- 1.1.0
echo "2026-10-15T04:00:00Z CLAIM GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review pid=$$ log=x (release by d1_seal_queue-1.2.0)" >> $CLAIMS
check "claimed by lane d1seal" "seal queue CLAIM GPU1,GPU2,GPU3 (pid alive)"
echo "2026-10-15T05:00:00Z RELEASE GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review released" >> $CLAIMS
check "" "seal queue RELEASE line frees it (1.0.0 ignored RELEASE lines)"
: > $CLAIMS; rm -f $LOCK3
( POLL_S=1; acquire; echo "$(seal_reads) $(lanes_reads) $(lock_busy)" > $D/held; release "test"; echo "$(seal_reads) $(lanes_reads) $(lock_busy)" > $D/after ) > /dev/null 2>&1
[ "$(cat $D/held 2>/dev/null)" = "held open busy" ] && ok "acquire: seal reads held, lanes.py open, gpu3.lock held ($(cat $D/held))" \
  || bad "acquire" "want 'held open busy' got '$(cat $D/held 2>/dev/null)'"
[ "$(cat $D/after 2>/dev/null)" = "free closed free" ] && ok "release: seal reads free, lanes.py closed, lock free ($(cat $D/after))" \
  || bad "release" "want 'free closed free' got '$(cat $D/after 2>/dev/null)'"
grep -q "CLAIM GPU3 owner=claude lane=fold-research job=x3-08b pid=" $CLAIMS && ! grep "CLAIM GPU3" $CLAIMS | grep -q -w released \
  && ok "claim line is CLAIM format with no 'released' word" || bad "claim line" "$(grep CLAIM $CLAIMS)"
grep -q "RELEASE GPU3 owner=claude lane=fold-research job=x3-08b released (test)" $CLAIMS && ok "release line is RELEASE format" \
  || bad "release line" "$(tail -1 $CLAIMS)"
: > $CLAIMS
( exec 8>>$LOCK3; flock -n 8; sleep 6 ) & HOLDER=$!
sleep 1
( POLL_S=1; trap - EXIT; acquire ) > /dev/null 2>&1 & ACQ=$!
sleep 3; touch $X/STOP_QUEUE; wait $ACQ; rc=$?
kill $HOLDER 2>/dev/null; wait $HOLDER 2>/dev/null; rm -f $X/STOP_QUEUE
grep -q "WAIT GPU3: gpu3.lock busy" $X/queue.status && ! grep -q "lane=fold-research" $CLAIMS \
  && ok "a held gpu3.lock blocks the claim (waits, writes no claim; STOP_QUEUE exit rc=$rc)" \
  || bad "busy lock" "$(tail -2 $X/queue.status) / $(cat $CLAIMS)"
( PREEMPT=1; run_stage never touch $D/stage-ran; echo $? > $D/rs ) > /dev/null 2>&1
[ "$(cat $D/rs)" = 75 ] && [ ! -e $D/stage-ran ] && ok "a pending TERM stops run_stage before it starts a stage (rc 75)" \
  || bad "pending TERM" "rc $(cat $D/rs) ran=$([ -e $D/stage-ran ] && echo yes || echo no)"
[ $fail = 0 ] && echo "ALL OK" || { echo "FAILURES"; exit 1; }
