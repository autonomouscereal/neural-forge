#!/usr/bin/env bash
# test_run_toolcall_screen.sh -- run_toolcall_screen_2026-10-15.sh against a stub screen and a stand-in driver, every
# path under declared scratch (lane toolscreen, 2026-10-15).  No card, no live lock, no LANES.json (silent).
#   bash test_run_toolcall_screen.sh [script under test, default $T/run_toolcall_screen_2026-10-15.sh]
#   waits     the watcher waits while the "driver" lives, then screens the sealed export once: rc 0, "screened" status
#   noexport  the driver ends with no export (or a sidecar naming another sha): rc 3, the screen never runs
#   wrongpid  a live pid that is not the driver: rc 2, nothing runs
#   failed    the screen refuses: the watcher returns its rc and writes "screen failed"
#   giveup    the driver outlives the give-up time: rc 4, nothing runs
VERSION="test_run_toolcall_screen 1.0.0"
S=${1:-/home/cereal/omni-campaign/distill/tools/run_toolcall_screen_2026-10-15.sh}
PYV=/home/cereal/model-surgery-lab/venv/bin/python
TD=/mnt/lexar/omni-campaign/scratch/toolcall-screen-test-$$
mkdir -p "$TD/tc/rescore" "$TD/q"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }
cat > "$TD/stub.py" <<'EOF'
import hashlib, os, sys
if sys.argv[1] == "--version":
    print("toolcall_heldout_screen 9.9.9-stub"); sys.exit(0)
cmd, corpus = sys.argv[1], sys.argv[sys.argv.index("--corpus") + 1]
marker = corpus + ".screened"
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if cmd == "check":
    sys.exit(0 if os.path.exists(marker) and open(marker).read() == digest(corpus) else 10)
open(os.environ["STUB_CALLS"], "a").write("screen\n")
rc = int(os.environ.get("STUB_RC", "0"))
if rc == 0:
    open(marker, "w").write(digest(corpus))
    open(os.path.join(os.path.dirname(corpus), "heldout_screen.json"), "w").write('{"hits_in_export": 2}')
sys.exit(rc)
EOF
export TOOLCALL_SCREEN_PY=$PYV TOOLCALL_SCREEN_Q=$TD/q TOOLCALL_SCREEN_TC=$TD/tc TOOLCALL_SCREEN_TOOL=$TD/stub.py \
       TOOLCALL_SCREEN_MATCH="sleep 3133" TOOLCALL_SCREEN_POLL_S=1 STUB_CALLS=$TD/calls
C=$TD/tc/rescore/toolcall-q38-2026-10-14.safetensors
export_ok() { head -c 2048 /dev/urandom > "$C"; echo "{\"corpus\": {\"sha256\": \"$(sha256sum "$C" | cut -c1-64)\"}}" > "${C%.safetensors}.rows.json"; }

# waits
export_ok; rm -f "$TD/calls"
sleep 3133 & K=$!
( sleep 3; kill $K ) &
OUT=$(timeout 60 bash "$S" $K 2>&1); rc=$?
wait $K 2>/dev/null
[ $rc -eq 0 ] && [ "$(grep -c screen "$TD/calls" 2>/dev/null)" = 1 ] && grep -q "toolcall_screen-1.0.0 waiting for the extend driver" "$TD/tc/logs/extend.status" \
  && grep -q "toolcall_screen-1.0.0 screened corpus .* dropped=2" "$TD/tc/logs/extend.status"
check "waits: screens the sealed export once, after the driver ends" $? "rc=$rc $(tail -3 "$TD/tc/logs/extend.status") $OUT"

# noexport
rm -f "$C" "${C%.safetensors}.rows.json" "$TD/calls"
OUT=$(timeout 60 bash "$S" 999999 2>&1); rc=$?
[ $rc -eq 3 ] && [ ! -e "$TD/calls" ] && grep -q "no sealed export" "$TD/tc/logs/extend.status"
check "noexport: no export -> rc 3, the screen never runs" $? "rc=$rc $OUT"
export_ok; echo '{"corpus": {"sha256": "0000"}}' > "${C%.safetensors}.rows.json"
OUT=$(timeout 60 bash "$S" 999999 2>&1); rc=$?
[ $rc -eq 3 ] && [ ! -e "$TD/calls" ]; check "noexport: a sidecar naming another sha -> rc 3" $? "rc=$rc"

# wrongpid
export_ok; sleep 30 & K=$!
OUT=$(timeout 60 bash "$S" $K 2>&1); rc=$?
kill $K 2>/dev/null; wait $K 2>/dev/null
[ $rc -eq 2 ] && [ ! -e "$TD/calls" ] && echo "$OUT" | grep -q "is not sleep 3133"
check "wrongpid: a live pid that is not the driver -> rc 2, nothing runs" $? "rc=$rc $OUT"

# failed
export_ok; rm -f "$TD/calls"
OUT=$(STUB_RC=2 timeout 60 bash "$S" 999999 2>&1); rc=$?
[ $rc -eq 2 ] && [ "$(grep -c screen "$TD/calls")" = 1 ] && grep -q "screen failed rc=2" "$TD/tc/logs/extend.status"
check "failed: a refused screen -> its rc, 'screen failed' status" $? "rc=$rc $(tail -1 "$TD/tc/logs/extend.status")"

# giveup
export_ok; rm -f "$TD/calls"
sleep 3133 & K=$!
OUT=$(TOOLCALL_SCREEN_GIVE_UP=2000-01-01T00:00:00Z timeout 60 bash "$S" $K 2>&1); rc=$?
kill $K 2>/dev/null; wait $K 2>/dev/null
[ $rc -eq 4 ] && [ ! -e "$TD/calls" ]; check "giveup: the driver outlives the give-up time -> rc 4, nothing runs" $? "rc=$rc $OUT"

rm -rf "$TD"
echo "TOOLCALL SCREEN WATCHER TEST: $pass passed, $fail failed"
[ $fail -eq 0 ]
