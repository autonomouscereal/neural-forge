#!/usr/bin/env bash
# test_k3_seal_decide.sh -- k3_seal_decide.sh on a throwaway queue dir under declared scratch.  Lane sealdecide, Claude,
# 2026-10-15.  Every external piece is a fixture or a stub: k3_multicard.sh (plan + eqcheck), systemctl, the process
# table (K3D_PROCS), the detached launcher (K3D_LAUNCH records instead of starting), the GPU6 status and pass log, the
# B2 receipts (read by the REAL k3_lib.sh b2_ok), the K3 corpus headers.  No case can reach a GPU or the live queue dir.
#   bash test_k3_seal_decide.sh [script under test, default beside this file] [k3_lib.sh, default $T/k3_lib.sh]
#   usage       an unknown argument -> 2
#   gates       no d1.sealed / D1 unit active / no P2 receipt / P2 running: --gate-once 75 with the reason; armed 75 and
#               NOTHING written (no decision, no GO file, no status line)
#   b2          B2 pending -> 75, nothing but status lines; B2 FAILED -> d2-first without any eqcheck
#   k3first     the section 4.5 arithmetic on fixtures (3090 3.15 h / pass; GPU6 2 layers/h over a midnight wrap, 30 of 93
#               done) -> K3-first 67.9 h vs D2-first 186.7 h: GO gpus=0,4 (no eqcheck_only), max_passes 1, runner 0 and
#               the stagger for 4 launched; the eqcheck ran under a GO "gpus=0 eqcheck_only=1"; a rerun redoes nothing
#   d2first     GPU6 nearly done and fast, slow 3090 -> D2-first 62.7 h vs K3-first 74.22 h: the eqcheck GO removed (backed
#               up), a max_passes of 1 moved to backups, nothing launched
#   eqfail      the 3090 class fails its eqcheck (runner exit 3) -> d2-first, blocked reason recorded, nothing launched
#   q130        K3-first wins but the live gpu6 queue is 1.3.0 -> d2-first "blocked", nothing launched
#   d2plan      (1.0.1) every world has a fixture d2_plan 1.3.0 naming k3_pass_tools.json.  K3-first wins but d2_plan is
#               1.2.0 with a v124 3090 tool -> d2-first "blocked", nothing launched; d2_plan 1.2.0 that merely MENTIONS
#               k3_pass_tools.json (1.0.0's grep passed it) -> blocked; no d2_plan -> blocked; P2 split + d2_plan 1.2.0 ->
#               d2-first, no GO; a v130 3090 tool with d2_plan 1.3.0 -> k3-first (never blocked for being v130);
#               d2_plan 1.12.0 -> reads (a version compare, not a string compare)
#   layers      an eqcheck log with 92 of 93 layers -> no K3-first estimate -> d2-first
#   planning    P2 engine=split -> planning: GO gpus=0, max_passes 1, only runner 0; the read term (compute 2 s/layer
#               -> 68 s/layer, 1.76 h per pass)
#   manual      a GO file this script did not write -> manual, untouched, no eqcheck
#   live0       a k3_multicard.sh 0 runner already lives -> 75, nothing written
#   gatewait    the eqcheck runner stays at its gates past K3D_EQ_START_MAX_S -> stopped, GO removed, 75
#   stagger     --stagger 4 0 starts runner 4 after gpu0's pass START + settled ballast; without a START it keeps waiting
#   dry         --dry-run writes nothing and names the branch it would take
#   single      a second instance -> 5
S=${1:-$(cd "$(dirname "$0")" && pwd)/k3_seal_decide.sh}
LIB=${2:-/home/cereal/omni-campaign/distill/tools/k3_lib.sh}
TD=/mnt/lexar/omni-campaign/scratch/k3d-test-$$
mkdir -p "$TD"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }
jq_() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); [v := (v or {}).get(k) if isinstance(v, dict) else None for k in sys.argv[2].split(".")]; print(v)' "$1" "$2" 2>/dev/null; }

# ------------------------------------------------------------------------------------------------ fixtures
mk_fixture() {  # a fresh world in $W: $1 = eqcheck compute seconds per layer, $2 = eqcheck layers printed (93)
  W=$TD/w$((++NW)); rm -rf "$W"; mkdir -p "$W/q/post_d1" "$W/q/logs/gpu6" "$W/q/logs/k3mc-gpu0" "$W/kc" "$W/k3out" "$W/b2/review-pass" "$W/frozen"
  echo "frozen k3 1.2.4 fixture $NW" > "$W/frozen/k3.py"
  K3SHA=$(sha256sum "$W/frozen/k3.py" | cut -c1-64)
  echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"$K3SHA\"}" > "$W/b2/review-pass/manifest.json"
  echo '{"verdict": "pass"}' > "$W/b2/parity_b2_receipt.json"
  for k in 0 1 2 3; do mkdir -p "$W/kc/pass-$k"; python3 - "$W/kc/pass-$k/k3_input_ids.safetensors" <<'PY'
import json, struct, sys
h = json.dumps({"input_ids": {"dtype": "I32", "shape": [122, 4096], "data_offsets": [0, 0]}}).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(h)) + h)
PY
  done
  echo "2026-10-16T05:00:00Z D1 sealed (fixture)" > "$W/q/d1.sealed"
  echo '{"job": "p2", "rc": 0}' > "$W/q/post_d1/p2.json"
  printf '#!/bin/bash\necho "${SYSTEMCTL_STATE:-inactive}"\n' > "$W/systemctl"; chmod +x "$W/systemctl"
  # GPU6: queue 1.4.0 live, pass 0 started by it, running; its log: 30 layers, 30 min apart, across midnight
  cat > "$W/q/gpu6.status" <<EOF
2026-10-15T13:00:00Z queue START queue_gpu6.sh 1.4.0 k3tool=k3_teacher_cache 1.2.4 sha256=${K3SHA:0:16}
2026-10-15T18:00:00Z k3-pass-0 WAIT S8/GPU6
2026-10-15T20:00:00Z k3-pass-0 START v124 ${K3SHA:0:16} on quadro-rtx-8000-sm75, group tokens 65536
EOF
  gpu6_log 30 1800
  cat > "$W/procs" <<EOF
200 bash $W/q/gpu6.sh
300 /venv/bin/python $W/q/frozen/k3_teacher_cache-gpu6.py run --input-ids $W/kc/pass-0/k3_input_ids.safetensors --pass-count 1 --pass-index 0 --pass-dir $W/k3out/pass-0-of-4 --state-dir /lexar/scratch/k3cache-state/tool-${K3SHA:0:16} --device cuda:0 --resume
1 /sbin/init
EOF
  eq_log "$1" "${2:-93}"
  cat > "$W/mc" <<'EOF'
#!/bin/bash
# stub k3_multicard.sh: the plan selftest, and an eqcheck run driven by STUB_EQ (pass | fail | hang)
if [ "${K3MC_SELFTEST:-}" = plan ]; then
  echo "k3_multicard.sh 1.0.0 gpu 0 (nvidia-geforce-rtx-3090-sm86) 2026-10-16T05:00:00Z: tool ${STUB_KIND:-v124} k3_teacher_cache 1.2.4 ${STUB_SHA:0:16}, group tokens 32768, B2 dir $STUB_W/b2 (rebase 0)"
  echo "gates: d1 x | go x | b2 b2_ok | ruling x | lock x | eqcheck $STUB_W/q/k3_cardclass/v.json in $STUB_W/eq"
  echo "thermal: stop 90 C x3, hard 96 C"
  echo "K3 out $STUB_W/k3out; state /x; lib k3_lib.sh 1.0.0 e70ba063"
  echo "SELFTEST plan: exit before any gate"; exit 0
fi
st() { echo "$(date -u +%FT%TZ) gpu$1 $2 $3 ${4:-}" >> "$STUB_W/q/k3mc-gpu$1.status"; }
echo "mc $1 go=[$(tr '\n' ' ' < "$STUB_W/q/k3_multicard.go" 2>/dev/null)]" >> "$STUB_W/calls"
st "$1" runner START stub
case "${STUB_EQ:-pass}" in
  hang) st "$1" gate WAIT "lock: held"; sleep 60; exit 0 ;;
  fail) st "$1" eqcheck START x; mkdir -p "$STUB_W/q/k3_cardclass"
        echo "{\"verdict\": \"fail\", \"tool_sha256\": \"$STUB_SHA\", \"card_class\": \"nvidia-geforce-rtx-3090-sm86\"}" > "$STUB_W/q/k3_cardclass/v.json"
        st "$1" eqcheck FAIL "functional"; exit 3 ;;
  *)    st "$1" eqcheck START x; mkdir -p "$STUB_W/q/k3_cardclass"
        echo "{\"verdict\": \"pass\", \"tool_sha256\": \"$STUB_SHA\", \"card_class\": \"nvidia-geforce-rtx-3090-sm86\"}" > "$STUB_W/q/k3_cardclass/v.json"
        st "$1" eqcheck PASS "PARITY PASS"; st "$1" runner END "eqcheck_only=1"; exit 0 ;;
esac
EOF
  chmod +x "$W/mc"
  cat > "$W/launch" <<'EOF'
#!/bin/bash
# stub launcher: records, and a "runner 0" says it took gpu0.lock (as k3_multicard's status line does)
echo "launch $*" >> "$STUB_W/calls"
[ "$1 $2" = "runner 0" ] && echo "$(date -u +%FT%TZ) gpu0 gate OK lock $STUB_W/q/gpu0.lock (helper pid 1)" >> "$STUB_W/q/k3mc-gpu0.status"
echo 4242
EOF
  chmod +x "$W/launch"
  d2plan_fixture 1.3.0 yes
  export STUB_W=$W STUB_SHA=$K3SHA STUB_EQ=pass STUB_KIND=v124
  export K3D_Q=$W/q K3D_T=$W/t K3D_MC=$W/mc K3D_K3LIB=$LIB K3D_K3TOOL=$W/frozen/k3.py K3D_SYSTEMCTL=$W/systemctl \
         K3D_PROCS=$W/procs K3D_KC=$W/kc K3D_LAUNCH=$W/launch K3D_D2_PLAN=$W/d2_plan.py K3D_POLL_S=1 K3D_LOCK_WAIT_S=10 \
         K3D_MC_LOGS=$W/q/logs K3D_MC_STATUS_DIR=$W/q LANES_FILE=$W/none/LANES.json LANES_PY=$W/none/lanes.py
  unset SYSTEMCTL_STATE K3D_EQ_START_MAX_S K3D_STAGGER_FROM_LINE
}
gpu6_log() {  # $1 layers printed, $2 seconds between them, from 12:30:00 host time (30 x 30 min: L23 at midnight, so the
  # rate window L18-L29 spans it)
  python3 - "$W/q/logs/gpu6/k3-pass-0.log" "$1" "$2" <<'PY'
import sys
path, n, gap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "w") as f:
    f.write("== k3-pass-0 attempt 1 group_tokens 65536 tool x (v124) class quadro-rtx-8000-sm75 gpu 6\n")
    for l in range(n):
        t = (45000 + l * gap) % 86400
        f.write(f"[k3_teacher_cache {t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}] L{l:02d} {'MLA' if l % 4 == 3 else 'KDA'}: "
                f"read 70.0s (243.0 MB/s), compute 900.0s over 499,712 positions (330,000 real, 16 groups; 555 pos/s) "
                f"(engine dequant 1.0s), blocks 3, device peak 1 B, rss 1 B\n")
PY
}
eq_log() {  # $1 compute s per layer at 45,056 positions, $2 layers printed
  python3 - "$W/q/logs/k3mc-gpu0/eqcheck.log" "$1" "$2" "${K3SHA:0:16}" <<'PY'
import sys
path, c, n, sha16 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with open(path, "w") as f:
    f.write("== eqcheck 2026-10-16T05:10:00Z v124 0000000000000000 on nvidia-geforce-rtx-3090-sm86\n")
    f.write("[k3_teacher_cache 23:10:00] L00 KDA: read 1.0s (9.0 MB/s), compute 9999.0s over 45,056 positions (5,868 real, 2 groups; 5 pos/s) x\n")
    f.write(f"== eqcheck 2026-10-16T05:12:00Z v124 {sha16} on nvidia-geforce-rtx-3090-sm86\n")
    for l in range(n):
        f.write(f"[k3_teacher_cache 23:{l % 60:02d}:00] L{l:02d} {'MLA' if l % 4 == 3 else 'KDA'}: read 20.0s (850.0 MB/s), "
                f"compute {c:.1f}s over 45,056 positions (5,868 real, 2 groups; {45056 / c:,.0f} pos/s) (engine dequant 1.0s), "
                f"blocks 1, device peak 1 B, rss 1 B\n")
PY
}
d2plan_fixture() {  # $1 TOOL_VERSION, $2 yes = reads k3_pass_tools.json | mention = only a comment names it | no
  case "$2" in
    yes)     printf 'TOOL_VERSION = "%s"  # fixture\nPASS_TOOLS = "k3_pass_tools.json"\n' "$1" > "$W/d2_plan.py" ;;
    mention) printf 'TOOL_VERSION = "%s"  # fixture\n# TODO: read k3_pass_tools.json (plan section 6)\n' "$1" > "$W/d2_plan.py" ;;
    *)       printf 'TOOL_VERSION = "%s"  # fixture\n' "$1" > "$W/d2_plan.py" ;;
  esac
}
nothing_written() { [ ! -e "$W/q/k3_seal_decision.json" ] && [ ! -e "$W/q/k3_multicard.go" ] && [ ! -e "$W/calls" ]; }
NW=0

# usage --------------------------------------------------------------------------------------------------------
mk_fixture 10
bash "$S" --bogus > /dev/null 2>&1; check "usage: an unknown argument -> 2" $([ $? -eq 2 ]; echo $?) "rc"

# gates --------------------------------------------------------------------------------------------------------
rm -f "$W/q/d1.sealed"
OUT=$(bash "$S" --gate-once 2>&1); rc=$?
[ $rc -eq 75 ] && echo "$OUT" | grep -q "no_d1.sealed"; check "gates: no d1.sealed -> --gate-once 75" $? "rc=$rc $OUT"
OUT=$(bash "$S" 2>&1); rc=$?
[ $rc -eq 75 ] && nothing_written && [ ! -e "$W/q/k3_seal_decide.status" ]; check "gates: armed without d1.sealed -> 75, nothing written" $? "rc=$rc $(ls "$W/q") $OUT"
echo sealed > "$W/q/d1.sealed"
OUT=$(SYSTEMCTL_STATE=active bash "$S" 2>&1); rc=$?
[ $rc -eq 75 ] && echo "$OUT" | grep -q "D1_unit_active" && nothing_written; check "gates: D1 unit active -> 75, nothing written" $? "rc=$rc $OUT"
rm -f "$W/q/post_d1/p2.json"
OUT=$(bash "$S" --gate-once 2>&1); rc=$?
[ $rc -eq 75 ] && echo "$OUT" | grep -q "no_P2_receipt"; check "gates: no P2 receipt -> 75" $? "rc=$rc $OUT"
echo '{"rc": 1}' > "$W/q/post_d1/p2.json"; echo "77 bash /x/tools/run_p2_v133.sh 0" >> "$W/procs"
OUT=$(bash "$S" 2>&1); rc=$?
[ $rc -eq 75 ] && echo "$OUT" | grep -q "P2_running" && nothing_written; check "gates: P2 still running -> 75, nothing written" $? "rc=$rc $OUT"
sed -i '/run_p2_v133/d' "$W/procs"
OUT=$(bash "$S" --gate-once 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$OUT" = ready ]; check "gates: all open (a failed P2 still counts as 'after P2') -> ready" $? "rc=$rc $OUT"

# b2 -----------------------------------------------------------------------------------------------------------
rm -f "$W/b2/parity_b2_receipt.json"
OUT=$(bash "$S" 2>&1); rc=$?
[ $rc -eq 75 ] && nothing_written && grep -q "deferred: B2 has not passed" "$W/q/k3_seal_decide.status"
check "b2: B2 pending -> 75, no decision / GO / eqcheck" $? "rc=$rc $(tail -2 "$W/q/k3_seal_decide.status" 2>/dev/null) $OUT"
OUT=$(bash "$S" --gate-once 2>&1); rc=$?
[ $rc -eq 75 ] && echo "$OUT" | grep -q "B2_not_passed_yet($W/b2)"; check "b2: --gate-once names B2 pending (post_d1 never starts the script for it)" $? "rc=$rc $OUT"
echo '{"verdict": "fail"}' > "$W/b2/parity_b2_receipt.json"
OUT=$(bash "$S" 2>&1); rc=$?
[ $rc -eq 0 ] && [ "$(jq_ "$W/q/k3_seal_decision.json" branch)" = d2-first ] && [ "$(jq_ "$W/q/k3_seal_decision.json" applied)" = True ] \
  && ! grep -q "^mc " "$W/calls" 2>/dev/null && [ ! -e "$W/q/k3_multicard.go" ] && jq_ "$W/q/k3_seal_decision.json" reason | grep -q "B2 FAILED"
check "b2: B2 FAILED -> d2-first, no eqcheck" $? "rc=$rc $(jq_ "$W/q/k3_seal_decision.json" reason) $(cat "$W/calls" 2>/dev/null)"

# k3first ------------------------------------------------------------------------------------------------------
mk_fixture 10
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = k3-first ] && [ "$(jq_ "$D" applied)" = True ]; check "k3first: branch k3-first, applied" $? "rc=$rc $(jq_ "$D" reason) $(echo "$OUT" | tail -3)"
[ "$(jq_ "$D" estimate.finish.k3_first_finish_h)" = 67.9 ] && [ "$(jq_ "$D" estimate.finish.d2_first_finish_h)" = 186.7 ] \
  && [ "$(jq_ "$D" estimate.finish.pass_3090_hours.3)" = 3.15 ] && [ "$(jq_ "$D" estimate.gpu6.layers_per_hour)" = 2.0 ] \
  && [ "$(jq_ "$D" estimate.gpu6.current_remaining_hours)" = 31.5 ] && [ "$(jq_ "$D" estimate.gpu6.full_pass_hours)" = 46.5 ] \
  && [ "$(jq_ "$D" estimate.eqcheck.layers_printed)" = 93 ] && [ "$(jq_ "$D" estimate.gpu6.layers_done)" = 30 ]
check "k3first: arithmetic (3090 3.15 h/pass; GPU6 2.0 layers/h across midnight, 31.5 h left, 46.5 h/pass; K3-first 67.9 h, D2-first 186.7 h)" $? \
  "$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["estimate"]; print(d["finish"], d["gpu6"])' "$D" 2>&1 | cut -c1-900)"
S3=$(python3 -c 'import json,sys; print([(s["pass"], s["gpu"], s["start_h"], s["end_h"]) for s in json.load(open(sys.argv[1]))["estimate"]["finish"]["schedule_3090"]])' "$D")
python3 - "$D" <<'PY'
import json, sys
got = [(s["pass"], s["gpu"], s["start_h"], s["end_h"]) for s in json.load(open(sys.argv[1]))["estimate"]["finish"]["schedule_3090"]]
want = [(3, 0, 0.0, 3.15), (2, 4, 3.15, 6.30), (1, 0, 6.30, 9.455)]
sys.exit(0 if len(got) == 3 and all(g[:2] == w[:2] and abs(g[2] - w[2]) < 0.011 and abs(g[3] - w[3]) < 0.011 for g, w in zip(got, want)) else 1)
PY
check "k3first: at most 2 K3 passes at once (GPU6's pass + one 3090 until GPU6 ends; passes claimed 3, 2, 1)" $? "$S3"
grep -qx "mc 0 go=\[k3-multicard-go gpus=0 eqcheck_only=1 by=k3_seal_decide .*\]" "$W/calls"
check "k3first: the eqcheck ran under GO gpus=0 eqcheck_only=1" $? "$(cat "$W/calls")"
G=$(tr '\n' ' ' < "$W/q/k3_multicard.go")
[ "$(head -1 "$W/q/k3_multicard.go")" = k3-multicard-go ] && grep -qx "gpus=0,4" "$W/q/k3_multicard.go" && ! grep -q eqcheck_only "$W/q/k3_multicard.go" \
  && [ "$(cat "$W/q/k3_gpu6.max_passes")" = 1 ]
check "k3first: GO gpus=0,4 without eqcheck_only; k3_gpu6.max_passes = 1 (the passes the live gpu6 queue started)" $? "go=[$G] maxp=$(cat "$W/q/k3_gpu6.max_passes" 2>&1)"
L=$(grep "^launch" "$W/calls" | tr '\n' '|')
echo "$L" | grep -q "^launch runner 0|launch stagger 4 0 [0-9]*|$"; check "k3first: runner 0 launched, then the stagger for 4 (from the status line count)" $? "$L"
python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["actions"]; sys.exit(0 if any(x.get("action")=="wait" and "gate OK lock" in x.get("result","") for x in a) else 1)' "$D"
check "k3first: the job waited until runner 0 held gpu0.lock" $? "$(jq_ "$D" actions)"
ls "$W/q/backups/k3_multicard.go.eqcheck-"* > /dev/null 2>&1; check "k3first: the eqcheck GO was backed up before the rewrite" $? "$(ls "$W/q/backups" 2>&1)"
grep -q "decision=k3-first applied=1 .*receipt=$D" "$W/q/k3_seal_decide.status"; check "k3first: status line with the decision and its receipt" $? "$(tail -2 "$W/q/k3_seal_decide.status")"
n=$(wc -l < "$W/calls"); OUT=$(bash "$S" 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$OUT" | grep -q "decision already applied" && [ "$(wc -l < "$W/calls")" = "$n" ]
check "k3first: a rerun redoes nothing" $? "rc=$rc $OUT"
OUT=$(bash "$S" --gate-once 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$OUT" | grep -q "already applied"; check "k3first: --gate-once after the decision -> ready (post_d1 records it)" $? "rc=$rc $OUT"

# d2first ------------------------------------------------------------------------------------------------------
mk_fixture 60
gpu6_log 90 600
echo 1 > "$W/q/k3_gpu6.max_passes"
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && [ "$(jq_ "$D" estimate.finish.d2_first_finish_h)" = 62.7 ] \
  && [ "$(jq_ "$D" estimate.finish.k3_first_finish_h)" = 74.22 ] && [ "$(jq_ "$D" estimate.finish.pass_3090_hours.1)" = 18.91 ]
check "d2first: GPU6 6 layers/h, 3 left; 3090 18.91 h/pass -> D2-first 62.7 h < K3-first 74.22 h" $? "rc=$rc $(jq_ "$D" reason) $(jq_ "$D" estimate.finish)"
[ ! -e "$W/q/k3_multicard.go" ] && ls "$W/q/backups/k3_multicard.go.d2first-"* > /dev/null 2>&1 && ! grep -q "^launch" "$W/calls" \
  && [ ! -e "$W/q/k3_gpu6.max_passes" ] && ls "$W/q/backups/k3_gpu6.max_passes.d2first-"* > /dev/null 2>&1
check "d2first: eqcheck GO removed (backed up), max_passes 1 moved to backups, nothing launched" $? "$(ls "$W/q" "$W/q/backups" 2>&1 | tr '\n' ' ') $(cat "$W/calls")"

# eqfail -------------------------------------------------------------------------------------------------------
mk_fixture 10; export STUB_EQ=fail
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "blocked: 3090 class eqcheck: fail" \
  && [ "$(jq_ "$D" card_class_check.runner_rc)" = 3 ] && ! grep -q "^launch" "$W/calls" && [ ! -e "$W/q/k3_multicard.go" ]
check "eqfail: the 3090 class fails -> d2-first (K3-first blocked), GO removed, nothing launched" $? "rc=$rc $(jq_ "$D" reason)"

# q130 ---------------------------------------------------------------------------------------------------------
mk_fixture 10
sed -i 's/queue_gpu6.sh 1.4.0/queue_gpu6.sh 1.3.0/' "$W/q/gpu6.status"
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "queue_gpu6.sh 1.3.0" && ! grep -q "^launch" "$W/calls" \
  && [ ! -e "$W/q/k3_gpu6.max_passes" ]
check "q130: K3-first wins but the live gpu6 queue is 1.3.0 -> d2-first, blocked, nothing launched" $? "rc=$rc $(jq_ "$D" reason)"

# d2plan (1.0.1) -----------------------------------------------------------------------------------------------
mk_fixture 10; d2plan_fixture 1.2.0 no
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "but blocked: d2_plan does not read k3_pass_tools.json (d2_plan 1.2.0 < 1.3.0" \
  && [ "$(jq_ "$D" estimate.d2_plan.reads_k3_pass_tools)" = False ] && [ "$(jq_ "$D" estimate.d2_plan.plan_tool_kind)" = v124 ] \
  && ! grep -q "^launch" "$W/calls" && [ ! -e "$W/q/k3_gpu6.max_passes" ] && [ ! -e "$W/q/k3_multicard.go" ]
check "d2plan: K3-first wins, v124 3090 tool, d2_plan 1.2.0 -> d2-first, blocked, nothing launched" $? "rc=$rc $(jq_ "$D" reason) $(jq_ "$D" estimate.d2_plan)"
mk_fixture 10; d2plan_fixture 1.2.0 mention
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "d2_plan 1.2.0 < 1.3.0" && ! grep -q "^launch" "$W/calls"
check "d2plan: d2_plan 1.2.0 that only mentions k3_pass_tools.json (1.0.0's grep passed it) -> blocked" $? "rc=$rc $(jq_ "$D" reason)"
mk_fixture 10; rm -f "$W/d2_plan.py"
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "d2_plan.py missing" && ! grep -q "^launch" "$W/calls"
check "d2plan: no d2_plan at all -> blocked" $? "rc=$rc $(jq_ "$D" reason)"
mk_fixture 2; d2plan_fixture 1.2.0 no
printf 'engine=split\nmicro_batch=4\naccumulation=4\n' > "$W/q/p2.verdict"
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "P2 engine split (planning, plan 4.4) but K3 on GPU0 is blocked: d2_plan does not read" \
  && ! grep -q "^launch" "$W/calls" && [ ! -e "$W/q/k3_multicard.go" ] && [ ! -e "$W/q/k3_gpu6.max_passes" ]
check "d2plan: P2 split (planning) with d2_plan 1.2.0 -> d2-first, no GO, nothing launched" $? "rc=$rc $(jq_ "$D" reason)"
mk_fixture 10; export STUB_KIND=v130
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = k3-first ] && [ "$(jq_ "$D" estimate.d2_plan.plan_tool_kind)" = v130 ] \
  && [ "$(jq_ "$D" estimate.d2_plan.reads_k3_pass_tools)" = True ] && [ "$(jq_ "$D" estimate.d2_plan.tool_version)" = 1.3.0 ] \
  && [ "$(jq_ "$D" estimate.k3_on_3090_blocked)" = "[]" ] && grep -q "^launch runner 0" "$W/calls"
check "d2plan: a v130 3090 tool with d2_plan 1.3.0 -> k3-first (not blocked for being v130)" $? "rc=$rc $(jq_ "$D" reason) $(jq_ "$D" estimate.k3_on_3090_blocked)"
export STUB_KIND=v124
mk_fixture 10; d2plan_fixture 1.12.0 yes
OUT=$(bash "$S" --dry-run 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$OUT" | grep -q "d2_plan: d2_plan 1.12.0 reads k3_pass_tools.json"
check "d2plan: d2_plan 1.12.0 reads (version compare, not string compare)" $? "rc=$rc $(echo "$OUT" | grep d2_plan)"

# layers -------------------------------------------------------------------------------------------------------
mk_fixture 10 92
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = d2-first ] && jq_ "$D" reason | grep -q "no K3-first estimate" && [ "$(jq_ "$D" estimate.eqcheck.layers_printed)" = 92 ]
check "layers: 92 of 93 eqcheck layers (the other tool's segment ignored) -> no estimate -> d2-first" $? "rc=$rc $(jq_ "$D" reason)"

# planning -----------------------------------------------------------------------------------------------------
mk_fixture 2
printf 'engine=split\nmicro_batch=4\naccumulation=4\n' > "$W/q/p2.verdict"
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = planning ] && grep -qx "gpus=0" "$W/q/k3_multicard.go" && [ "$(cat "$W/q/k3_gpu6.max_passes")" = 1 ] \
  && [ "$(grep -c '^launch' "$W/calls")" = 1 ] && grep -q "^launch runner 0" "$W/calls"
check "planning: P2 split -> GO gpus=0, max_passes 1, runner 0 only" $? "rc=$rc $(jq_ "$D" reason) $(cat "$W/calls")"
[ "$(jq_ "$D" estimate.finish.pass_3090_hours.3)" = 1.76 ]; check "planning: the read term (2 s/layer -> 68 s/layer: 1.76 h per pass)" $? "$(jq_ "$D" estimate.finish.pass_3090_hours)"

# manual -------------------------------------------------------------------------------------------------------
mk_fixture 10
printf 'k3-multicard-go\ngpus=0,4\n' > "$W/q/k3_multicard.go"
OUT=$(bash "$S" 2>&1); rc=$?
D=$W/q/k3_seal_decision.json
[ $rc -eq 0 ] && [ "$(jq_ "$D" branch)" = manual ] && [ "$(tr '\n' ' ' < "$W/q/k3_multicard.go")" = "k3-multicard-go gpus=0,4 " ] && [ ! -e "$W/calls" ]
check "manual: a GO not written by this script -> manual, untouched, no eqcheck" $? "rc=$rc $(jq_ "$D" reason) $(cat "$W/calls" 2>/dev/null)"

# live0 --------------------------------------------------------------------------------------------------------
mk_fixture 10
echo "555 bash $W/q/k3_multicard.sh 0" >> "$W/procs"
OUT=$(bash "$S" 2>&1); rc=$?
[ $rc -eq 75 ] && [ ! -e "$W/q/k3_seal_decision.json" ] && [ ! -e "$W/q/k3_multicard.go" ] && [ ! -e "$W/calls" ]
check "live0: a k3_multicard.sh 0 runner already lives -> 75, nothing written" $? "rc=$rc $OUT"
OUT=$(bash "$S" --gate-once 2>&1); rc=$?
[ $rc -eq 75 ] && echo "$OUT" | grep -q "k3_multicard_0_already_live"; check "live0: --gate-once names the live runner" $? "rc=$rc $OUT"

# gatewait -----------------------------------------------------------------------------------------------------
mk_fixture 10; export STUB_EQ=hang
t0=$(date +%s); OUT=$(K3D_EQ_START_MAX_S=3 timeout 50 bash "$S" 2>&1); rc=$?
[ $rc -eq 75 ] && [ ! -e "$W/q/k3_multicard.go" ] && [ ! -e "$W/q/k3_seal_decision.json" ] && grep -q "still at its gates" "$W/q/k3_seal_decide.status" \
  && [ $(( $(date +%s) - t0 )) -lt 40 ] && ! pgrep -f "$W/mc 0" > /dev/null
check "gatewait: runner still at its gates after K3D_EQ_START_MAX_S -> stopped, GO removed, 75" $? "rc=$rc $(tail -1 "$W/q/k3_seal_decide.status" 2>/dev/null) $(pgrep -af "$W/mc")"

# stagger ------------------------------------------------------------------------------------------------------
mk_fixture 10
echo "2026-10-16T08:00:00Z gpu0 runner END eqcheck_only=1 (the eqcheck run)" > "$W/q/k3mc-gpu0.status"
n0=$(wc -l < "$W/q/k3mc-gpu0.status"); echo "601 bash $W/q/k3_multicard.sh 0" >> "$W/procs"
( export K3D_STAGGER_FROM_LINE=$n0; timeout 5 bash "$S" --stagger 4 0 > /dev/null 2>&1 ); rc=$?
[ $rc -eq 124 ] && ! grep -q "launch exec 4" "$W/calls" 2>/dev/null
check "stagger: gpu0 alive without a pass START (an old END line ignored) -> keeps waiting" $? "rc=$rc $(cat "$W/calls" 2>/dev/null)"
echo "2026-10-16T08:01:00Z gpu0 k3-pass-3 START v124 x" >> "$W/q/k3mc-gpu0.status"
echo "2026-10-16T08:02:00Z pid 9 RssAnon 3.0 GiB, ballast 80 GiB (want 80), MemAvailable 60.0 GiB" > "$W/q/logs/k3mc-gpu0/k3-pass-3.ram.log"
( export K3D_STAGGER_FROM_LINE=$n0; timeout 20 bash "$S" --stagger 4 0 > /dev/null 2>&1 ); rc=$?
[ $rc -eq 0 ] && grep -q "^launch exec 4" "$W/calls" && grep -q "stagger gpu4: gpu0 k3-pass-3 started and its ballast settled" "$W/q/k3_seal_decide.status"
check "stagger: gpu0's pass START + settled ballast -> exec k3_multicard.sh 4" $? "rc=$rc $(cat "$W/calls" 2>/dev/null) $(tail -1 "$W/q/k3_seal_decide.status")"
bash "$S" --stagger 5 0 > /dev/null 2>&1; check "stagger: only 4 0 / 0 4 -> otherwise 2" $([ $? -eq 2 ]; echo $?) "rc"

# dry ----------------------------------------------------------------------------------------------------------
mk_fixture 10
before=$(ls -laR --time-style=full-iso "$W" | md5sum)
OUT=$(bash "$S" --dry-run 2>&1); rc=$?
after=$(ls -laR --time-style=full-iso "$W" | md5sum)
[ $rc -eq 0 ] && [ "$before" = "$after" ] && echo "$OUT" | grep -q "DRY RUN: nothing was written" && echo "$OUT" | grep -q "would take: d2-first" \
  && echo "$OUT" | grep -q "blocked \['3090 class eqcheck: absent'\]"
check "dry: writes nothing; names the branch (d2-first while the 3090 class is unchecked)" $? "rc=$rc $(echo "$OUT" | tail -4)"

# single -------------------------------------------------------------------------------------------------------
flock "$W/q/k3_seal_decide.lock" sleep 15 & H=$!; sleep 1
OUT=$(timeout 10 bash "$S" 2>&1); rc=$?
kill $H 2>/dev/null; wait $H 2>/dev/null
[ $rc -eq 5 ]; check "single: a second instance exits 5" $? "rc=$rc $OUT"

pkill -f "$TD/" 2>/dev/null
rm -rf "$TD"
echo "K3 SEAL DECIDE SELFTEST ($(sed -n 's/^VERSION="\(.*\)"$/\1/p' "$S")): $pass passed, $fail failed"
[ $fail -eq 0 ]
