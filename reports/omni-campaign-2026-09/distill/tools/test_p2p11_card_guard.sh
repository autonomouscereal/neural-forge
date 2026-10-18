#!/usr/bin/env bash
# test_p2p11_card_guard.sh -- tests for p2p11_card_guard.sh 1.0.0 and for how run_p2_v133.sh / run_p11_omni1_v110.sh
# 1.1.0 use it.  CPU only, no GPU, no docker: `docker` and `nvidia-smi` are stub scripts in a temp dir whose answers
# a test changes mid-run; find / awk / the lease-dir mtimes / the seal marker are real files.  Never touches the real
# runtime_leases or queues.  Run:  bash test_p2p11_card_guard.sh   (prints "N passed, M failed"; exit 1 on a failure)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GUARD=$HERE/p2p11_card_guard.sh
PASS=0; FAIL=0
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT

# One sandbox per case: fake queue dir, fake lease root (with an OLD D1 lease dir), stubs driven by files.
sandbox() {
  S=$TMP/$1; rm -rf $S; mkdir -p $S/q $S/leases/D1-native-01a10f78-oct14-c37-recovery-r6 $S/bin
  touch -d '-3 hours' $S/leases/D1-native-01a10f78-oct14-c37-recovery-r6/live.json   # the file first, then its dir
  touch -d '-3 hours' $S/leases/D1-native-01a10f78-oct14-c37-recovery-r6
  echo "" > $S/docker.out; echo 0 > $S/docker.rc; echo "0, 368" > $S/smi.out; : > $S/docker.calls
  printf 'MemTotal: 263000000 kB\nMemAvailable: 104857600 kB\n' > $S/meminfo   # 100 GiB
  cat > $S/bin/docker <<EOF
#!/usr/bin/env bash
echo x >> $S/docker.calls
n=\$(wc -l < $S/docker.calls)
if [ -f $S/docker.out.after1 ] && [ \$n -ge 2 ]; then cat $S/docker.out.after1; else cat $S/docker.out; fi
exit \$(cat $S/docker.rc)
EOF
  cat > $S/bin/nvidia-smi <<EOF
#!/usr/bin/env bash
cat $S/smi.out
EOF
  chmod +x $S/bin/docker $S/bin/nvidia-smi
}
seal() { echo "/mnt/lexar/omni-campaign/deltas/OC02-D-test/seal.json $(date -u +%FT%TZ)" > $S/q/d1.sealed; touch -d "${1:--2 hours}" $S/q/d1.sealed; }

# run_guard <gpu> <shell snippet>: source the guard in a subshell with the sandbox wired in; prints its output, returns its rc
run_guard() {
  local gpu=$1 snippet=$2
  ( GPU_INDEX=$gpu Q=$S/q SLOT_FILE=${SLOT:-}
    P2P11_DOCKER=$S/bin/docker P2P11_NVSMI=$S/bin/nvidia-smi P2P11_MEMINFO=$S/meminfo P2P11_LEASES=$S/leases
    P2P11_MEM_POLL_S=${POLL:-1} P2P11_MEM_WAIT_MAX_S=${WAITMAX:-20} P2P11_D1_QUIET_MIN=30
    stamp() { date -u +%H:%M:%S; }
    refuse() { echo "REFUSED: $*"; exit 2; }
    . $GUARD
    eval "$snippet"
    echo "SNIPPET-END" ) 2>&1
}
check() {  # check <name> <expected rc> <expected output regex> <actual output> <actual rc>
  local name=$1 want_rc=$2 want=$3 out=$4 rc=$5
  if [ "$rc" = "$want_rc" ] && echo "$out" | grep -Eq -- "$want"; then PASS=$((PASS + 1)); echo "PASS $name"
  else FAIL=$((FAIL + 1)); echo "FAIL $name (rc $rc, want $want_rc; want /$want/)"; echo "$out" | sed 's/^/    | /'; fi
}

# ---- card_clear on the D1 cards
sandbox c1; seal; out=$(run_guard 0 'card_clear t'); check "clear: sealed, quiet leases, empty card" 0 "CARD CLEAR \[t\] GPU0 368 MiB; D1 seal marker: /mnt" "$out" $?
sandbox c2; seal; echo codex-d1-native-01a10f78-oct14-c37-recovery-r6 > $S/docker.out
out=$(run_guard 4 'card_clear t'); check "refuse: D1 container running" 2 "D1 container codex-d1-native-01a10f78-oct14-c37-recovery-r6 is running" "$out" $?
sandbox c3; seal; echo 1 > $S/docker.rc
out=$(run_guard 0 'card_clear t'); check "refuse: docker ps fails" 2 "docker ps failed" "$out" $?
sandbox c4; out=$(run_guard 0 'card_clear t'); check "refuse: no container but NO seal marker" 2 "no D1 seal marker" "$out" $?
sandbox c5; : > $S/q/d1.sealed; touch -d '-2 hours' $S/q/d1.sealed
out=$(run_guard 0 'card_clear t'); check "refuse: empty seal marker" 2 "no D1 seal marker" "$out" $?
sandbox c6; seal '-2 hours'; mkdir -p $S/leases/D1-native-01a10f78-oct14-c37-recovery-r7; touch -d '-1 hours' $S/leases/D1-native-01a10f78-oct14-c37-recovery-r7
out=$(run_guard 0 'card_clear t'); check "refuse: a D1 lease dir newer than the seal marker (attempt after the seal)" 2 "recovery-r7 is newer than the D1 seal marker" "$out" $?
sandbox c7; seal '-1 minutes'; touch -d '-5 minutes' $S/leases/D1-native-01a10f78-oct14-c37-recovery-r6/live.json
out=$(run_guard 0 'card_clear t'); check "refuse: a D1 lease changed in the last 30 min" 2 "changed in the last 30 min" "$out" $?
sandbox c8; seal; echo "0, 9000" > $S/smi.out
out=$(run_guard 0 'card_clear t'); check "refuse: card holds memory" 2 "GPU0 holds 9000 MiB" "$out" $?
sandbox c9; seal; echo "4, 368" > $S/smi.out
out=$(run_guard 0 'card_clear t'); check "refuse: card missing from nvidia-smi" 2 "GPU0 holds \? MiB" "$out" $?
sandbox c10; seal; rm -rf $S/leases
out=$(run_guard 0 'card_clear t'); check "refuse: lease root missing" 2 "lease root .* missing" "$out" $?
sandbox c11; seal; mkdir -p $S/leases/K3-unrelated; touch $S/leases/K3-unrelated
out=$(run_guard 0 'card_clear t'); check "clear: a fresh NON-D1 lease does not block" 0 "CARD CLEAR" "$out" $?

# ---- GPU6 slot and other cards
sandbox g1; echo "6, 5" > $S/smi.out; out=$(SLOT=$S/q/gpu6.slot.p2 run_guard 6 'card_clear t'); check "refuse: GPU6 without slot file" 2 "slot file .*gpu6.slot.p2 missing" "$out" $?
sandbox g2; echo "6, 5" > $S/smi.out; touch $S/q/gpu6.slot.p2
out=$(SLOT=$S/q/gpu6.slot.p2 run_guard 6 'card_clear t'); check "clear: GPU6 with slot file (no D1 checks there)" 0 "CARD CLEAR \[t\] GPU6 5 MiB$" "$out" $?
sandbox g3; echo "6, 30020" > $S/smi.out; touch $S/q/gpu6.slot.p2
out=$(SLOT=$S/q/gpu6.slot.p2 run_guard 6 'card_clear t'); check "refuse: GPU6 slot but K3 still on the card" 2 "GPU6 holds 30020 MiB" "$out" $?
sandbox g4; seal; out=$(run_guard 2 'card_clear t'); check "refuse: a P40" 2 "GPU2 is not a card of this lane" "$out" $?

# ---- gate_card: the re-check AFTER the MemAvailable wait (the review finding)
sandbox w1; seal; printf 'MemAvailable: 10485760 kB\n' > $S/meminfo; echo codex-d1-native-01a10f78-oct14-c37-recovery-r7 > $S/docker.out.after1
( sleep 2; printf 'MemAvailable: 104857600 kB\n' > $S/meminfo ) &
out=$(run_guard 0 'gate_card 60 p2'); rc=$?; wait
check "gate: D1 attempt appears DURING the mem wait -> refused after it, nothing past the gate" 2 "\[p2:after-mem-wait\] D1 container codex-d1-native-01a10f78-oct14-c37-recovery-r7" "$out" $rc
echo "$out" | grep -q "CARD CLEAR \[p2:before-mem-wait\]" && echo "$out" | grep -q "GiB < 60; waiting" && ! echo "$out" | grep -q SNIPPET-END \
  && { PASS=$((PASS + 1)); echo "PASS gate: it did pass before the wait, did wait, and stopped"; } \
  || { FAIL=$((FAIL + 1)); echo "FAIL gate: before/wait/stop sequence"; echo "$out" | sed 's/^/    | /'; }
sandbox w2; seal '-2 hours'; printf 'MemAvailable: 10485760 kB\n' > $S/meminfo; echo "4, 368" > $S/smi.out
( sleep 2; touch $S/leases/D1-native-01a10f78-oct14-c37-recovery-r6/live.json; printf 'MemAvailable: 104857600 kB\n' > $S/meminfo ) &
out=$(run_guard 4 'gate_card 60 p2'); rc=$?; wait
check "gate: a D1 lease is touched during the wait (container not up yet) -> refused" 2 "\[p2:after-mem-wait\] .*live.json is newer than the D1 seal marker" "$out" $rc
sandbox w3; seal; printf 'MemAvailable: 10485760 kB\n' > $S/meminfo; echo "0, 368" > $S/smi.out
( sleep 2; echo "0, 6000" > $S/smi.out; printf 'MemAvailable: 104857600 kB\n' > $S/meminfo ) &
out=$(run_guard 0 'gate_card 60 p2'); rc=$?; wait
check "gate: someone loads the card during the wait -> refused" 2 "\[p2:after-mem-wait\] GPU0 holds 6000 MiB" "$out" $rc
sandbox w4; seal; out=$(run_guard 0 'gate_card 60 p2'); check "gate: clear both times when nothing changes" 0 "CARD CLEAR \[p2:after-mem-wait\].*" "$out" $?
echo "$out" | grep -q SNIPPET-END && { PASS=$((PASS + 1)); echo "PASS gate: returns to the caller when clear"; } || { FAIL=$((FAIL + 1)); echo "FAIL gate: did not return"; }
sandbox w5; seal; printf 'MemAvailable: 10485760 kB\n' > $S/meminfo
out=$(WAITMAX=3 run_guard 0 'gate_card 60 p2'); check "gate: MemAvailable never recovers -> refused at the cap" 2 "MemAvailable stayed below 60 GiB" "$out" $?
sandbox w6; touch $S/q/gpu6.slot.p11; printf 'MemAvailable: 10485760 kB\n' > $S/meminfo; echo "6, 5" > $S/smi.out
( sleep 2; rm -f $S/q/gpu6.slot.p11; printf 'MemAvailable: 104857600 kB\n' > $S/meminfo ) &
out=$(SLOT=$S/q/gpu6.slot.p11 run_guard 6 'gate_card 40 p11'); rc=$?; wait
check "gate: GPU6 slot withdrawn during the wait -> refused" 2 "\[p11:after-mem-wait\] GPU6 slot file" "$out" $rc

# ---- the drivers: the gate precedes every GPU job, and every later GPU job re-checks
P2=$HERE/run_p2_v133.sh; P11=$HERE/run_p11_omni1_v110.sh
first() { grep -n -- "$2" "$1" | head -1 | cut -d: -f1; }
order_ok() { local a b; a=$(first "$1" "$2"); b=$(first "$1" "$3"); [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]; }
for pair in "gate_card 60 p2|^C4=\$(cache_for" "gate_card 60 p2|^run_cell L4096-mb8" "card_clear \"p2:target-cache\"|^C4=\$(cache_for" \
            "card_clear \"p2:start\"|^exec 9>" "^\. \$G|card_clear \"p2:start\""; do
  order_ok $P2 "${pair%%|*}" "${pair#*|}" && { PASS=$((PASS + 1)); echo "PASS p2 order: ${pair%%|*} before ${pair#*|}"; } \
    || { FAIL=$((FAIL + 1)); echo "FAIL p2 order: ${pair%%|*} before ${pair#*|}"; }
done
awk '/^cell\(\) \{/,/^\}/' $P2 | grep -q 'card_clear "p2:cell $tag"' && awk '/^cell\(\) \{/,/^\}/' $P2 | grep -B200 '\$PY \$T train' | grep -q card_clear \
  && { PASS=$((PASS + 1)); echo "PASS p2: every cell (incl. the OOM retry) re-checks the card before train"; } || { FAIL=$((FAIL + 1)); echo "FAIL p2 cell re-check"; }
[ "$(grep -c 'docker ps' $P2)" = 0 ] && [ "$(grep -c 'docker ps' $P11)" = 0 ] \
  && { PASS=$((PASS + 1)); echo "PASS drivers: no private D1 check left (the guard is the only one)"; } || { FAIL=$((FAIL + 1)); echo "FAIL drivers still carry their own docker ps"; }
for pair in "gate_card 40 p11|recurrence-parity \"\${COMMON" "card_clear \"p11:e2-parity\"|\$T parity \"\${COMMON" "card_clear \"p11:read\"|\$T read \"\${COMMON" \
            "card_clear \"p11:start\"|^exec 9>"; do
  order_ok $P11 "${pair%%|*}" "${pair#*|}" && { PASS=$((PASS + 1)); echo "PASS p11 order: ${pair%%|*} before ${pair#*|}"; } \
    || { FAIL=$((FAIL + 1)); echo "FAIL p11 order: ${pair%%|*} before ${pair#*|}"; }
done
for f in $P2 $P11; do
  ! grep -q '@@WANT' $f && { PASS=$((PASS + 1)); echo "PASS $(basename $f): pins filled"; } || { FAIL=$((FAIL + 1)); echo "FAIL $(basename $f): unfilled pin"; }
  want=$(grep -o 'WANT_G=[0-9a-f]\{64\}' $f | cut -d= -f2)
  [ "$want" = "$(sha256sum $GUARD | cut -c1-64)" ] && { PASS=$((PASS + 1)); echo "PASS $(basename $f): guard pin = this guard"; } \
    || { FAIL=$((FAIL + 1)); echo "FAIL $(basename $f): guard pin ${want:0:16} != $(sha256sum $GUARD | cut -c1-16)"; }
done
bash -n $P2 && bash -n $P11 && bash -n $GUARD && { PASS=$((PASS + 1)); echo "PASS bash -n on both drivers and the guard"; } || { FAIL=$((FAIL + 1)); echo "FAIL bash -n"; }

echo "$PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
