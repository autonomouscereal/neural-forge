#!/bin/bash
# test_d1_launch_guard.sh -- d1_launch.sh 1.1.1-superseded must refuse every weight-moving invocation (2026-10-14, Claude).
# Runs the launcher with a throw-away status file; never touches the real d1_launch.status, GPUs, locks or the ledger.
# usage: bash test_d1_launch_guard.sh [path/to/d1_launch.sh]
set -u
L=${1:-$(dirname "$0")/d1_launch.sh}
REAL_STATUS=$(dirname "$L")/d1_launch.status
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok() { echo "PASS $*"; pass=$((pass+1)); }
no() { echo "FAIL $*"; fail=$((fail+1)); }
before=$(sha256sum "$REAL_STATUS" 2>/dev/null | cut -c1-64)

bash -n "$L" && ok "syntax" || no "syntax"
grep -q '^LAUNCHER_VERSION=d1_launch-1.1.1-superseded$' "$L" && ok "version 1.1.1-superseded" || no "version"

n=0
for args in "" "--run" "--confirm" "--run --resume"; do
  n=$((n+1))
  # shellcheck disable=SC2086
  out=$(D1_LAUNCH_STATUS="$T/status" timeout 20 bash "$L" $args 2>&1); rc=$?
  if [ $rc -eq 4 ] && echo "$out" | grep -q "d1_launch REFUSED"; then ok "refuses '${args:-<no-arg>}' rc=4"
  else no "'${args:-<no-arg>}' rc=$rc out=$(echo "$out" | tail -1)"; fi
done
[ "$(grep -c 'd1_launch REFUSED' "$T/status" 2>/dev/null)" = "$n" ] && ok "status line per refusal ($n)" || no "status lines"
after=$(sha256sum "$REAL_STATUS" 2>/dev/null | cut -c1-64)
[ "$before" = "$after" ] && ok "real d1_launch.status untouched" || no "real d1_launch.status changed"
echo "RESULT d1_launch guard: $pass passed, $fail failed"
[ $fail -eq 0 ]
