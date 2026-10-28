#!/usr/bin/env bash
set -euo pipefail

pid=3551402
expected_runtime=/home/cereal/model-surgery-lab/deployments/neural-forge-b65c81f5/experiments/run_lfm2_dynamic_catalog.py
expected_run=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0026-exact4x-h816-mgv3-r12-w13l-cold
cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
case "$cmd" in
  *"$expected_runtime"*"$expected_run"*) ;;
  *)
    echo "PID identity mismatch; refusing TERM" >&2
    exit 91
    ;;
esac

kill -TERM "$pid"
for i in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "W13L_STOPPED pid=$pid signal=TERM wait_seconds=$i"
    exit 0
  fi
  sleep 1
done

echo "W13L_STILL_LIVE pid=$pid after_seconds=30" >&2
exit 92
