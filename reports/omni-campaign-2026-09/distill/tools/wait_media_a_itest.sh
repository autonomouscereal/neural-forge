#!/usr/bin/env bash
# wait_media_a_itest.sh -- launch run_media_a_itest.sh on the first lane-legal P40 (3, then 2) that stays
# <= 2000 MiB for 3 checks 20 s apart; gives up after $3 seconds (default 3600). Lane media-a, 2026-10-12.
set -uo pipefail
SCRATCH="${1:?scratch}"; ORDER="${2:-3 2}"; LIMIT="${3:-3600}"; PATTERN="${4:-}"
TOOLS=/home/cereal/omni-campaign/distill/tools
start=$(date +%s)
declare -A calm
while [ $(( $(date +%s) - start )) -lt "$LIMIT" ]; do
  for g in $ORDER; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -le 2000 ]; then calm[$g]=$(( ${calm[$g]:-0} + 1 )); else calm[$g]=0; fi
    if [ "${calm[$g]}" -ge 3 ]; then
      echo "$(date -u +%FT%TZ) GPU$g calm (${used} MiB) x3: launching"
      "$TOOLS/run_media_a_itest.sh" "$g" "$SCRATCH" "$PATTERN" && exit 0
      calm[$g]=0
    fi
  done
  sleep 20
done
echo "$(date -u +%FT%TZ) gave up after ${LIMIT}s"; exit 1
