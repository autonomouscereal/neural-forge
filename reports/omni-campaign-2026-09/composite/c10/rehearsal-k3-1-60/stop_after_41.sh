#!/bin/sh
# M4 (2026-10-10 ~19:05Z): end the rehearsal once the last K3-carrying trunk layer (41) is persisted.
# Layers 42-63 carry no K3 (no K3 L61-92 charts in this rehearsal); each K3-free group still costs ~17 min of
# streamed KDA capture/advance, so they are left for a later `STOP=63 sh launch.sh --resume`.
PID=$1
R=/home/cereal/omni-campaign/composite/c10/rehearsal-k3-1-60
while kill -0 $PID 2>/dev/null; do
  if [ -f $R/child.partial/L41.json ] && [ -f $R/child.partial/L41.safetensors ]; then
    sleep 5
    echo "$(date -u +%FT%TZ) L41 persisted; SIGTERM pid $PID (planned stop after the last K3 layer)" >> $R/logs/watchdog.log
    kill -TERM $PID
    exit 0
  fi
  sleep 20
done
