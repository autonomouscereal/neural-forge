#!/bin/sh
# M4 watchdog (2026-10-10): protects the root-disk floor and host RAM for K3 while the c10 rehearsal runs.
# SIGTERMs the c10 python (layers already persisted under child.partial survive; resume later) when
#   root avail < ${ROOT_FLOOR_GB:-62} GB  or  MemAvailable < ${MEM_FLOOR_GB:-16} GB.
PID=$1
R=/home/cereal/omni-campaign/composite/c10/rehearsal-k3-1-60
LOG=$R/logs/watchdog.log
echo "$(date -u +%FT%TZ) watching pid $PID (root floor ${ROOT_FLOOR_GB:-62} GB, mem floor ${MEM_FLOOR_GB:-16} GB)" >> $LOG
while kill -0 $PID 2>/dev/null; do
  a=$(df --output=avail -B1G / | tail -1 | tr -d ' ')
  m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt "${ROOT_FLOOR_GB:-62}" ] || [ "$m" -lt "${MEM_FLOOR_GB:-16}" ]; then
    echo "$(date -u +%FT%TZ) TERM pid $PID: root avail ${a}G, MemAvailable ${m}G" >> $LOG
    kill -TERM $PID
    sleep 30
    kill -0 $PID 2>/dev/null && kill -KILL $PID
    exit 0
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) pid $PID exited; watchdog done" >> $LOG
