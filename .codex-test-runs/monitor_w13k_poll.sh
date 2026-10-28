set -u
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0025-exact4x-h816-mgv3-r12-w13k-cold
PID=3262674
printf 'REMOTE_TIME '; date -Ins
printf 'UTC_TIME '; date -u -Ins
ps -p "$PID" -o pid=,stat=,etime=,etimes=,%cpu=,%mem=,rss=,lstart= || true
if [ -r "/proc/$PID/io" ]; then
  awk '/^(rchar|wchar|read_bytes|write_bytes):/ {printf "%s=%s ",$1,$2} END {print ""}' "/proc/$PID/io"
  printf 'WCHAN '
  cat "/proc/$PID/wchan" 2>/dev/null || true
  printf '\n'
fi
printf 'TOP_THREADS\n'
ps -L -p "$PID" -o tid=,stat=,pcpu=,time=,comm= --sort=-pcpu 2>/dev/null | head -8 || true
printf 'GPU\n'
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | sed -n '1p;5p' || true
printf 'ARTIFACTS\n'
if [ -d "$RUN" ]; then
  find "$RUN" -maxdepth 2 -type f -not -path '*/snapshots/*' -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n' | sort | tail -20
  printf 'SNAPSHOT_SUMMARY\n'
  find "$RUN/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true
  du -sh "$RUN/snapshots" 2>/dev/null || true
  printf 'CAPSULES\n'
  find "$RUN/restart-capsules" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n' 2>/dev/null | sort | tail -20 || true
  printf 'RUN_LOG_STAGES\n'
  tail -40 "$RUN/run.log" 2>/dev/null | sed -n 's/.*"stage":"\([^"]*\)".*"timestamp_utc":"\([^"]*\)".*/\2 \1/p' || true
  printf 'RUN_LOG_TAIL\n'
  tail -3 "$RUN/run.log" 2>/dev/null || true
  printf 'PHASE_LOG_TAIL\n'
  tail -20 "$RUN/balanced-basis-phase-events.jsonl" 2>/dev/null || true
else
  echo MISSING
fi
