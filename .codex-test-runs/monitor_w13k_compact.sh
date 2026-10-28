set -u
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0025-exact4x-h816-mgv3-r12-w13k-cold
PID=3262674
printf 'UTC '; date -u -Ins
ps -p "$PID" -o pid=,stat=,etime=,time=,%cpu=,%mem=,rss= || true
if [ -r "/proc/$PID/io" ]; then
  awk '/^(rchar|wchar|read_bytes|write_bytes):/ {printf "%s=%s ",$1,$2} END {print ""}' "/proc/$PID/io"
  printf 'WCHAN '; cat "/proc/$PID/wchan" 2>/dev/null || true; printf '\n'
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | sed -n '1p;5p' || true
if [ -f "$RUN/run.log" ]; then
  stat -c 'RUN_LOG %y %s' "$RUN/run.log"
  printf 'STAGES\n'
  tail -12 "$RUN/run.log" | sed -n 's/.*"stage":"\([^"]*\)".*"timestamp_utc":"\([^"]*\)".*/\2 \1/p'
  printf 'LAST\n'; tail -1 "$RUN/run.log"
fi
printf 'CAPSULES\n'
find "$RUN/restart-capsules" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n' 2>/dev/null | sort | tail -12 || true
