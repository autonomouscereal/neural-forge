set -u
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0025-exact4x-h816-mgv3-r12-w13k-cold
PID=3262674
printf 'UTC '; date -u -Ins
printf 'PROCESS\n'
ps -p "$PID" -o pid=,ppid=,stat=,etime=,time=,%cpu=,%mem=,rss=,cmd= || true
PPID_VALUE=$(ps -p "$PID" -o ppid= 2>/dev/null | tr -d ' ' || true)
if [ -n "$PPID_VALUE" ]; then
  printf 'PARENT\n'
  ps -p "$PPID_VALUE" -o pid=,ppid=,stat=,etime=,time=,%cpu=,%mem=,rss=,cmd= || true
  printf 'PARENT_CHILDREN\n'
  ps --ppid "$PPID_VALUE" -o pid=,ppid=,stat=,etime=,time=,%cpu=,%mem=,rss=,cmd= || true
fi
printf 'PROC_STATUS\n'
grep -E '^(Name|State|Tgid|Pid|PPid|Threads|VmPeak|VmSize|VmHWM|VmRSS|VmSwap|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):' "/proc/$PID/status" 2>/dev/null || true
printf 'ROOT_ARTIFACTS\n'
find "$RUN" -maxdepth 2 -type f -not -path '*/snapshots/*' -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n' 2>/dev/null | sort | tail -60 || true
printf 'FAILURE_MATCHES\n'
grep -iE '"stage":"(failure|failed|error)"|traceback|exception|killed|out of memory|cuda.*error' "$RUN/run.log" 2>/dev/null | tail -40 || true
printf 'LAST_LOG\n'
tail -12 "$RUN/run.log" 2>/dev/null || true
printf 'KERNEL_OOM\n'
journalctl -k --since '2026-08-05 15:25:00' --no-pager 2>/dev/null | grep -iE 'oom|out of memory|killed process|segfault|python' | tail -60 || true
dmesg 2>/dev/null | grep -iE 'oom|out of memory|killed process|segfault|python' | tail -60 || true
printf 'MEMORY\n'
free -h || true
printf 'CAPSULES\n'
find "$RUN/restart-capsules" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n' 2>/dev/null | sort | tail -20 || true
