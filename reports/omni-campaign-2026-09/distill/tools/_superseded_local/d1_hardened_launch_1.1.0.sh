#!/bin/bash
# d1_hardened_launch.sh -- start D1 (d1_hardened_entry.py -> d1_trunk_recovery.py) as a transient systemd --user
# service.  Lane d1, Claude, 2026-10-14 (state doc step 4a).  Used by $Q/d1_takeover.sh and by the fault-injection
# test, so the test exercises the production launch path.
#
# Why systemd --user: the service belongs to user@1000.service, not to the SSH session that started it (an SSH drop
# cannot kill it); its cgroup gives MemoryMax and a memory.current the trainer logs per layer; the unit keeps its
# exit result; Restart=on-failure brings it back after an OOM kill / CUDA OOM / crash, and the trainer resumes from
# the boundary cache.  RestartPreventExitStatus=3: a functional refusal stays down for a human.  ExecStopPost
# writes the exit state into runtime-status.json even after SIGKILL.  No allocator, no Docker, no PG, no Windows.
#
# usage: d1_hardened_launch.sh --unit NAME --run-dir DIR [--memory-max 128G] [--restart-sec 300]
#          [--cuda-visible GPU-uuid,GPU-uuid | ""] [--lock FILE]... [--lock-wait-s 1800] [--threads 16]
#          [--python PY] [--tools DIR] [--chimera TREE] [--start-limit-burst 6] [--heartbeat-s 60]
#          -- <d1_trunk_recovery args>
# Writes DIR/runtime-status.json (the entry), DIR/runtime-log.txt (stdout+stderr, appended), DIR/launch.json.
# Exit: 0 started (prints the unit, MainPID); 5 the unit is already active; 2 bad usage / systemd-run failed.
set -u
LAUNCH_VERSION=d1_hardened_launch-1.1.0   # 1.1.0: --gpu-uuid from --cuda-visible, --heartbeat-s
UNIT=""; RUN=""; MEMMAX=128G; RESTART_SEC=300; CUDA_VIS=__unset__; LOCKS=(); LOCK_WAIT=1800; THREADS=16
PY=/home/cereal/model-surgery-lab/venv/bin/python
T=/home/cereal/omni-campaign/distill/tools
EF=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9
BURST=6
HEARTBEAT=60
while [ $# -gt 0 ]; do
  case "$1" in
    --unit) UNIT=$2; shift 2 ;;
    --run-dir) RUN=$2; shift 2 ;;
    --memory-max) MEMMAX=$2; shift 2 ;;
    --restart-sec) RESTART_SEC=$2; shift 2 ;;
    --cuda-visible) CUDA_VIS=$2; shift 2 ;;
    --lock) LOCKS+=("$2"); shift 2 ;;
    --lock-wait-s) LOCK_WAIT=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --tools) T=$2; shift 2 ;;
    --chimera) EF=$2; shift 2 ;;
    --start-limit-burst) BURST=$2; shift 2 ;;
    --heartbeat-s) HEARTBEAT=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "$LAUNCH_VERSION: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$UNIT" ] && [ -n "$RUN" ] && [ $# -gt 0 ] || { echo "$LAUNCH_VERSION: --unit, --run-dir and trainer args after -- are required" >&2; exit 2; }
[ "$CUDA_VIS" != __unset__ ] || { echo "$LAUNCH_VERSION: --cuda-visible is required (\"\" for CPU)" >&2; exit 2; }
for f in "$PY" "$T/d1_hardened_entry.py" "$T/d1_trunk_recovery.py"; do
  [ -e "$f" ] || { echo "$LAUNCH_VERSION: missing $f" >&2; exit 2; }
done
case "$RUN$T$PY$EF" in *" "*) echo "$LAUNCH_VERSION: paths with spaces are not supported (ExecStopPost)" >&2; exit 2 ;; esac
mkdir -p "$RUN"
STATUS=$RUN/runtime-status.json
LOG=$RUN/runtime-log.txt

state=$(systemctl --user show -p ActiveState --value "$UNIT.service" 2>/dev/null)
case "$state" in
  active|activating|reloading|deactivating) echo "$LAUNCH_VERSION: $UNIT.service is $state; refusing a second D1" >&2; exit 5 ;;
esac
systemctl --user reset-failed "$UNIT.service" 2>/dev/null
systemctl --user stop "$UNIT.service" 2>/dev/null   # an inactive leftover with RemainAfterExit is cleared

LOCK_ARGS=()
for l in "${LOCKS[@]}"; do LOCK_ARGS+=(--lock "$l"); done
IFS=',' read -r -a VIS <<< "$CUDA_VIS"
for g in "${VIS[@]}"; do case "$g" in GPU-*) LOCK_ARGS+=(--gpu-uuid "$g") ;; esac; done   # wait out foreign compute
systemd-run --user --unit="$UNIT" --description="D1 hardened ($LAUNCH_VERSION)" \
  --working-directory="$T" \
  -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
  -p Restart=on-failure -p RestartSec="$RESTART_SEC" -p RestartPreventExitStatus=3 \
  -p StartLimitIntervalSec=86400 -p StartLimitBurst="$BURST" \
  -p KillMode=mixed -p TimeoutStopSec=300 -p Nice=10 \
  -p StandardOutput="append:$LOG" -p StandardError="append:$LOG" \
  -p ExecStopPost="$PY $T/d1_hardened_entry.py --finalize --status $STATUS" \
  --setenv=CHIMERA_SRC="$EF/src" --setenv=CHIMERA_OMNI_SRC="$EF" --setenv=PYTHONPATH="$T:$EF/src" \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES="$CUDA_VIS" \
  --setenv=OMP_NUM_THREADS="$THREADS" --setenv=MKL_NUM_THREADS="$THREADS" --setenv=OPENBLAS_NUM_THREADS="$THREADS" \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv=PYTHONUNBUFFERED=1 \
  "$PY" "$T/d1_hardened_entry.py" --status "$STATUS" "${LOCK_ARGS[@]}" --lock-wait-s "$LOCK_WAIT" --heartbeat-s "$HEARTBEAT" -- "$@" \
  || { echo "$LAUNCH_VERSION: systemd-run failed" >&2; exit 2; }
sleep 2
MAINPID=$(systemctl --user show -p MainPID --value "$UNIT.service")
INV=$(systemctl --user show -p InvocationID --value "$UNIT.service")
"$PY" - "$RUN/launch.json" "$UNIT" "$MAINPID" "$INV" "$LAUNCH_VERSION" "$MEMMAX" "$CUDA_VIS" "$@" <<'EOF'
import json, sys, time
path, unit, pid, inv, ver, mem, cuda, *argv = sys.argv[1:]
doc = {"launcher": ver, "unit": unit + ".service", "main_pid": int(pid or 0), "invocation_id": inv,
       "memory_max": mem, "cuda_visible_devices": cuda, "trainer_argv": argv,
       "launched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
open(path + ".tmp", "w").write(json.dumps(doc, indent=1))
import os; os.replace(path + ".tmp", path)
EOF
echo "$LAUNCH_VERSION: started $UNIT.service MainPID $MAINPID invocation $INV; status $STATUS; log $LOG"
exit 0
