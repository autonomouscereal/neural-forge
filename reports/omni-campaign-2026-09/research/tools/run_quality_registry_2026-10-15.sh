#!/usr/bin/env bash
# run_quality_registry_2026-10-15.sh -- quality scorer v1 over the registry + D2 (research lane, Claude, 2026-10-15).
#
# CPU only (CUDA_VISIBLE_DEVICES empty), nice 15 / idle I/O.  Reads the sealed corpus files, writes ONLY under
# $R/data/.  A watchdog kills the scorer if MemAvailable drops below 66 GiB (K3's identity gate needs >= 70 GiB; the
# scorer's own peak is recorded by /usr/bin/time -v).  Step 1 is a 2,000-row smoke (screen build + timing), step 2 the
# full run.  QUALITY_OUT=<dir> SKIP_SMOKE=1 re-run only the full pass into another directory.
#
#   setsid nohup bash run_quality_registry_2026-10-15.sh > $R/data/logs/quality_registry.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
R=/mnt/lexar/omni-campaign/research
PY=/home/cereal/model-surgery-lab/venv/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=$OC/distill/tools CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
FLOOR_KIB=$((66 * 1048576))
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

run_watched() {  # run_watched <label> <out-dir> <args...>
  local label=$1 out=$2; shift 2
  mkdir -p "$out"
  nice -n 15 ionice -c3 /usr/bin/time -v "$PY" "$R/tools/quality_scorer.py" registry --out-dir "$out" "$@" > "$out.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$avail" -lt "$FLOOR_KIB" ]; then
      stamp "$label: MemAvailable $((avail / 1048576)) GiB < 66 GiB: killing the scorer (K3 comes first)"
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; wait $pid; return 9
    fi
  done
  wait $pid; local rc=$?
  stamp "$label rc=$rc peak RSS $(grep 'Maximum resident' "$out.log" | awk '{print $NF}') kB"
  tail -3 "$out.log"
  return $rc
}

stamp "start; MemAvailable $(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 )) GiB"
OUT=${QUALITY_OUT:-$R/data/quality-v1-registry-2026-10-15}
if [ -z "${SKIP_SMOKE:-}" ]; then run_watched smoke "$R/data/quality-v1-smoke-2026-10-15" --limit 2000 || { stamp "smoke failed"; exit 1; }; fi
run_watched full "$OUT" || { stamp "full run failed"; exit 1; }
sha256sum "$OUT/"*.json* "$OUT/minhash.npz"
stamp "done"
