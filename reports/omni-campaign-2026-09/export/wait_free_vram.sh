#!/usr/bin/env bash
# Wait until a GPU has enough FREE memory, then exec the command that follows.
#
# Why: the 3090 run of the 256k sweep died at load with
#   `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15088.32 MiB on device 0:
#    cudaMalloc failed: out of memory`
# because another lane's calibration job (oc_run_frozen_bank.py, 9,218 MiB) was resident on
# GPU0 when our server started. That process is not ours to stop, so the only correct move
# is to wait for the card, not to take it.
#
# This waits. It never signals, kills or reconfigures anything, and if the card never frees
# within the deadline it exits non-zero with a line saying so, having started nothing.
#
#   ./wait_free_vram.sh --gpu-index 0 --min-free-mib 20000 --deadline-s 10800 -- \
#       /path/to/python envelope_bench.py --tag ...
set -u

GPU_INDEX=0
MIN_FREE_MIB=20000
DEADLINE_S=10800
POLL_S=60

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-index)     GPU_INDEX="$2"; shift 2 ;;
    --min-free-mib)  MIN_FREE_MIB="$2"; shift 2 ;;
    --deadline-s)    DEADLINE_S="$2"; shift 2 ;;
    --poll-s)        POLL_S="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "nothing to run after --" >&2
  exit 2
fi

started=$(date +%s)
while : ; do
  free_mib=$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free \
             --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  now=$(date +%s)
  elapsed=$(( now - started ))
  if [ -n "${free_mib}" ] && [ "${free_mib}" -ge "${MIN_FREE_MIB}" ] 2>/dev/null; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu${GPU_INDEX} free=${free_mib} MiB >= ${MIN_FREE_MIB}; starting after ${elapsed}s of waiting"
    exec "$@"
  fi
  if [ "${elapsed}" -ge "${DEADLINE_S}" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu${GPU_INDEX} still only ${free_mib} MiB free after ${elapsed}s; giving up WITHOUT starting anything"
    exit 1
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu${GPU_INDEX} free=${free_mib} MiB (< ${MIN_FREE_MIB}); waited ${elapsed}s"
  sleep "${POLL_S}"
done
