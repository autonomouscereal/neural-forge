#!/usr/bin/env bash
set -euo pipefail
rm -f /tmp/audit_lfm2_qdeim_rank.log /tmp/audit_lfm2_qdeim_rank.pid
nohup env MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  /home/cereal/model-surgery-lab/venv/bin/python /tmp/audit_lfm2_qdeim_rank.py \
  >/tmp/audit_lfm2_qdeim_rank.log 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >/tmp/audit_lfm2_qdeim_rank.pid
sleep 2
kill -0 "$pid"
printf 'pid=%s\n' "$pid"
