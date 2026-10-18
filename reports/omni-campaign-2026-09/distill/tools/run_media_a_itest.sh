#!/usr/bin/env bash
# run_media_a_itest.sh -- stage_c_media_a 1.0.2 integration tests on ONE P40 (lane media-a, Claude, 2026-10-12).
# Usage: run_media_a_itest.sh <gpu index> <scratch dir> [unittest -k pattern]   (detaches; poll the log and the allocator receipt)
# Refuses on a card above 2000 MiB, on GPU0/4/5/6 (step-1 / reserved), and when the root disk is under 60 GB free.
set -euo pipefail
GPU="${1:?gpu index}"
SCRATCH="${2:?scratch dir}"
PATTERN="${3:-}"
KARG=""; [ -n "$PATTERN" ] && KARG="-k '$PATTERN'"
OC=/home/cereal/omni-campaign
IMG=comfyui:codex-minimax-h3-runtime-env-r5-20260808
TOOLS=$OC/distill/tools
ALLOC=$OC/composite/receipts/allocator/media-a-itest-gpu${GPU}-claude.json
case "$GPU" in 1|2|3) ;; *) echo "refuse: GPU$GPU is not a lane-media-a P40 (1-3)"; exit 2;; esac
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU")
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$NAME" = "Tesla P40" ] || { echo "refuse: GPU$GPU is $NAME"; exit 2; }
[ "$USED" -le 2000 ] || { echo "refuse: GPU$GPU has $USED MiB in use"; exit 2; }
[ "$FREE_GB" -ge 60 ] || { echo "refuse: root has ${FREE_GB} GB free (< 60)"; exit 2; }
mkdir -p "$SCRATCH"
LOG=$SCRATCH/itest.log
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"lane":"media-a itest (stage_c_media_a 1.0.2, itest_stage_c_media_a.py)","gpu":%s,"uuid":"%s","card":"%s","by":"claude media-a lane","started_utc":"%s","used_mib_at_start":%s,"log":"%s","state":"running"}\n' \
  "$GPU" "$UUID" "$NAME" "$STARTED" "$USED" "$LOG" > "$ALLOC"
setsid nohup bash -c "
  docker run --rm --name media_a_itest_gpu${GPU} --user $(id -u):$(id -g) --cpus 16 \
    --gpus '\"device=${UUID}\"' -e NVIDIA_VISIBLE_DEVICES=${UUID} -e CUDA_VISIBLE_DEVICES=0 \
    -e HOME=/tmp -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
    -e OC=$OC -e ITEST_DIR=$SCRATCH -w $TOOLS \
    -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro -v $OC:$OC \
    --entrypoint nice $IMG -n 15 /opt/environments/python/comfyui/bin/python -m unittest -v $KARG itest_stage_c_media_a
  RC=\$?
  printf '{\"lane\":\"media-a itest (stage_c_media_a 1.0.2, itest_stage_c_media_a.py)\",\"gpu\":%s,\"uuid\":\"%s\",\"card\":\"%s\",\"by\":\"claude media-a lane\",\"started_utc\":\"%s\",\"ended_utc\":\"%s\",\"exit\":%s,\"log\":\"%s\",\"state\":\"ended\"}\n' \
    $GPU $UUID '$NAME' $STARTED \$(date -u +%Y-%m-%dT%H:%M:%SZ) \$RC $LOG > $ALLOC
" > "$LOG" 2>&1 < /dev/null &
echo "launched itest on GPU$GPU ($UUID), log $LOG, allocator $ALLOC"
