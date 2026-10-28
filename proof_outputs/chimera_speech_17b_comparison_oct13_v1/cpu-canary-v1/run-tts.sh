#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "/home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1/cpu-canary-v1"
timeout --signal=TERM --kill-after=10 630 nice -n 15 /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/speech-runtimes-v1/tts/venv/bin/python -B capsule/speech_teacher_canary.py tts --request capsule/request.json --output tts-candidate > tts.log 2>&1
code=$?
printf '%s\n' "$code" > tts-exitcode
exit "$code"
