#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "/home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1/cpu-canary-v1"
overall=0
for mode in asr_candidate asr_baseline; do
 timeout --signal=TERM --kill-after=10 630 nice -n 15 /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/speech-runtimes-v1/asr/venv/bin/python -B capsule-asr-v2/speech_teacher_canary.py "$mode" --request capsule-asr-v2/asr-request.json --output "$mode-v2" > "$mode-v2.log" 2>&1
 code=$?
 printf '%s\n' "$code" > "$mode-v2-exitcode"
 if [ "$code" -ne 0 ]; then overall=1; fi
done
printf '%s\n' "$overall" > asr-chain-v2-exitcode
exit "$overall"
