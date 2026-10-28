#!/usr/bin/env bash
set -uo pipefail
umask 077
ulimit -v 4194304
export CUDA_VISIBLE_DEVICES=""
cd /home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1
timeout 1800s nice -n 10 python3 capsule/speech_teacher_comparison.py download-candidates --plan capsule/manifest.json --output parents
result=$?
printf '%s\n' "$result" > exitcode
exit "$result"
