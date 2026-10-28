set -u
cd /home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v3
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider --basetemp /home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v3/session-focused-v1/tmp tests/test_chimera_audio_cache.py tests/test_chimera_recovery_h3_audio_probe.py -q
rc=$?
printf "%s\n" "$rc" > /home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v3/session-focused-v1/exitcode
exit "$rc"
