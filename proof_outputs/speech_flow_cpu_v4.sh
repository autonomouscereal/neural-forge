set -u
cd /home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v4 && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 CHIMERA_D3_BASELINE=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py CHIMERA_STREAMED_BASELINE=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py STREAMED_KD_TRAINER_PATH=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider --basetemp /home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v4/pytest-tmp tests/test_chimera_speech_flow.py -k speaking_sampler -q
rc=$?
printf "%s\n" "$rc" > /home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v4/exitcode
exit "$rc"
