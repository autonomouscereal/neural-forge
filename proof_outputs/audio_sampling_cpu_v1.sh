set -u
cd /home/cereal/omni-campaign/distill/codex-audio-sampling-tests-modality-01a1326f-v1 && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 CHIMERA_D3_BASELINE=/home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py CHIMERA_STREAMED_BASELINE=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py STREAMED_KD_TRAINER_PATH=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py nice -n15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -q -p no:cacheprovider --basetemp /home/cereal/omni-campaign/distill/codex-audio-sampling-tests-modality-01a1326f-v1/pytest-tmp tests/test_chimera_audio_sampling.py
rc=$?
printf "%s\n" "$rc" > /home/cereal/omni-campaign/distill/codex-audio-sampling-tests-modality-01a1326f-v1/exitcode
exit "$rc"
