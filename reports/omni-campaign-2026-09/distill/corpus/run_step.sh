#!/bin/bash
# usage: run_step.sh <step>   (distill_corpus_build 1.0.0 lane; CPU only)
export CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src PYTHONPATH=/home/cereal/omni-campaign/distill/tools
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=12 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=""
cd /home/cereal/omni-campaign/distill/tools
exec nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python distill_corpus_build.py "$@"
