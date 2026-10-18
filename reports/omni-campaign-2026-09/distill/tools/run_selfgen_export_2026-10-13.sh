#!/usr/bin/env bash
# Self-gen lane, step 1: the parent's tensor-map dry run (P9 for the parent) and its
# bf16 -> Q4_K_M GGUF export, written to the DECLARED scratch on Lexar
# (STAGE_D_RULINGS "Transient scratch": never registered, plain rm after use).
#
# Uses the export lane's own exporter ($OC/export/export_hf_to_gguf.py, llama.cpp
# b10752 / b96806d9 built for sm_61;75;86) unchanged. CPU only, nice 15, <= 16 threads.
# The bf16 GGUF is deleted by `selfgen_sampler.py export-check --delete-full` after the
# Q4 is hashed and the parent's post-export digest is compared; not here.
#
#   setsid nohup bash run_selfgen_export_2026-10-13.sh > $OC/distill/selfgen/logs/export.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
PARENT=/home/cereal/models/Qwen3.8-27B-Uncensored
SCRATCH=/mnt/lexar/omni-campaign/scratch/selfgen-q38-2026-10-13
LLAMA=$OC/export/llamacpp
LOGS=$OC/distill/selfgen/logs
TAG=Q38-27BU-parent
# bf16 55.6 GB + Q4 ~16.8 GB + 15 GB floor for everyone else on Lexar
NEED_GB=88

mkdir -p "$SCRATCH" "$LOGS"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONPATH=$LLAMA/llama.cpp/gguf-py

echo "[$(date -u +%FT%TZ)] start; df:"; df -BG / /mnt/lexar
free_gb=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$NEED_GB" ]; then
  echo "REFUSED: /mnt/lexar has ${free_gb} GB free, the export needs ${NEED_GB} GB"; exit 3
fi

echo "[$(date -u +%FT%TZ)] tensor-map dry run"
nice -n 15 timeout 1800 "$PY" "$LLAMA/llama.cpp/convert_hf_to_gguf.py" --outtype bf16 --dry-run \
  --outfile "$SCRATCH/dryrun.bf16.gguf" "$PARENT" > "$LOGS/tensor_map_dryrun.log" 2>&1
echo "dry-run rc=$?"; ls -la "$SCRATCH"

echo "[$(date -u +%FT%TZ)] export"
nice -n 15 timeout 14400 "$PY" "$OC/export/export_hf_to_gguf.py" \
  --hf-dir "$PARENT" --out-dir "$SCRATCH" --tag "$TAG" \
  --llamacpp-root "$LLAMA" --outtype bf16 --quants Q4_K_M --threads 16 \
  --quantization-stage "selfgen-sampler-transient-U7 (parent Q4 for sampling only; logits rescored bf16)"
rc=$?
echo "[$(date -u +%FT%TZ)] export rc=$rc"; ls -la "$SCRATCH"; df -BG / /mnt/lexar
exit $rc
