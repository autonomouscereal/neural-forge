#!/usr/bin/env bash
# Toolcall lane, step 1: re-export the PARENT's own Q4_K_M for the tool-use sampler.
#
# The self-gen lane's Q4 (sha 76c1e4c6...) was removed from declared scratch after its run
# (STAGE_D_RULINGS "Transient scratch"). The only Q4 left on the box,
# /home/cereal/models/gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf (sha 4c5e2db0...), is a
# third-party conversion (JonathanColetti/Qwen3.8-27B-Uncensored-GGUF @45d0fc0a), not the
# campaign exporter's output, so its weights are not tied to the parent directory digest.
# U7 says the sampler is "the parent's Q4_K_M": this re-runs the campaign exporter
# unchanged ($OC/export/export_hf_to_gguf.py, llama.cpp b10752 / b96806d9, sm_61;75;86)
# into a NEW declared scratch directory, then `selfgen_sampler.py export-check` re-hashes
# the Q4, re-digests the parent (before == after) and removes the bf16 GGUF.
#
# Host-RAM gate: waits (never competes) until MemAvailable >= 40 GiB. CPU only, nice 19,
# 12 threads (D1 r4 shares the host).
#
#   setsid nohup bash run_toolcall_export_2026-10-14.sh > $OC/distill/toolcall/logs/export.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
T=$OC/distill/tools
PARENT=/home/cereal/models/Qwen3.8-27B-Uncensored
SCRATCH=/mnt/lexar/omni-campaign/scratch/toolcall-q38-2026-10-14
LLAMA=$OC/export/llamacpp
TC=$OC/distill/toolcall
LOGS=$TC/logs
TAG=Q38-27BU-parent
# bf16 55.6 GB + Q4 ~16.8 GB + 15 GB floor for everyone else on Lexar
NEED_GB=88
NEED_MEM_GIB=40

mkdir -p "$SCRATCH" "$LOGS" "$TC/receipts"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PYTHONPATH=$LLAMA/llama.cpp/gguf-py

echo "[$(date -u +%FT%TZ)] start; df:"; df -BG / /mnt/lexar
free_gb=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$NEED_GB" ]; then
  echo "REFUSED: /mnt/lexar has ${free_gb} GB free, the export needs ${NEED_GB} GB"; exit 3
fi
while :; do
  avail_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  avail_gib=$((avail_kib / 1048576))
  [ "$avail_gib" -ge "$NEED_MEM_GIB" ] && break
  echo "[$(date -u +%FT%TZ)] waiting: MemAvailable ${avail_gib} GiB < ${NEED_MEM_GIB} GiB"; sleep 120
done
echo "[$(date -u +%FT%TZ)] MemAvailable ${avail_gib} GiB"

echo "[$(date -u +%FT%TZ)] export"
nice -n 19 timeout 14400 "$PY" "$OC/export/export_hf_to_gguf.py" \
  --hf-dir "$PARENT" --out-dir "$SCRATCH" --tag "$TAG" \
  --llamacpp-root "$LLAMA" --outtype bf16 --quants Q4_K_M --threads 12 \
  --quantization-stage "toolcall-sampler-transient-U7 (parent Q4 for tool-use sampling only; logits rescored bf16)"
rc=$?
echo "[$(date -u +%FT%TZ)] export rc=$rc"; ls -la "$SCRATCH"; df -BG / /mnt/lexar
[ "$rc" -eq 0 ] || exit "$rc"

echo "[$(date -u +%FT%TZ)] export-check (+ delete the bf16 GGUF from declared scratch)"
nice -n 15 "$PY" "$T/selfgen_sampler.py" export-check \
  --export-receipt "$SCRATCH/export-receipt-$TAG.json" --parent-dir "$PARENT" \
  --delete-full --scratch-root /mnt/lexar/omni-campaign/scratch \
  --out "$TC/receipts/export_check_${TAG}_toolcall.json"
crc=$?
echo "[$(date -u +%FT%TZ)] export-check rc=$crc"; ls -la "$SCRATCH"; df -BG / /mnt/lexar
exit $crc
