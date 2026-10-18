#!/bin/bash
# k3cache lane: the cross-tokenizer target and the K3-side route probe on the sealed review parity cache
# (CPU only; the boundary map pairs k3_review_input_ids <-> qwen_review_input_ids; the K3-native review
# file has no Qwen side and is skipped by both).  No Q-text cache exists yet, so the probe is K3-only.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
OUT=$OC/distill/parity/k3cache-2026-10-12
C=$OC/composite/corpus
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 15 "$PY" "$T/k3_teacher_cache.py" xtok --cache-dir "$OUT/review-pass" --boundary-map "$C/boundary_map.json" \
  --k3-ids "$C/k3_review_input_ids.safetensors" --qwen-ids "$C/qwen_review_input_ids.safetensors" \
  --out "$OUT/xtok-review"
echo "XTOK_EXIT=$?"
nice -n 15 "$PY" "$T/k3_teacher_cache.py" probe --cache-dir "$OUT/review-pass" --boundary-map "$C/boundary_map.json" \
  --k3-ids "$C/k3_review_input_ids.safetensors" --qwen-ids "$C/qwen_review_input_ids.safetensors" \
  --row-meta "$C/row_meta.json" --route-key source --out "$OUT/probe_review_k3only.json"
echo "PROBE_EXIT=$?"
