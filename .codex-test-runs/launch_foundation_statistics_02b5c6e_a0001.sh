#!/usr/bin/env bash
set -euo pipefail

SHA=02b5c6e4e92bd5eb38619376e273c75136c1f5c1
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_statistics_20260721/02b5c6e/a0001
PIDFILE="$RUN/backfill.pid"
mkdir -p "$RUN"
test ! -e "$RUN/backfill-receipt.json"

if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup env \
  CUDA_VISIBLE_DEVICES=6 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONPATH="$CODE/src" \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=16 \
  /home/cereal/model-surgery-lab/venv/bin/python -u \
  "$CODE/experiments/backfill_lfm2_semantic_foundation_statistics.py" \
  --input-checkpoint-directory /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/checkpoint \
  --expected-input-artifact-sha256 517515e3525c107db180953dcde6c1f5dc2ab6af3bc0f50486e442d6bf69899f \
  --semantic-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --expected-semantic-cache-key 76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --foundation-receipt-json /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/foundation-receipt.json \
  --expected-foundation-receipt-sha256 7bffddca0fc90e026a62d625cff9f14b436955dbc799f9d8cdb804dc54e07afd \
  --artifact-store-directory "$RUN/statistics-store" \
  --backfill-receipt-json "$RUN/backfill-receipt.json" \
  --device cuda:0 \
  --dtype bfloat16 \
  >"$RUN/backfill.log" 2>&1 </dev/null &

pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
sleep 1
if kill -0 "$pid" 2>/dev/null; then
  echo "launched pid=$pid"
else
  echo "process exited during launch"
  cat "$RUN/backfill.log"
  exit 1
fi
