#!/usr/bin/env bash
set -euo pipefail

SHA=48e79136785da21bc497dae42e099e37b5f8b36d
SHORT=${SHA:0:7}
ARCHIVE=/tmp/neural-forge-$SHA.tar
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_interface_frontier_companion_20260722/$SHORT/a0001
COMPANION=$RUN/frontier-companion.json

test -f "$ARCHIVE"
test ! -e "$CODE"
test ! -e "$RUN"
mkdir -p "$CODE" "$RUN"
tar -xf "$ARCHIVE" -C "$CODE"

env \
  PYTHONPATH="$CODE/src" \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=16 \
  /home/cereal/model-surgery-lab/venv/bin/python -u \
  "$CODE/experiments/migrate_lfm2_semantic_foundation_statistics_frontier.py" \
  --legacy-statistics-artifact-directory /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_statistics_20260721/02b5c6e/a0002/statistics-store/42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a \
  --expected-legacy-statistics-artifact-sha256 42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a \
  --semantic-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --expected-semantic-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --legacy-foundation-receipt-json /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/foundation-receipt.json \
  --expected-legacy-foundation-receipt-sha256 7bffddca0fc90e026a62d625cff9f14b436955dbc799f9d8cdb804dc54e07afd \
  --expected-legacy-foundation-receipt-file-sha256 85e93babc331f5d5cc3f53c1b0ac930a0249fbdb25ee39ccf3e9a29996722a1e \
  --output-companion-json "$COMPANION" \
  >"$RUN/migration-report.json"

test -s "$COMPANION"
printf 'runtime=%s\ncompanion=%s\ncompanion_file_sha256=%s\ncompanion_sha256=%s\n' \
  "$CODE" \
  "$COMPANION" \
  "$(sha256sum "$COMPANION" | awk '{print $1}')" \
  "$(jq -r .companion_sha256 "$COMPANION")"
