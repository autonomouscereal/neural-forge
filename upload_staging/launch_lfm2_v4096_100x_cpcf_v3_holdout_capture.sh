#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 EXPECTED_EXTERNAL_RECEIPT_FILE_SHA256 EXPECTED_RECEIPT_SHA256 EXPECTED_CANDIDATE_STATE_SHA256" >&2
  exit 64
fi

expected_receipt_file_sha256="$1"
expected_receipt_sha256="$2"
expected_candidate_state_sha256="$3"
repository="/home/cereal/model-surgery-lab/neural-forge-src"
python="/home/cereal/model-surgery-lab/venv/bin/python"
checkpoint="/home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_folds_20260717/lfm2_230m_to_v4096_100x_cpcf_v3"
external_receipt="${checkpoint}.receipt.json"
tokenizer="/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2"
destination="/home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_holdout_candidates_20260718"
candidate="${destination}/lfm2_230m_to_v4096_100x_cpcf_v3.candidate.json"
capture_receipt="${destination}/lfm2_230m_to_v4096_100x_cpcf_v3.capture.receipt.json"

[[ "${expected_receipt_file_sha256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${expected_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${expected_candidate_state_sha256}" =~ ^[0-9a-f]{64}$ ]]
test -d "${checkpoint}"
test -f "${checkpoint}/cpcf_run_receipt.json"
test -f "${external_receipt}"
test ! -e "${candidate}"
test ! -e "${capture_receipt}"

observed_receipt_file_sha256="$(
  sha256sum "${external_receipt}" | awk '{print $1}'
)"
test "${observed_receipt_file_sha256}" = "${expected_receipt_file_sha256}"
if kill -0 2768715 2>/dev/null; then
  echo "CPCF fold process 2768715 is still live; capture refused." >&2
  exit 75
fi

mkdir -p "${destination}"
cd "${repository}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561"

exec "${python}" -u experiments/capture_lfm2_cpcf_holdout_candidate.py \
  --checkpoint-directory "${checkpoint}" \
  --target-tokenizer-directory "${tokenizer}" \
  --bridge-directory "${tokenizer}" \
  --cpcf-receipt "${external_receipt}" \
  --expected-cpcf-receipt-file-sha256 "${expected_receipt_file_sha256}" \
  --expected-cpcf-receipt-sha256 "${expected_receipt_sha256}" \
  --expected-manifest-sha256 "5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690" \
  --expected-target-tokenizer-fingerprint "5e76c6c0f9180ac69fae4729ecf2239100d7409fd148be14d80574340160675b" \
  --expected-checkpoint-tokenizer-fingerprint "5e76c6c0f9180ac69fae4729ecf2239100d7409fd148be14d80574340160675b" \
  --expected-target-tokenizer-semantics-sha256 "cffb809c40ddd3382131a8ed8b1b8536457da4e46dfced85dff5669fb6d73342" \
  --expected-bridge-artifact-sha256 "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384" \
  --expected-bridge-profile-sha256 "e3fc98c666b70b367962e400435afe7588a9f1a2b13619dffc6f40c6458c5ca1" \
  --expected-candidate-state-fingerprint "${expected_candidate_state_sha256}" \
  --output-path "${candidate}" \
  --capture-receipt "${capture_receipt}" \
  --device cuda:0 \
  --dtype bfloat16
