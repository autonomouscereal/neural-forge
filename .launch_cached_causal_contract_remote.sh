#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
artifact_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-sharded-v1
work_root="$artifact_root/contract-work"
contract_document="$artifact_root/run-contract.json"
canonical_proposal_root="$artifact_root/canonical-proposals"
log_root="$artifact_root/logs"
log_path="$log_root/contract-gpu0.log"
pid_path="$log_root/contract-gpu0.pid"
venv_python=/home/cereal/model-surgery-lab/venv/bin/python
expected_producer_sha=db42b58b075ecde7d3ec1102bea9aa4db050cdb1369be1543570187f8965ad25

mkdir -p "$work_root" "$log_root"

actual_producer_sha="$(sha256sum "$source_root/experiments/build_lfm2_cached_causal_evidence.py" | awk '{print $1}')"
if [[ "$actual_producer_sha" != "$expected_producer_sha" ]]; then
  echo "producer_hash_mismatch expected=$expected_producer_sha actual=$actual_producer_sha"
  exit 1
fi

if [[ -f "$pid_path" ]]; then
  existing_pid="$(tr -dc '0-9' < "$pid_path")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "contract_builder_already_live pid=$existing_pid"
    exit 0
  fi
fi

cd "$source_root"
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$source_root/src:$source_root" \
  "$venv_python" experiments/build_lfm2_cached_causal_evidence.py \
    --mode contract \
    --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --parent-cache /home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
    --expected-parent-cache-key 5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
    --parent-output-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/heldout_parent_responses.envelope-v1-strict.json \
    --expected-parent-output-file-sha256 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 \
    --expected-parent-identity 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
    --expected-tokenizer-fingerprint eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
    --expected-parent-unique-parameters 229693184 \
    --work-root "$work_root" \
    --run-contract-document "$contract_document" \
    --canonical-proposal-root "$canonical_proposal_root" \
    --device cuda:0 \
    --dtype bfloat16 \
    --coarse-widths 1792,1280,896,512 \
    --profile balanced \
    --cluster-ridge 1e-8 \
    --cluster-max-iterations 8 \
    --cluster-distance-chunk-size 512 \
    --rollout-batch-size 16 \
    --max-capture-bytes 1073741824 \
    --maximum-stage-fraction 0.10 \
    --max-states 100000 \
    >"$log_path" 2>&1 &

builder_pid=$!
printf '%s\n' "$builder_pid" > "$pid_path"
echo "contract_builder_started pid=$builder_pid log=$log_path"
