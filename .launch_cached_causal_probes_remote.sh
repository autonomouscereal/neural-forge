#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
sharded_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-sharded-v1
contract_document="$sharded_root/run-contract.json"
canonical_proposal_root="$sharded_root/canonical-proposals"
log_root="$sharded_root/logs"
venv_python=/home/cereal/model-surgery-lab/venv/bin/python
expected_producer_sha=db42b58b075ecde7d3ec1102bea9aa4db050cdb1369be1543570187f8965ad25

if [[ ! -f "$contract_document" ]]; then
  echo "probe_deferred_contract_missing=yes"
  exit 1
fi
actual_producer_sha="$(sha256sum "$source_root/experiments/build_lfm2_cached_causal_evidence.py" | awk '{print $1}')"
if [[ "$actual_producer_sha" != "$expected_producer_sha" ]]; then
  echo "producer_hash_mismatch expected=$expected_producer_sha actual=$actual_producer_sha"
  exit 1
fi
mkdir -p "$log_root" "$sharded_root/probes"

launch_probe() {
  local device_index="$1"
  local work_root="$sharded_root/probes/gpu$device_index-work"
  local report_path="$sharded_root/probes/gpu$device_index.json"
  local log_path="$log_root/probe-gpu$device_index.log"
  local pid_path="$log_root/probe-gpu$device_index.pid"

  if [[ -f "$report_path" ]]; then
    echo "probe_gpu$device_index already_complete=yes"
    return
  fi
  if [[ -f "$pid_path" ]]; then
    local existing_pid
    existing_pid="$(tr -dc '0-9' < "$pid_path")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "probe_gpu$device_index already_live=yes pid=$existing_pid"
      return
    fi
  fi
  mkdir -p "$work_root"
  cd "$source_root"
  nohup env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$source_root/src:$source_root" \
    "$venv_python" experiments/build_lfm2_cached_causal_evidence.py \
      --mode probe \
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
      --probe-layer-index 0 \
      --probe-target-width 896 \
      --probe-report "$report_path" \
      --device "cuda:$device_index" \
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

  local probe_pid=$!
  printf '%s\n' "$probe_pid" > "$pid_path"
  echo "probe_gpu$device_index started=yes pid=$probe_pid log=$log_path"
}

for requested_device in "$@"; do
  launch_probe "$requested_device"
done
