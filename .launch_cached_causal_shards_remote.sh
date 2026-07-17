#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
artifact_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717
sharded_root="$artifact_root/cached-causal35-sharded-v1"
legacy_work_root="$artifact_root/cached-causal35-rtx3090-gpu4-v1/work"
contract_document="$sharded_root/run-contract.json"
canonical_proposal_root="$sharded_root/canonical-proposals"
log_root="$sharded_root/logs"
venv_python=/home/cereal/model-surgery-lab/venv/bin/python
expected_producer_sha=db42b58b075ecde7d3ec1102bea9aa4db050cdb1369be1543570187f8965ad25
eager_wrapper="$source_root/experiments/run_lfm2_cached_causal_turing_eager_shard.py"
expected_eager_wrapper_sha=c6a6d6f080a3c264c1b5b97b8cb2de1d5c4824bc6a5385d9dc3e094a36997263
expected_contract_file_sha=89ae610882ac764a904ddb1ca6ebd82f6e4112f4d8fb993aa165c327bbee89f7
expected_shard2_spec_file_sha=a088afa1956d7a4069e374e6a05ab2f82de025f78a2a6acd8339f7598fbb649d

if [[ ! -f "$contract_document" ]]; then
  echo "run_contract_missing=$contract_document"
  exit 1
fi

actual_producer_sha="$(sha256sum "$source_root/experiments/build_lfm2_cached_causal_evidence.py" | awk '{print $1}')"
if [[ "$actual_producer_sha" != "$expected_producer_sha" ]]; then
  echo "producer_hash_mismatch expected=$expected_producer_sha actual=$actual_producer_sha"
  exit 1
fi

mkdir -p "$log_root" "$sharded_root/shards"

launch_worker() {
  local shard_index="$1"
  local device_index="$2"
  local use_legacy="$3"
  local shard_tag
  local shard_root
  local log_path
  local pid_path
  local entrypoint="$source_root/experiments/build_lfm2_cached_causal_evidence.py"
  local -a adoption_args=()
  local -a runtime_environment=()

  printf -v shard_tag 'shard-%02d-of-03' "$shard_index"
  shard_root="$sharded_root/shards/$shard_tag"
  log_path="$log_root/$shard_tag-gpu$device_index.log"
  pid_path="$log_root/$shard_tag-gpu$device_index.pid"
  mkdir -p "$shard_root"

  if [[ -f "$shard_root/shard-manifest.json" ]]; then
    echo "$shard_tag already_complete=yes"
    return
  fi
  if [[ -f "$pid_path" ]]; then
    local existing_pid
    existing_pid="$(tr -dc '0-9' < "$pid_path")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "$shard_tag already_live=yes pid=$existing_pid"
      return
    fi
  fi
  if [[ "$use_legacy" == "yes" ]]; then
    adoption_args=(
      --adopt-candidate-record-root
      "$legacy_work_root"
    )
  fi
  if [[ "$shard_index" == 2 ]]; then
    local actual_wrapper_sha
    actual_wrapper_sha="$(sha256sum "$eager_wrapper" | awk '{print $1}')"
    if [[ "$actual_wrapper_sha" != "$expected_eager_wrapper_sha" ]]; then
      echo "eager_wrapper_hash_mismatch expected=$expected_eager_wrapper_sha actual=$actual_wrapper_sha"
      exit 1
    fi
    entrypoint="$eager_wrapper"
    runtime_environment=(
      NEURAL_FORGE_EXPECTED_PRODUCER_FILE_SHA256="$expected_producer_sha"
      NEURAL_FORGE_EXPECTED_RUN_CONTRACT_FILE_SHA256="$expected_contract_file_sha"
      NEURAL_FORGE_EXPECTED_SHARD_SPEC_FILE_SHA256="$expected_shard2_spec_file_sha"
      NEURAL_FORGE_LFM2_EAGER_RECEIPT="$sharded_root/control/shard-02-of-03-gpu6-eager-runtime.json"
    )
  fi

  cd "$source_root"
  nohup env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$source_root/src:$source_root" \
    "${runtime_environment[@]}" \
    "$venv_python" "$entrypoint" \
      --mode shard \
      --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
      --parent-cache /home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
      --expected-parent-cache-key 5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
      --parent-output-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/heldout_parent_responses.envelope-v1-strict.json \
      --expected-parent-output-file-sha256 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 \
      --expected-parent-identity 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
      --expected-tokenizer-fingerprint eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
      --expected-parent-unique-parameters 229693184 \
      --work-root "$shard_root" \
      --run-contract-document "$contract_document" \
      --canonical-proposal-root "$canonical_proposal_root" \
      --shard-count 3 \
      --shard-index "$shard_index" \
      --shard-weights 3,3,2 \
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
      "${adoption_args[@]}" \
      >>"$log_path" 2>&1 &

  local worker_pid=$!
  printf '%s\n' "$worker_pid" > "$pid_path"
  echo "$shard_tag started=yes pid=$worker_pid device=cuda:$device_index log=$log_path"
}

launch_worker 0 4 yes
launch_worker 1 0 no

gpu6_release_file="$sharded_root/control/gpu6-probe.release"
if [[ ! -f "$gpu6_release_file" ]]; then
  echo "shard-02-of-03 deferred=yes release_file=$gpu6_release_file"
else
  launch_worker 2 6 no
fi
