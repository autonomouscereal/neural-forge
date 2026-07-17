#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
artifact_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717
sharded_root="$artifact_root/cached-causal35-sharded-v1"
shard_root="$sharded_root/shards/shard-02-of-03"
contract_document="$sharded_root/run-contract.json"
canonical_proposal_root="$sharded_root/canonical-proposals"
log_root="$sharded_root/logs"
control_root="$sharded_root/control"
venv_python=/home/cereal/model-surgery-lab/venv/bin/python
producer="$source_root/experiments/build_lfm2_cached_causal_evidence.py"
wrapper="$source_root/experiments/run_lfm2_cached_causal_turing_eager_shard.py"
expected_producer_sha=db42b58b075ecde7d3ec1102bea9aa4db050cdb1369be1543570187f8965ad25
expected_wrapper_sha=c6a6d6f080a3c264c1b5b97b8cb2de1d5c4824bc6a5385d9dc3e094a36997263
expected_contract_file_sha=89ae610882ac764a904ddb1ca6ebd82f6e4112f4d8fb993aa165c327bbee89f7
expected_spec_file_sha=a088afa1956d7a4069e374e6a05ab2f82de025f78a2a6acd8339f7598fbb649d
pid_path="$log_root/shard-02-of-03-gpu6.pid"
log_path="$log_root/shard-02-of-03-gpu6-eager.log"
receipt_path="$control_root/shard-02-of-03-gpu6-eager-runtime.json"

require_live_worker() {
  local pid_path_value="$1"
  local label="$2"
  local pid
  if [[ ! -f "$pid_path_value" ]]; then
    echo "$label pid_file_missing=$pid_path_value"
    exit 1
  fi
  pid="$(tr -dc '0-9' < "$pid_path_value")"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$label worker_not_live pid=$pid"
    exit 1
  fi
  echo "$label preserved_live_pid=$pid"
}

require_live_worker "$log_root/shard-00-of-03-gpu4.pid" shard0_gpu4
require_live_worker "$log_root/shard-01-of-03-gpu0.pid" shard1_gpu0

if [[ -f "$pid_path" ]]; then
  stale_pid="$(tr -dc '0-9' < "$pid_path")"
  if [[ -n "$stale_pid" ]] && kill -0 "$stale_pid" 2>/dev/null; then
    echo "shard2_already_live=yes pid=$stale_pid"
    exit 1
  fi
  echo "shard2_stale_pid_authenticated=$stale_pid"
fi

for binding in \
  "$producer:$expected_producer_sha" \
  "$wrapper:$expected_wrapper_sha" \
  "$contract_document:$expected_contract_file_sha" \
  "$shard_root/shard-spec.json:$expected_spec_file_sha"
do
  path="${binding%%:*}"
  expected="${binding##*:}"
  if [[ ! -f "$path" ]] || [[ -L "$path" ]]; then
    echo "unsafe_binding=$path"
    exit 1
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "binding_hash_mismatch path=$path expected=$expected actual=$actual"
    exit 1
  fi
done

if [[ -f "$shard_root/shard-manifest.json" ]]; then
  echo "shard2_already_complete=yes"
  exit 1
fi
record_count=0
cache_file_count=0
if [[ -d "$shard_root/candidate-records" ]]; then
  record_count="$(find "$shard_root/candidate-records" -maxdepth 1 -type f -name '*.json' | wc -l)"
fi
if [[ -d "$shard_root/candidate-caches" ]]; then
  cache_file_count="$(find "$shard_root/candidate-caches" -type f | wc -l)"
fi
unexpected_count="$(
  find "$shard_root" -mindepth 1 -maxdepth 1 \
    ! -name shard-spec.json \
    ! -name candidate-records \
    ! -name candidate-caches | wc -l
)"
if [[ "$record_count" != 0 ]] || [[ "$cache_file_count" != 0 ]] || [[ "$unexpected_count" != 0 ]]; then
  echo "shard2_not_clean records=$record_count cache_files=$cache_file_count unexpected=$unexpected_count"
  exit 1
fi
echo "shard2_clean_boundary_authenticated=yes records=0 cache_files=0"

mkdir -p "$log_root" "$control_root"
cd "$source_root"
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$source_root/src:$source_root" \
  NEURAL_FORGE_EXPECTED_PRODUCER_FILE_SHA256="$expected_producer_sha" \
  NEURAL_FORGE_EXPECTED_RUN_CONTRACT_FILE_SHA256="$expected_contract_file_sha" \
  NEURAL_FORGE_EXPECTED_SHARD_SPEC_FILE_SHA256="$expected_spec_file_sha" \
  NEURAL_FORGE_LFM2_EAGER_RECEIPT="$receipt_path" \
  "$venv_python" "$wrapper" \
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
    --shard-index 2 \
    --shard-weights 3,3,2 \
    --device cuda:6 \
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

worker_pid=$!
printf '%s\n' "$worker_pid" > "$pid_path"
if ! kill -0 "$worker_pid" 2>/dev/null; then
  echo "shard2_launch_failed pid=$worker_pid log=$log_path"
  exit 1
fi
echo "shard2_started=yes pid=$worker_pid device=cuda:6 backend=eager log=$log_path receipt=$receipt_path"
