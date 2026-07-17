#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
sharded_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-sharded-v1
contract_document="$sharded_root/run-contract.json"
canonical_proposal_root="$sharded_root/canonical-proposals"
work_root="$sharded_root/merge"
frontier_document="$sharded_root/frontier.json"
finalizer_report="$sharded_root/canonical-finalizer-report.json"
log_root="$sharded_root/logs"
log_path="$log_root/merge.log"
pid_path="$log_root/merge.pid"
venv_python=/home/cereal/model-surgery-lab/venv/bin/python
expected_producer_sha=db42b58b075ecde7d3ec1102bea9aa4db050cdb1369be1543570187f8965ad25
expected_finalizer_sha=87222c699b77f601e53e7085c7a792635846787b06a1df5147a82ddbcf440e3c
expected_canonical_allocator_sha=f543c9180ff3da8d6e5c378cb0e203df4a1ba4570469c13538d7196c1afa7364

for shard_index in 00 01 02; do
  manifest="$sharded_root/shards/shard-$shard_index-of-03/shard-manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "merge_deferred_missing_manifest=$manifest"
    exit 1
  fi
done

actual_producer_sha="$(sha256sum "$source_root/experiments/build_lfm2_cached_causal_evidence.py" | awk '{print $1}')"
if [[ "$actual_producer_sha" != "$expected_producer_sha" ]]; then
  echo "producer_hash_mismatch expected=$expected_producer_sha actual=$actual_producer_sha"
  exit 1
fi
actual_finalizer_sha="$(sha256sum "$source_root/experiments/finalize_lfm2_cached_causal_merge_canonical.py" | awk '{print $1}')"
if [[ "$actual_finalizer_sha" != "$expected_finalizer_sha" ]]; then
  echo "finalizer_hash_mismatch expected=$expected_finalizer_sha actual=$actual_finalizer_sha"
  exit 1
fi
actual_canonical_allocator_sha="$(sha256sum "$source_root/src/neural_forge/hf_lfm2_interaction_beam_allocator.py" | awk '{print $1}')"
if [[ "$actual_canonical_allocator_sha" != "$expected_canonical_allocator_sha" ]]; then
  echo "canonical_allocator_hash_mismatch expected=$expected_canonical_allocator_sha actual=$actual_canonical_allocator_sha"
  exit 1
fi

mkdir -p "$work_root" "$log_root"
if [[ -f "$pid_path" ]]; then
  existing_pid="$(tr -dc '0-9' < "$pid_path")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "merge_already_live pid=$existing_pid"
    exit 0
  fi
fi

cd "$source_root"
nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$source_root/src:$source_root" \
  NEURAL_FORGE_CANONICAL_PROPOSAL_ROOT="$canonical_proposal_root" \
  NEURAL_FORGE_CAUSAL_RUN_CONTRACT="$contract_document" \
  NEURAL_FORGE_CANONICAL_FINALIZER_REPORT="$finalizer_report" \
  "$venv_python" experiments/finalize_lfm2_cached_causal_merge_canonical.py \
    --mode merge \
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
    --frontier-document "$frontier_document" \
    --merge-shard-root "$sharded_root/shards/shard-00-of-03" \
    --merge-shard-root "$sharded_root/shards/shard-01-of-03" \
    --merge-shard-root "$sharded_root/shards/shard-02-of-03" \
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

merge_pid=$!
printf '%s\n' "$merge_pid" > "$pid_path"
echo "merge_started pid=$merge_pid log=$log_path"
