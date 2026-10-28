set -euo pipefail

root=/home/cereal/model-surgery-lab/neural-forge-src
cd "$root"

for path in \
  experiments/rerun_lfm2_sparse_semantic_capped_prompts.py \
  experiments/lfm2_a9_a11_high_cap_reuse_manifest_20260730.json \
  experiments/launch_lfm2_a9_a11_high_cap_reuse_20260730.sh \
  src/neural_forge/lfm2_decoded_review.py \
  src/neural_forge/hf_lfm2_semantic_step_model_runtime.py \
  experiments/run_lfm2_semantic_step_bridge_control.py \
  experiments/run_lfm2_sparse_semantic_codebook.py \
  experiments/launch_detached.py \
  research/lfm2_semantic_whole_response_prompts_v1.json \
  src/neural_forge/dynamic_circuit_ir.py \
  src/neural_forge/hf_lfm2_direct_local.py \
  src/neural_forge/hf_lfm2_semantic_step_token_fold.py \
  src/neural_forge/hf_lfm2_sparse_semantic_codebook.py \
  src/neural_forge/hf_tokenizer_bridge.py
do
  if [ -f "$path" ]; then
    sha256sum "$path"
  else
    echo "MISSING  $path"
  fi
done

echo GPU4
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | grep 'GPU-2ce6a95c-7b1f-644a-3808-d781732a0582' || true
