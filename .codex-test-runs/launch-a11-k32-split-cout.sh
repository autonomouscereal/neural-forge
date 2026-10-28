set -euo pipefail

SCRIPT=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/launch-a11-k32-split-cout.sh
RUNTIME=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a11-k32-split-cout-lambda0001
PYTHON=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
CODEBOOK=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/5b5557b8da7aa125ada8cac305005de7b9e157933c3d4878f6be133fb5c51cc7/payload/artifact
COUT_ROOT=/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a10-k32-cout-lambda0001/math-cache/75cf54bc04438952a033ea494a6a269c1abd414cdb62a4b87c9bc77b73c57284

if [ "${1:-}" = "--worker" ]; then
  mkdir -p "$RUN"
  exec >>"$RUN/job.log" 2>&1
  export CUDA_VISIBLE_DEVICES=4
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$RUNTIME/src"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd "$RUNTIME"
  printf '{"event":"worker-start","utc":"%s","cuda_visible_devices":"4","base_codebook_reuse":true,"split_cout":true}\n' "$(date -u +%FT%TZ)"
  set +e
  "$PYTHON" -u experiments/run_lfm2_sparse_semantic_codebook.py \
    --run-directory "$RUN" \
    --phase-cache-directory /home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1 \
    --checkpoint-directory /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723/89cd45b0/same-body-v4096-p40-gpu1-a0001/checkpoint \
    --base-codebook-directory "$CODEBOOK" \
    --expected-base-codebook-artifact-sha256 4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d \
    --functional-output-coefficients "$COUT_ROOT/output-coefficients.safetensors" \
    --functional-output-refit-receipt "$COUT_ROOT/result.json" \
    --expected-functional-output-refit-cache-key 75cf54bc04438952a033ea494a6a269c1abd414cdb62a4b87c9bc77b73c57284 \
    --expected-functional-output-refit-result-sha256 b301539b49e0156e5854e3d84730ead019b32cadd1d0436e017474556408bfc8 \
    --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
    --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --complete-causal-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
    --expected-checkpoint-artifact-inventory-sha256 41007288f9a1a7b9dcdf96b4d1d39a1eeec416c46fb928b5bb2cc35ea9cbf8b2 \
    --expected-child-state-sha256 3b9f2f7dc2451b2df910e76b8efdb40159beb9690a02059e39bcd48b7899266e \
    --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
    --expected-parent-unique-parameters 229693184 \
    --expected-child-unique-parameters 166778624 \
    --prompts-json "$RUNTIME/research/lfm2_semantic_whole_response_prompts_v1.json" \
    --expected-prompts-file-sha256 ff6fa27e08a3cdbaa647bdeec1d0630d218f98ee13b195a836b85de7438974fa \
    --parent-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json \
    --expected-parent-envelope-sha256 e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97 \
    --expected-parent-envelope-file-sha256 c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7 \
    --support-selection residual-omp \
    --k 32 \
    --candidate-ridge 1e-5 \
    --refit-ridge 1e-5 \
    --neighbour-pool-size 32 \
    --row-chunk-size 256 \
    --device cuda:0 \
    --dtype auto \
    --checkpoint-provenance-device cpu \
    --checkpoint-provenance-dtype bfloat16 \
    --selection-device cuda:0 \
    --selection-dtype float32 \
    --solve-device cpu \
    --solve-dtype float64 \
    --maximum-span-rows-per-chunk 1024 \
    --maximum-prompt-tokens 1024 \
    --maximum-new-tokens 256 \
    --maximum-embedding-rows-per-chunk 1024 \
    --maximum-donor-rows-per-score-chunk 4096 \
    --maximum-hidden-rows-per-score-chunk 8 \
    --expected-prior-direct-winners 293 \
    --expected-prior-multi-code-winners 72 \
    --console-progress-every 1
  code=$?
  set -e
  printf '{"event":"worker-exit","utc":"%s","exit_code":%s}\n' "$(date -u +%FT%TZ)" "$code"
  printf '%s\n' "$code" >"$RUN/job.status"
  exit "$code"
fi

mkdir -p "$RUN"
if [ -f "$RUN/run-receipt.json" ]; then
  printf '{"event":"launch-skipped","reason":"sealed-run-exists","run":"%s"}\n' "$RUN"
  exit 0
fi
if [ -f "$RUN/job.pid" ]; then
  old_pid=$(cat "$RUN/job.pid")
  if kill -0 "$old_pid" 2>/dev/null; then
    printf '{"event":"launch-skipped","reason":"already-running","pid":%s,"run":"%s"}\n' "$old_pid" "$RUN"
    exit 0
  fi
fi
rm -f "$RUN/job.status"
nohup bash "$SCRIPT" --worker >/dev/null 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$RUN/job.pid"
printf '{"event":"job-started","pid":%s,"run":"%s"}\n' "$pid" "$RUN"
