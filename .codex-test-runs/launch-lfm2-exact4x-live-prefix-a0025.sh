set -euo pipefail

run=/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/neural-forge-92b2e091/fp32-source-local-residual-rho025-live-full-prefix-whole4x-a0025
test ! -e "$run"
mkdir -p "$(dirname "$run")"

export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd
nohup /home/cereal/model-surgery-lab/venv/bin/python -u /home/cereal/model-surgery-lab/runtimes/neural-forge-92b2e091/experiments/run_lfm2_exact4x_composite_fold.py \
  --run-directory "$run" \
  --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --compiler-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --expected-compiler-cache-key 76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --semantic-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --expected-semantic-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --trajectory-corpus /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_folding_corpora_20260717/v4096_broad256_v1/trajectories.json \
  --selective-catalog-store-directory /home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_v5_frontier_20260719/8be9f891254519950e114bfd54196ce661bcf5cf/selective-catalogs \
  --selective-catalog-receipt-json /home/cereal/model-surgery-lab/artifacts/lfm2_selective_catalog_20260721/97ae4b9/a0002/selective-catalog-preparation.json \
  --expected-selective-catalog-receipt-sha256 4fbe0a9750446926c2a63f70d31078af2786971048d4c7f9d45882c4fdeb8dc9 \
  --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-parent-unique-parameters 229693184 \
  --prompts-json /home/cereal/model-surgery-lab/runtimes/neural-forge-92b2e091/research/lfm2_semantic_whole_response_prompts_v1.json \
  --expected-prompts-file-sha256 ff6fa27e08a3cdbaa647bdeec1d0630d218f98ee13b195a836b85de7438974fa \
  --parent-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json \
  --expected-parent-envelope-sha256 e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97 \
  --expected-parent-envelope-file-sha256 c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7 \
  --body-recipe nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected \
  --expected-body-recipe-sha256 a10962bd077a6412ab3becbb22ffac760fbb57fc56b6d2c6d1dd7b3cfd6b29d9 \
  --short-conv-construction body-global-coordinate-complete-structural-v1 \
  --coordinate-completion-recipe bridge-procrustes-isometric \
  --normalized-input-request-capture-directory "$run/normalized-input-request-capsules" \
  --live-child-prefix-recapture-mode full-depth-full-prefix-after-material-install \
  --blank-seed 0 \
  --model-device cuda:0 \
  --fold-compute-device cuda:0 \
  --analytic-solve-device cuda:0 \
  --global-residual-backend cuda \
  --global-residual-basis-rank 128 \
  --correction-scale 1.0 \
  --normalized-input-correction-rho 0.25 \
  --child-dtype float32 \
  --target-dtype-representability-level requested \
  --maximum-prompt-tokens 1024 \
  --maximum-new-tokens 256 \
  > "$run.log" 2>&1 &

pid=$!
echo "$pid" > "$run.pid"
echo "$pid"
