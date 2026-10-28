#!/usr/bin/env bash
set -euo pipefail

runtime=/home/cereal/model-surgery-lab/runtime/neural-forge-299d171c7f43533bf9954da815fe3a5e3d492739
source_run=/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/d6f5e1c8/structural-first-effectful-a0005
run_parent=/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_causal_head_frontier_20260724/299d171c
run_directory=${run_parent}/six-head-control-a0003
log=${run_directory}.log
pid_file=${run_directory}.pid

install -d "${run_parent}"
test ! -e "${run_directory}"
test ! -e "${log}"

nohup env \
  CUDA_VISIBLE_DEVICES=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 \
  PYTHONPATH="${runtime}:${runtime}/src" \
  /home/cereal/model-surgery-lab/venv/bin/python -u \
  "${runtime}/experiments/run_lfm2_exact4x_causal_head_frontier.py" \
  --run-directory "${run_directory}" \
  --source-run-directory "${source_run}" \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --semantic-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --expected-semantic-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-source-structural-interface-sha256 973623ef420c9c939a0cae66276723b91807d0c09af32fea18ed096b853b53a2 \
  --expected-source-deferred-w2-sha256 3e96e6e103ec50b14edc290c6472e55ee32fc88c13dc6d7b4bda20aa2b32640f \
  --expected-source-terminal-evidence-sha256 9818e8046727dc55044f9c02f632e04bdc563408fdefe52af0e216e9a995938e \
  --expected-source-causal-trie-sha256 3e18b636752ebb10b5ed7f81f37c419ccb6aee4bc52f39bcc4aa59886dd2f1df \
  --expected-source-sequence-plan-sha256 f1feb0de723cf4d417caa1f0e5753732e9037fe88a768673b1cc9e31b76fc15d \
  --expected-source-checkpoint-inventory-sha256 48da8b622bfe3273017f20b648884b0a1074858821afc97ffe589311ae73fb6e \
  --prompts-json "${runtime}/research/lfm2_semantic_whole_response_prompts_v1.json" \
  --expected-prompts-file-sha256 ff6fa27e08a3cdbaa647bdeec1d0630d218f98ee13b195a836b85de7438974fa \
  --parent-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json \
  --expected-parent-envelope-sha256 e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97 \
  --expected-parent-envelope-file-sha256 c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7 \
  --model-device cuda:0 \
  --analytic-solve-device cuda:0 \
  --maximum-prompt-tokens 1024 \
  --maximum-new-tokens 256 \
  >"${log}" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${pid_file}"
printf '%s\n' "${pid}"
