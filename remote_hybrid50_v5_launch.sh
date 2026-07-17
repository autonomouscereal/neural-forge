set -euo pipefail

LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PYTHON="$LAB/venv/bin/python"
PARENT_MODEL="$LAB/models/LFM2.5-230M"
CALIBRATION="$LAB/artifacts/lfm2_residual_attention_fold_20260716/calibration_parent_greedy.json"
PARENT_OUTPUTS="$LAB/artifacts/lfm2_residual_attention_fold_20260716/heldout_parent_responses.envelope-v1-strict.json"
CACHE_ROOT="$LAB/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache"
ROOT="$LAB/artifacts/lfm2_hybrid_50pct_20260717/direct-large-v5-producer-first"
GPU_INDEX=6
GPU_UUID=GPU-f5a09535-fc85-a027-50f7-734b344747dd

verify_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}

verify_sha 69dd8cc84d0a7d185700d493b7f05561a1af514b28bdf450921ad6fa47d81a53 "$SRC/experiments/run_lfm2_dynamic_catalog.py"
verify_sha 1ea7d02ca323573bdaef265f0f80ce2f066ec1045fbadc52fd6822f06d764748 "$SRC/src/neural_forge/hf_lfm2_hybrid_producer_first_model_fold.py"
verify_sha 5f4653c824015ed32a65a41bd43d28d455564a254f9ca0a5e276bf02635f26f0 "$SRC/src/neural_forge/dynamic_lfm2_adapter.py"
verify_sha a92fa75015e6a8addcad8f952581b4dce7912a0067d78c8648ac8645ee0e3e3d "$SRC/src/neural_forge/hf_lfm2_model_fold.py"
verify_sha 1cb05fefc8fe5c41cd768b2f35ef945860ab1cf1a7ab54ca4075dff6cb274d6b "$SRC/src/neural_forge/hf_lfm2_producer_first_model_fold.py"
verify_sha e009cb3c05cecbbca4233b5d23efa770d35dd881360f6f28da7fcd33f9f69590 "$SRC/src/neural_forge/dynamic_fold_compiler.py"
verify_sha 733ff5f74e563922f09b36f0247017a8ba84181513060af4e73cc520a703e4f3 "$SRC/src/neural_forge/dynamic_fold_planner.py"
verify_sha c483c6cc2baecafedbd661f3c2d4eabc7ec177f10817401b4a0e1e95fce4b4fd "$SRC/src/neural_forge/dynamic_fold_registry.py"
verify_sha 1ec392e268a85f2e244e2fb52461350a4ba60ca10fb3cab3988c51c24cb248c5 "$SRC/src/neural_forge/dynamic_model_inspector.py"
verify_sha 620d74748e789fdcb14438eb91c4c76b72f2dc6aa1d85d026916cfd9f445fca6 "$SRC/src/neural_forge/dynamic_target_synthesis.py"
verify_sha 42a700e0abdecf701caf1a86bd2c90d323c9ea7c12aac58a47d2b26862f14e9a "$SRC/src/neural_forge/dynamic_lfm2_targets.py"
verify_sha d8d8551c33996e66af258a22a23206dc9da2e462667153dd9031919e13f0af15 "$SRC/src/neural_forge/dynamic_lfm2_system.py"
verify_sha 4a7cd925dea24a84a1441bd71adcdfe3d3ee956ecb65cb8cfac0d225bdf47a49 "$SRC/src/neural_forge/dynamic_lfm2_behavior.py"
verify_sha 50c02515ef3b2e2e16b7c784144a62acae6f13d6b8b639246749291399401e96 "$SRC/src/neural_forge/dynamic_fold_executor.py"
verify_sha de0c29ad7ab9be9a8a99ad7649c9654e8ea7b4556491f3f6d615e1c534c06ad7 "$SRC/src/neural_forge/dynamic_production_catalog.py"
verify_sha d5abe55b05f4db2685a60efb500108f89bb951e2cf3270b93def8debfd7eccc3 "$SRC/src/neural_forge/dynamic_system.py"
verify_sha 520a8c4d6a07799a9fca00baefba448f6fe559a9b88fe0dfcd777dab1dffdd2d "$SRC/src/neural_forge/dynamic_behavior_gate.py"
verify_sha 26f312f582a5ffeae44cbf47d284ff7b1e550483223b14b0ed33d0035fde7f94 "$SRC/src/neural_forge/dynamic_fold_snapshot.py"
verify_sha a0613d25978d9e48f806523a0d3dfc178d12aa6f37db6f8b9f1e1b63b366ac1e "$SRC/src/neural_forge/dynamic_fold_artifact.py"
verify_sha 33fcad0de10288faf1f98299780c4b05c7bdee21e2227c986b2ce06a3f67e4ba "$SRC/src/neural_forge/hf_lfm2_circuit_trace_cache.py"
verify_sha 2efb1cd2a9a3a8e765cb292704b81ad5da4a992812b073691f38cad8e428f5ec "$SRC/src/neural_forge/hf_lfm2_closed_loop_fold.py"
verify_sha 5956f90939ab57dcabb32377d795b00eda0d948666c63b5688259b5680aa738a "$SRC/src/neural_forge/hf_lfm2_canonical.py"
verify_sha cbe2acd7d11fbee54c6a7ac1302f7ce4521438f01b4b623c7eb58445906a0521 "$SRC/src/neural_forge/hf_lfm2_intelligence_gate.py"
verify_sha 1c98485d8056b1288de10e2ad1d395d5213249a683bb9a1086fb8c558a891509 "$CALIBRATION"
verify_sha 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 "$PARENT_OUTPUTS"

test ! -e "$ROOT"
test "$(nvidia-smi --id="$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
test "$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')" -ge 44000
if nvidia-smi --id="$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits |
  grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'
then
  printf 'GPU %s already has an active compute process.\n' "$GPU_INDEX" >&2
  exit 1
fi

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
test "GPU-$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_properties(6).uuid)')" = "$GPU_UUID"

mkdir -p "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SRC"
nohup "$PYTHON" -u experiments/run_lfm2_dynamic_catalog.py \
  --parent-model "$PARENT_MODEL" \
  --tokenizer-directory "$PARENT_MODEL" \
  --calibration "$CALIBRATION" \
  --cached-parent-outputs "$PARENT_OUTPUTS" \
  --cached-parent-outputs-sha256 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 \
  --cache-root "$CACHE_ROOT" \
  --snapshot-root "$ROOT/snapshots" \
  --publish-destination "$ROOT/published-child" \
  --ledger "$ROOT/ledger.jsonl" \
  --receipt "$ROOT/receipt.json" \
  --diagnostics "$ROOT/diagnostics.json" \
  --parent-name lfm2.5-230m-local \
  --child-name lfm2-hybrid-50pct \
  --device cuda:0 \
  --dtype bfloat16 \
  --target-query-heads 14 \
  --target-key-value-heads 7 \
  --target-intermediate-size 1458 \
  --exact-target-parameters 114846080 \
  --hybrid-model-fold \
  --target-layer-types conv,full_attention,conv,full_attention,conv,full_attention \
  --ordered-source-spans '0,1;2,3;4,5;6,7;8,9,10;11,12,13' \
  --sample-last-tokens 8 \
  --max-decode-steps 8 \
  --max-capture-bytes 134217728 \
  --capture-logit-rtol 0.2 \
  --capture-logit-atol 1.0 \
  --child-decode-trace-mode full_prefix_canonical \
  --max-selection-rows 256 \
  --max-fit-rows 512 \
  --max-validation-rows 128 \
  --validation-fraction 0.2 \
  --endpoint-ridge 1e-6 \
  --max-endpoint-relative-rmse 1000000 \
  --min-endpoint-cosine-similarity -1.0 \
  --endpoint-max-prior-delta-ratio 1.0 \
  --min-teacher-top1 0.05 \
  --min-free-run-match 0.05 \
  --min-residual-cosine 0.0 \
  --whole-output-max-new-tokens 64 \
  --whole-output-maximum-prompt-tokens 512 \
  --whole-output-maximum-new-empty-responses 0 \
  --whole-output-maximum-new-fragment-responses 0 \
  --whole-output-minimum-parent-nonempty-retention 0.9 \
  --whole-output-minimum-task-score-retention 0.9 \
  --whole-output-maximum-parent-task-score-regressions 2 \
  --whole-output-maximum-new-generation-limit-saturations 2 \
  --whole-output-maximum-repeated-ngram-share-regressions 2 \
  --whole-output-maximum-repeated-bigram-share-delta 0.1 \
  > "$ROOT/run.log" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$ROOT/pid"
sleep 2
kill -0 "$PID"
printf 'launched hybrid50 producer-first pid=%s root=%s gpu=%s\n' \
  "$PID" "$ROOT" "$GPU_UUID"
