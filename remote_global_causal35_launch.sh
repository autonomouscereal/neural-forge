#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PYTHON="$LAB/venv/bin/python"
PARENT_MODEL="$LAB/models/LFM2.5-230M"
PARENT_CACHE="$LAB/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656"
PARENT_ENVELOPE="$LAB/artifacts/lfm2_residual_attention_fold_20260716/heldout_parent_responses.envelope-v1-strict.json"
RUN_ROOT="$LAB/artifacts/lfm2_global_causal_20260717/cached-causal35-rtx3090-gpu4-v1"
GPU_UUID=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582

verify_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}

verify_sha 6151c337405cc8a0e01f3d8c9394fda06b171540e21d4830a84ccf01b9a1ead1 "$SRC/experiments/build_lfm2_cached_causal_evidence.py"
verify_sha 07907cab56ec2a4cf1cfadd68c5a9d26fd91851d9bc9106778794059a224c2c9 "$SRC/experiments/run_lfm2_global_causal_allocator.py"
verify_sha 45e2d16f619b0814e88b27b396de436df6104c36acfe1eac82aa467ab67ae808 "$SRC/experiments/run_lfm2_long_horizon_repair.py"
verify_sha e9f8a556ed7441e2fe9c4d3f29ec1c56f0eb73e114be1bb45a0cfea9deb370c5 "$SRC/experiments/run_lfm2_residual_attention_fold.py"
verify_sha 0c4c11625133aa93db3da224784817a61f81f6dec3e3506bae437b465efafc00 "$SRC/experiments/run_lfm2_dynamic_catalog.py"
verify_sha a5996325ae26632326c89b011c126ed3ba3ee3c7d834a185468829410f12d568 "$SRC/src/neural_forge/hf_lfm2_cached_causal_evidence.py"
verify_sha 6452ac4d1aff90c9c52410197d3da7d23d2c8f2fcf122e4f8fa6b77407d48640 "$SRC/src/neural_forge/hf_lfm2_global_causal_allocator.py"
verify_sha 87ba1faa6a7767f6260613f4da5439871057bb251499e597ead51bc2273e9b5b "$SRC/src/neural_forge/hf_lfm2_variable_width_fold.py"
verify_sha 33fcad0de10288faf1f98299780c4b05c7bdee21e2227c986b2ce06a3f67e4ba "$SRC/src/neural_forge/hf_lfm2_circuit_trace_cache.py"
verify_sha 2b0d04a679902b7ffa8d43629b16cc4a63936d9047348db7a43dc0cc079277dc "$SRC/src/neural_forge/hf_lfm2_long_horizon_anchors.py"
verify_sha 4a7cd925dea24a84a1441bd71adcdfe3d3ee956ecb65cb8cfac0d225bdf47a49 "$SRC/src/neural_forge/dynamic_lfm2_behavior.py"
verify_sha 50c02515ef3b2e2e16b7c784144a62acae6f13d6b8b639246749291399401e96 "$SRC/src/neural_forge/dynamic_fold_executor.py"
verify_sha eed7deb354e2d17ca17862a30735bc4b1cbe310f0c568e859d3423b24af87f83 "$SRC/src/neural_forge/hf_activation_cluster.py"
verify_sha 5956f90939ab57dcabb32377d795b00eda0d948666c63b5688259b5680aa738a "$SRC/src/neural_forge/hf_lfm2_canonical.py"
verify_sha 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 "$PARENT_ENVELOPE"

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
test "$(nvidia-smi --id=4 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
test "$(nvidia-smi --id=4 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')" -ge 20000
test "GPU-$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_properties(4).uuid)')" = "$GPU_UUID"

if test -f "$RUN_ROOT/run.pid"; then
  OLD_PID="$(cat "$RUN_ROOT/run.pid")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    printf 'existing producer is running: pid=%s\n' "$OLD_PID"
    exit 1
  fi
fi

mkdir -p "$RUN_ROOT"

export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SRC"

nohup "$PYTHON" experiments/build_lfm2_cached_causal_evidence.py \
  --parent-model "$PARENT_MODEL" \
  --parent-cache "$PARENT_CACHE" \
  --expected-parent-cache-key 5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
  --parent-output-envelope "$PARENT_ENVELOPE" \
  --expected-parent-output-file-sha256 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 \
  --expected-parent-identity 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-tokenizer-fingerprint eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --expected-parent-unique-parameters 229693184 \
  --work-root "$RUN_ROOT/work" \
  --frontier-document "$RUN_ROOT/frontier.json" \
  --device cuda:4 \
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
  > "$RUN_ROOT/run.log" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$RUN_ROOT/run.pid"
printf 'prepared causal evidence producer pid=%s root=%s\n' "$PID" "$RUN_ROOT"
