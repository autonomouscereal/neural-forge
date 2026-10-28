#!/usr/bin/env bash
set -euo pipefail

SRC=/home/cereal/model-surgery-lab/neural-forge-src
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python
RUN_ROOT=/home/cereal/model-surgery-lab/artifacts/lfm2_sparse_residual_vocab_frontier_20260717/strong35-rtx3090-gpu4-v2-native-norm
AUTH="$RUN_ROOT/authorization.json"

STRONG_MODEL=/home/cereal/model-surgery-lab/artifacts/lfm2_large_ratio_20260716/wave2-producer-first-r2/lfm2_10pct_mlp_q16_kv8_ffn2026_producer_first/output/final-child
STRONG_CACHE=/home/cereal/model-surgery-lab/artifacts/lfm2_long_horizon_repair_20260716/circuit-cache/child/59a7ead8c22bb484aeb463dbccb765fb02c07c876a1035d4e902735fe11cc96b
PARENT_MODEL=/home/cereal/model-surgery-lab/models/LFM2.5-230M
PARENT_TRAJECTORY=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/4164681e417f9dbab9ab92c2dc073649600832f764544530a724663303cbe3f1
RECOVERY_CACHE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656
RECOVERY_ENVELOPE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/heldout_parent_responses.envelope-v1-strict.json
HOLDOUT_ENVELOPE=/home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json
FAILED_AUTH=/home/cereal/model-surgery-lab/artifacts/lfm2_sparse_residual_vocab_frontier_20260717/strong35-rtx3090-gpu4-v1/authorization.json
GPU_UUID=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582

verify_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}

test ! -e "$RUN_ROOT"

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test "$(nvidia-smi --id=4 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
test "$(nvidia-smi --id=4 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')" -ge 20000
test "GPU-$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_properties(4).uuid)')" = "$GPU_UUID"

verify_sha 3437503b7f36c35e09eeeaeafdb1bf0ec4a03e19db0804dc441c31eb6a44c4d6 "$SRC/experiments/run_lfm2_sparse_residual_vocab_frontier.py"
verify_sha 3137e32dd7f83d3bcb929e1ba5a0b5631615878f1be5d637aa300b73cf977724 "$SRC/src/neural_forge/hf_sparse_residual_vocab.py"
verify_sha 558ab01ed72a1d475ba000d0ff6ff45b4fe056a118b02753284f338eff31229d "$SRC/src/neural_forge/hf_sparse_residual_vocab_selection.py"
verify_sha cdec34441c4401361ce69c1f06227b951865c883f9565ada4a94fa6a33751251 "$SRC/src/neural_forge/hf_lfm2_vocab_interface_fold.py"
verify_sha 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 "$RECOVERY_ENVELOPE"
verify_sha a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b "$HOLDOUT_ENVELOPE"
verify_sha 6d7997105cce03159152fcc45359415adebeec06c1b1af2069e5ca8cdf55bd02 "$FAILED_AUTH"

mkdir -p "$RUN_ROOT"
cp "$FAILED_AUTH" "$AUTH"
verify_sha 6d7997105cce03159152fcc45359415adebeec06c1b1af2069e5ca8cdf55bd02 "$AUTH"

cd "$SRC"

nohup "$PYTHON" experiments/run_lfm2_sparse_residual_vocab_frontier.py \
  --strong-source-model "$STRONG_MODEL" \
  --strong-source-cache "$STRONG_CACHE" \
  --original-parent-model "$PARENT_MODEL" \
  --parent-trajectory-cache "$PARENT_TRAJECTORY" \
  --recovery-horizon-cache "$RECOVERY_CACHE" \
  --recovery-parent-envelope "$RECOVERY_ENVELOPE" \
  --expected-recovery-parent-envelope-file-sha256 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 \
  --expected-recovery-horizon-cache-key 5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
  --expected-parent-state-fingerprint 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --tokenizer-directory "$PARENT_MODEL" \
  --expected-tokenizer-fingerprint eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --expected-strong-source-state-sha256 433f925c94fb8c508d7624db1c970e9e2d84c8af383fe54aa27a594159077271 \
  --expected-strong-source-cache-key 59a7ead8c22bb484aeb463dbccb765fb02c07c876a1035d4e902735fe11cc96b \
  --expected-original-parent-state-sha256 fd580146d97c970226d4b1667305a92d7f04bfd4f01aa8afedfd7d826dcacde0 \
  --expected-parent-trajectory-cache-key 4164681e417f9dbab9ab92c2dc073649600832f764544530a724663303cbe3f1 \
  --independent-holdout-parent-envelope "$HOLDOUT_ENVELOPE" \
  --expected-holdout-parent-envelope-file-sha256 a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b \
  --expected-holdout-parent-envelope-sha256 b0a2c815b7442df5a3cb9fa113aecaf2de5f094222f7d74e57140719e486e8d2 \
  --expected-holdout-manifest-sha256 5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690 \
  --expected-holdout-parent-state-fingerprint fd580146d97c970226d4b1667305a92d7f04bfd4f01aa8afedfd7d826dcacde0 \
  --expected-holdout-parent-model-fingerprint 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --execution-authorization-file "$AUTH" \
  --expected-execution-authorization-file-sha256 6d7997105cce03159152fcc45359415adebeec06c1b1af2069e5ca8cdf55bd02 \
  --work-root "$RUN_ROOT/work" \
  --checkpoint-root "$RUN_ROOT/checkpoint" \
  --report-path "$RUN_ROOT/report.json" \
  --device cuda:4 \
  --dtype bfloat16 \
  --strong-compression-fractions 0.35 \
  --direct-parent-compression-fractions 0.15 \
  --skip-direct-parent-control \
  --expected-original-parent-unique-parameters 229693184 \
  --expected-strong-source-unique-parameters 206726912 \
  --expected-vocab-size 65536 \
  --expected-hidden-size 1024 \
  > "$RUN_ROOT/run.log" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$RUN_ROOT/run.pid"
printf 'launched pid=%s root=%s\n' "$PID" "$RUN_ROOT"
