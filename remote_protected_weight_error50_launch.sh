#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PY="$LAB/venv/bin/python"
MODEL="$LAB/models/LFM2.5-230M"
ENVELOPE="$LAB/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json"
ROOT="$LAB/artifacts/lfm2_protected_weight_error_20260717/lfm2-230m-exact50-independent64-v1"
LOG="$ROOT.runner.log"
PID_FILE="$ROOT.runner.pid"
GPU_UUID=GPU-f5a09535-fc85-a027-50f7-734b344747dd

verify_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}

verify_sha 27886a1215a2b46298db5a6096a57a488b9b948d97cdb0f5938cccb37577e9b2 \
  "$SRC/src/neural_forge/hf_lfm2_protected_weight_error_allocator.py"
verify_sha 07e805277272ba5d4ba4b0868998641f6c06b533cbf31e249869b4d61001b7b0 \
  "$SRC/experiments/run_lfm2_protected_weight_error_allocator.py"
verify_sha ac4a9d9431768330318929a2316fc6a1142045d353dc9a6ca23b30d7bcd6103c \
  "$SRC/src/neural_forge/hf_lfm2_causal_value_allocator.py"
verify_sha 6bbbd78ee2f470f834da67b466ca7540a520f15097998c674be636a55cefa32b \
  "$SRC/src/neural_forge/hf_lfm2_parent_relative_manual_gate.py"
verify_sha 9df14a2929a0a3998a41c5b826ed846d121463f763492639a14dd608137b74a4 \
  "$SRC/experiments/run_lfm2_causal_value_allocator.py"
verify_sha a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b \
  "$ENVELOPE"

test ! -e "$ROOT"
test ! -e "$LOG"
test ! -e "$PID_FILE"
mkdir -p "$(dirname "$ROOT")"

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
test "$(nvidia-smi --id=6 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
test "$(nvidia-smi --id=6 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')" -ge 40000

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SRC"
nohup "$PY" experiments/run_lfm2_protected_weight_error_allocator.py \
  --parent-model-directory "$MODEL" \
  --tokenizer-directory "$MODEL" \
  --output-root "$ROOT" \
  --parent-envelope "$ENVELOPE" \
  --expected-parent-envelope-file-sha256 a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b \
  --expected-parent-envelope-sha256 b0a2c815b7442df5a3cb9fa113aecaf2de5f094222f7d74e57140719e486e8d2 \
  --expected-manifest-sha256 5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690 \
  --expected-parent-state-fingerprint fd580146d97c970226d4b1667305a92d7f04bfd4f01aa8afedfd7d826dcacde0 \
  --expected-parent-model-fingerprint 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-tokenizer-fingerprint eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --device cuda:0 \
  --dtype bfloat16 \
  --group-size 256 \
  --analysis-chunk-groups 4096 \
  > "$LOG" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf 'launched protected exact-50 allocator pid=%s root=%s\n' "$PID" "$ROOT"
