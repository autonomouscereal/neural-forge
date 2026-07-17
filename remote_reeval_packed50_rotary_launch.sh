#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PY="$LAB/venv/bin/python"
MODEL="$LAB/models/LFM2.5-230M"
ENVELOPE="$LAB/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json"
BASE_CHECKPOINT="$LAB/artifacts/lfm2_value_fold_control_20260717/lfm2-230m-sealed64-b0a2c815b744-src-3b9fa3145aef-run-fda4a6553719-v1/target-50pct/checkpoint"
PROTECTED_CHECKPOINT="$LAB/artifacts/lfm2_protected_weight_error_20260717/lfm2-230m-exact50-independent64-v1/checkpoint"
ROOT="$LAB/artifacts/lfm2_packed_rotary_reeval_20260717"
BASE_OUTPUT="$ROOT/value50-corrected-rotary-v1"
PROTECTED_OUTPUT="$ROOT/protected50-corrected-rotary-v1"
LOG="$ROOT/runner.log"
PID_FILE="$ROOT/runner.pid"
GPU_UUID=GPU-f5a09535-fc85-a027-50f7-734b344747dd

verify_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}

verify_sha 4c2491afeafcc53152c097a0d109abcb39d77a10c8f6d82a2435f28d8b0fb8bb \
  "$SRC/experiments/reevaluate_lfm2_packed_checkpoint.py"
verify_sha f02e055abef2932ed7de91e2456e32467e2b425176b995329d488185a49d7e00 \
  "$SRC/src/neural_forge/hf_lfm2_value_fold_control.py"
verify_sha 27886a1215a2b46298db5a6096a57a488b9b948d97cdb0f5938cccb37577e9b2 \
  "$SRC/src/neural_forge/hf_lfm2_protected_weight_error_allocator.py"
verify_sha a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b \
  "$ENVELOPE"

test ! -e "$ROOT"
mkdir -p "$ROOT"

unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
test "$(nvidia-smi --id=6 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
test "$(nvidia-smi --id=6 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')" -ge 40000

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common_args=(
  --tokenizer-directory "$MODEL"
  --parent-envelope "$ENVELOPE"
  --expected-parent-envelope-file-sha256 a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b
  --expected-parent-envelope-sha256 b0a2c815b7442df5a3cb9fa113aecaf2de5f094222f7d74e57140719e486e8d2
  --expected-manifest-sha256 5a4c4031045a7cfeaa46f38fdd243c383a4e737dfef6f127a195edef1ad11690
  --expected-parent-state-fingerprint fd580146d97c970226d4b1667305a92d7f04bfd4f01aa8afedfd7d826dcacde0
  --expected-parent-model-fingerprint 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb
  --expected-tokenizer-fingerprint eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46
  --device cuda:0
)

cd "$SRC"
(
  "$PY" experiments/reevaluate_lfm2_packed_checkpoint.py \
    --checkpoint-kind value_control \
    --checkpoint "$BASE_CHECKPOINT" \
    --output-root "$BASE_OUTPUT" \
    --expected-checkpoint-manifest-sha256 7c9645cd0ada58a0d881bec2c8fbb983381ee51d8b6c4849c2a5cde2f0f158ea \
    "${common_args[@]}"
  "$PY" experiments/reevaluate_lfm2_packed_checkpoint.py \
    --checkpoint-kind protected_weight_error \
    --checkpoint "$PROTECTED_CHECKPOINT" \
    --output-root "$PROTECTED_OUTPUT" \
    --expected-checkpoint-manifest-sha256 3241c6199522855e3df5e17a3ca854d5e395cf78886be271af7e421600365413 \
    "${common_args[@]}"
) > "$LOG" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf 'launched corrected packed50 reevaluations pid=%s root=%s\n' "$PID" "$ROOT"
