#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PY="$LAB/venv/bin/python"
MODEL="$LAB/models/LFM2.5-230M"
ENVELOPE="$LAB/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json"
ROOT="$LAB/artifacts/lfm2_exact50_all8_20260717/lfm2-230m-all8-independent64-v2-rotary-exact"
LOG="$ROOT.runner.log"
PID_FILE="$ROOT.runner.pid"
GPU_UUID=GPU-f5a09535-fc85-a027-50f7-734b344747dd

verify_sha() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c -
}

verify_sha 3f98626eafcd6716ebd8887b77ba0c121001301584ffa684ee83146e39995ee5 \
  "$SRC/src/neural_forge/hf_lfm2_exact50_all8.py"
verify_sha f02e055abef2932ed7de91e2456e32467e2b425176b995329d488185a49d7e00 \
  "$SRC/src/neural_forge/hf_lfm2_value_fold_control.py"
verify_sha c989ff6e4c76a0e182668a35079cf927898d8bc5884b9910920c7721f373abfe \
  "$SRC/experiments/run_lfm2_exact50_all8.py"
verify_sha c8d86afaf8ea79779ecefccfbd5d173787fb63be77c4ce74550027666f89423d \
  "$SRC/experiments/finalize_lfm2_exact50_all8.py"
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
nohup "$PY" experiments/run_lfm2_exact50_all8.py \
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
printf 'launched uniform-all8 exact-50 v2 rotary-exact pid=%s root=%s\n' "$PID" "$ROOT"
