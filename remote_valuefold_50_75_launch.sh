#!/usr/bin/env bash
set -euo pipefail

export AUTHORIZE_LFM2_VALUE_FOLD_CONTROL=YES
SRC=/home/cereal/model-surgery-lab/neural-forge-src
PY=/home/cereal/model-surgery-lab/venv/bin/python
MODEL=/home/cereal/model-surgery-lab/models/LFM2.5-230M
ENVELOPE=/home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json
ROOT=/home/cereal/model-surgery-lab/artifacts/lfm2_value_fold_control_20260717/lfm2-230m-sealed64-b0a2c815b744-src-3b9fa3145aef-run-fda4a6553719-v1
LOG=${ROOT}.runner.log
PIDFILE=${ROOT}.pid
INPUTS=${ROOT}.launch-inputs.sha256
GPU=GPU-f5a09535-fc85-a027-50f7-734b344747dd

[ "${AUTHORIZE_LFM2_VALUE_FOLD_CONTROL:-}" = YES ] || {
  echo authorization_missing
  exit 3
}
[ ! -e "$ROOT" ] && [ ! -e "$LOG" ] && [ ! -e "$PIDFILE" ] && [ ! -e "$INPUTS" ] || {
  echo output_collision
  exit 4
}
mkdir -p "$(dirname "$ROOT")"
cd "$SRC"
sha256sum -c <<'HASHES'
3b9fa3145aefabc6ba7c1c67042dde9016ddd1a439538250151896c5e683f193  src/neural_forge/hf_lfm2_value_fold_control.py
fda4a655371944d2f96e3b37e32165fb54549619d8bf584ab6425d36dbd3be4d  experiments/run_lfm2_value_fold_control.py
ae813d5df788494e8aaa8b7b5f6d5bbbdd0dd24a770f5a61b37cbdb2a660b52f  tests/test_hf_lfm2_value_fold_control.py
cd3c2f328c320692cbe9c1d5d6fe3918c65579925845551566d9cf8db6c780d6  tests/test_run_lfm2_value_fold_control.py
a8922977266e879129184ea37cb2718f34f435b6f5f8eb8f6cddecb16451a01b  /home/cereal/model-surgery-lab/artifacts/lfm2_intelligence_holdout_v1/parent-envelope.json
f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd  /home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors
f7d0bcc454b7a30fa471b1e7b9e359e11fb25b56f5b4ffd59bb18248e3c2ea3d  /home/cereal/model-surgery-lab/models/LFM2.5-230M/config.json
4f88574c47c3215f7f952e1f520d1df7387422dde0345655228fb7b3fde6858c  /home/cereal/model-surgery-lab/models/LFM2.5-230M/generation_config.json
df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd  /home/cereal/model-surgery-lab/models/LFM2.5-230M/tokenizer.json
75c287923e252b08b0a0f1c367bbe557ab23a681d0b71c5a34e0932ddbe2f5ee  /home/cereal/model-surgery-lab/models/LFM2.5-230M/tokenizer_config.json
6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963  /home/cereal/model-surgery-lab/models/LFM2.5-230M/chat_template.jinja
HASHES
AVAILABLE=$(df --output=avail -B1 /home/cereal/model-surgery-lab | tail -1 | tr -d ' ')
[ "$AVAILABLE" -ge 1073741824 ] || {
  echo insufficient_disk
  exit 5
}
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fx "$GPU"; then
  echo gpu_busy
  exit 6
fi
nvidia-smi --id="$GPU" --query-gpu=uuid,name,memory.free --format=csv,noheader
"$PY" -m pytest -q tests/test_hf_lfm2_value_fold_control.py tests/test_run_lfm2_value_fold_control.py
sha256sum \
  src/neural_forge/hf_lfm2_value_fold_control.py \
  experiments/run_lfm2_value_fold_control.py \
  tests/test_hf_lfm2_value_fold_control.py \
  tests/test_run_lfm2_value_fold_control.py \
  "$ENVELOPE" \
  "$MODEL/model.safetensors" \
  "$MODEL/config.json" \
  "$MODEL/generation_config.json" \
  "$MODEL/tokenizer.json" \
  "$MODEL/tokenizer_config.json" \
  "$MODEL/chat_template.jinja" \
  > "$INPUTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
nohup "$PY" experiments/run_lfm2_value_fold_control.py \
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
  > "$LOG" 2>&1 &
printf '%s\n' "$!" > "$PIDFILE"
printf 'started pid=%s log=%s root=%s\n' "$(cat "$PIDFILE")" "$LOG" "$ROOT"
