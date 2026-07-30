#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/launch-a12-teacher-forced-trajectory.sh
RUNTIME=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_teacher_forced_trajectory_20260730/a12-k32-split-cout-rtx8000-fp32
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python
CODEBOOK=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/5b5557b8da7aa125ada8cac305005de7b9e157933c3d4878f6be133fb5c51cc7/payload/artifact
COUT=/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a10-k32-cout-lambda0001/math-cache/75cf54bc04438952a033ea494a6a269c1abd414cdb62a4b87c9bc77b73c57284
SPAN=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-semantic-step-global-raw-span/3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/payload/span-evidence.json
PROMPTS="$RUNTIME/research/lfm2_semantic_whole_response_prompts_v1.json"

if [[ "${1:-}" == "--worker" ]]; then
  mkdir -p "$RUN"
  exec >>"$RUN/job.log" 2>&1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$RUNTIME/src"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  cd "$RUNTIME"
  printf '{"event":"worker-start","utc":"%s","physical_gpu_uuid":"GPU-f5a09535-fc85-a027-50f7-734b344747dd","visible_runtime_device":"cuda:0","requested_runtime_dtype":"bfloat16","expected_effective_runtime_dtype":"float32"}\n' "$(date -u +%FT%TZ)"
  set +e
  "$PYTHON" -u experiments/run_lfm2_teacher_forced_trajectory_audit.py \
    --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --parent-tensor-key model.embed_tokens.weight \
    --expected-parent-source-file-sha256 f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd \
    --expected-parent-table-sha256 3d7f57f485f704026386589246fc000b11f193bb2fb63c3a424351f2806aec47 \
    --checkpoint-directory /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723/89cd45b0/same-body-v4096-p40-gpu1-a0001/checkpoint \
    --expected-checkpoint-inventory-sha256 41007288f9a1a7b9dcdf96b4d1d39a1eeec416c46fb928b5bb2cc35ea9cbf8b2 \
    --expected-checkpoint-staged-state-sha256 0a0550400e0dcc716c5356f5f5240ca5cb605772d0d836ffb0953e152a5a8323 \
    --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
    --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
    --span-evidence-json "$SPAN" \
    --expected-span-evidence-sha256 134f1ccb0ac2b2b69829ea483263bb4d92c1783e9b809f5c4a87c4ff10528cf0 \
    --codebook-directory "$CODEBOOK" \
    --expected-codebook-artifact-sha256 4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d \
    --output-coefficients-safetensors "$COUT/output-coefficients.safetensors" \
    --output-coefficients-key output_semantic_code_coefficients \
    --output-refit-receipt-json "$COUT/result.json" \
    --expected-output-refit-cache-key 75cf54bc04438952a033ea494a6a269c1abd414cdb62a4b87c9bc77b73c57284 \
    --expected-output-refit-result-sha256 b301539b49e0156e5854e3d84730ead019b32cadd1d0436e017474556408bfc8 \
    --trace-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-trace-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-trace-parent-fingerprint 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
    --expected-trace-batch-count 16 \
    --prompt-labels-json "$PROMPTS" \
    --donor-row-count 64402 \
    --target-row-count 4096 \
    --hidden-size 1024 \
    --provenance-device cpu \
    --provenance-dtype bfloat16 \
    --runtime-device cuda:0 \
    --runtime-dtype bfloat16 \
    --metric-dtype float64 \
    --score-dtype float32 \
    --embedding-rows-per-chunk 1024 \
    --donor-score-rows-per-chunk 4096 \
    --output-directory "$RUN"
  code=$?
  set -e
  printf '{"event":"worker-exit","utc":"%s","exit_code":%s}\n' "$(date -u +%FT%TZ)" "$code"
  printf '%s\n' "$code" >"$RUN/job.status"
  exit "$code"
fi

mkdir -p "$RUN"
if find "$RUN" -mindepth 2 -maxdepth 2 -type f -name audit.json -print -quit | grep -q .; then
  printf '{"event":"launch-skipped","reason":"sealed-audit-exists","run":"%s"}\n' "$RUN"
  exit 0
fi
if [[ -f "$RUN/job.pid" ]]; then
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
