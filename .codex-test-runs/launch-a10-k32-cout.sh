set -euo pipefail

SCRIPT=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/launch-a10-k32-cout.sh
RUNTIME=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a10-k32-cout-lambda0001
PYTHON=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python

if [ "${1:-}" = "--worker" ]; then
  mkdir -p "$RUN"
  exec >>"$RUN/job.log" 2>&1
  export CUDA_VISIBLE_DEVICES=4
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="$RUNTIME/src"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd "$RUNTIME"
  printf '{"event":"worker-start","utc":"%s","cuda_visible_devices":"4"}\n' "$(date -u +%FT%TZ)"
  set +e
  "$PYTHON" -u experiments/run_lfm2_functional_output_refit.py \
    --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
    --expected-parent-source-inspection-sha256 ce8c860c0323f6fe5a445e7dabf902b0bb666da197d1069d5dfaadabba6f7559 \
    --parent-tensor-key model.embed_tokens.weight \
    --checkpoint-directory /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723/89cd45b0/same-body-v4096-p40-gpu1-a0001/checkpoint \
    --expected-checkpoint-artifact-inventory-sha256 41007288f9a1a7b9dcdf96b4d1d39a1eeec416c46fb928b5bb2cc35ea9cbf8b2 \
    --target-tensor-key model.embed_tokens.weight \
    --codebook-directory /home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/5b5557b8da7aa125ada8cac305005de7b9e157933c3d4878f6be133fb5c51cc7/payload/artifact \
    --expected-codebook-artifact-sha256 4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d \
    --expected-codebook-donor-row-count 64402 \
    --expected-codebook-target-row-count 4096 \
    --expected-hidden-size 1024 \
    --trace-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-trace-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-trace-parent-fingerprint 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
    --expected-trace-batch-count 16 \
    --expected-trace-row-count 648 \
    --output-directory "$RUN" \
    --holdout-modulus 4 \
    --holdout-remainder 3 \
    --functional-weight 0.001 \
    --solve-ridge 0.0001 \
    --metric-device cuda:0 \
    --metric-dtype fp32 \
    --solve-device cuda:0 \
    --solve-dtype fp32 \
    --rows-per-chunk 512 \
    --score-device cuda:0 \
    --score-dtype fp32 \
    --score-rows-per-chunk 64
  code=$?
  set -e
  printf '{"event":"worker-exit","utc":"%s","exit_code":%s}\n' "$(date -u +%FT%TZ)" "$code"
  printf '%s\n' "$code" >"$RUN/job.status"
  exit "$code"
fi

mkdir -p "$RUN"
if find "$RUN" -maxdepth 1 -type f -name 'run-*.json' -print -quit | grep -q .; then
  printf '{"event":"launch-skipped","reason":"sealed-run-exists","run":"%s"}\n' "$RUN"
  exit 0
fi
if [ -f "$RUN/job.pid" ]; then
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
