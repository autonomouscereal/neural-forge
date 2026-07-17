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

verify_sha f57a42c3b4d0630dc1e68139d40a5946a359a9912c79e5077cd56a148ad04b25 "$SRC/experiments/build_lfm2_cached_causal_evidence.py"
verify_sha 34afbdde8966ce664245bfd2f29a8dacd9f496824e201325434b88bff16b25e2 "$SRC/src/neural_forge/hf_lfm2_cached_causal_evidence.py"
verify_sha 0afdad925899c7365fb61307af3629ea3c0836f4f1b06f0d0ee057a083b111a8 "$SRC/src/neural_forge/hf_lfm2_global_causal_allocator.py"
verify_sha 9571646b41f47cd35c69eed9f1fdf657b89f46563e1b1e24619ce000f267a677 "$SRC/src/neural_forge/hf_lfm2_variable_width_fold.py"
verify_sha e85d2598a25ef3b2cde610dd9d64424c1490259fad844b6222b622c235f3c779 "$SRC/tests/test_hf_lfm2_global_causal_allocator.py"
verify_sha 282ba9851a973979bb44c78d37cf0c36d74c128c784e08836737541950cb5344 "$PARENT_ENVELOPE"

export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SRC"
"$PYTHON" -m pytest -q tests/test_hf_lfm2_global_causal_allocator.py

test "$(nvidia-smi --id=4 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
test "$(nvidia-smi --id=4 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')" -ge 20000
if test -f "$RUN_ROOT/run.pid"; then
  OLD_PID=$(cat "$RUN_ROOT/run.pid")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    printf 'producer already active pid=%s\n' "$OLD_PID"
    exit 1
  fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '\nRESTART exact-parent-identity-audit=%s post-audit-rollback=%s\n' 0afdad925899c7365fb61307af3629ea3c0836f4f1b06f0d0ee057a083b111a8 f57a42c3b4d0630dc1e68139d40a5946a359a9912c79e5077cd56a148ad04b25 >> "$RUN_ROOT/run.log"
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
  >> "$RUN_ROOT/run.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_ROOT/run.pid"
printf 'restarted producer pid=%s root=%s\n' "$PID" "$RUN_ROOT"
