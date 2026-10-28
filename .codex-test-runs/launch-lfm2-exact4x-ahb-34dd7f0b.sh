#!/usr/bin/env bash
set -euo pipefail

LAB=/home/cereal/model-surgery-lab
RUNTIME=$LAB/deployments/neural-forge-da5b9da6
SOURCE_COMMIT=da5b9da6ac3000213538a0b55624aeaa7ab7685f
RUN=${RUN:-$LAB/artifacts/neural_forge_exact4x_ahb_20260808/a0007-lfm2-exact4x-h816-ahb-fold}
CHILD_NAME=${CHILD_NAME:-$(basename "$RUN")}
AHB_ROOT=$LAB/artifacts/neural_forge_ahb_cache_local_v1/lfm2-a0003-dd9efbe2-lags-0-1-2-4-8-16
AHB_SOURCE=$LAB/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b

ROUND_ROOT=$LAB/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12
ROUND_MANIFEST=$ROUND_ROOT/round-000/round-manifest.json
ROUND_BASE_CACHE=$ROUND_ROOT/base-cache-recaptured-w12e/c29b67370d5e33be47a11daccb1e6c2144d6e85efb024621458dc14ca5bd5ef7
REBOUND=$ROUND_ROOT/bridge-bound-outputs-recaptured-w13d
PACK=$LAB/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5
DONOR=$LAB/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46
BRIDGE=$LAB/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2
CARRIER=$LAB/artifacts/neural_forge_causal_support_20260730/a9-a11-base-dual-route/4439a2b7-b301539b
SPAN=$LAB/artifacts/neural_forge_phase_cache_v1/lfm2-semantic-step-global-raw-span/3e780d07ac1034ee77756f973b483637cee846e17e6d7191d6b20c9533360167/payload/span-evidence.json
PARENT=$LAB/models/LFM2.5-230M
PROMPTS=$RUNTIME/research/lfm2_semantic_whole_response_prompts_v1.json
DERIVED=$LAB/artifacts/neural_forge_phase_cache_v1/lfm2-global-topology-derived
BALANCED_BASIS=$LAB/artifacts/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1
BALANCED_BASIS_EVENTS=$RUN/balanced-basis-phase-events.jsonl
BALANCED_BASIS_HMAC_KEY=$LAB/keys/neural-forge-balanced-basis-phase-v1.key
PYTHON=$LAB/venv/bin/python

test "$(cat "$RUNTIME/SOURCE_COMMIT")" = "$SOURCE_COMMIT"
test ! -e "$RUN"
test ! -e "$RUN.pid"
test -f "$ROUND_MANIFEST"
test -f "$ROUND_BASE_CACHE/manifest.json"
test -f "$REBOUND/canonical_parent_outputs_bridge_bound.json"
test -f "$REBOUND/mapped_parent_outputs.json"
test -f "$AHB_SOURCE/manifest.json"
test -f "$AHB_SOURCE/complete.json"
test -f "$AHB_ROOT/COPY_COMPLETE"
test -f "$BALANCED_BASIS_HMAC_KEY"
test -f "$PROMPTS"
test "$(sha256sum "$PROMPTS" | awk '{print $1}')" = \
  0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["generation"]["max_new_tokens"])' "$REBOUND/canonical_parent_outputs_bridge_bound.json")" = 64

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  printf 'preflight-ok source=%s run=%s ahb=%s\n' \
    "$SOURCE_COMMIT" "$RUN" "$AHB_ROOT"
  exit 0
fi

mkdir -p "$(dirname "$RUN")" "$AHB_ROOT"
mkdir "$RUN"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN/source-commit.txt"

export PYTHONPATH=$RUNTIME/src:$RUNTIME
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561,GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d,GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4,GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8,GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

nohup "$PYTHON" -u "$RUNTIME/experiments/run_lfm2_dynamic_catalog.py" \
  --parent-model "$PARENT" \
  --tokenizer-directory "$DONOR" \
  --target-tokenizer-directory "$BRIDGE" \
  --target-tokenizer-calibration "$PACK/target_tokenized_calibration.json" \
  --tokenizer-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --tokenizer-bridge-profile-sha256 e3fc98c666b70b367962e400435afe7588a9f1a2b13619dffc6f40c6458c5ca1 \
  --tokenizer-bridge-mapped-parent-outputs "$REBOUND/mapped_parent_outputs.json" \
  --tokenizer-bridge-mapped-parent-outputs-sha256 4fea48e1494fa0151f4b58d64824b2e8dac0829373183678537e57de3bf0989e \
  --tokenizer-bridge-minimum-parent-task-retention 0 \
  --calibration "$PACK/mapped_donor_calibration.json" \
  --sealed-parent-cache-directory "$ROUND_BASE_CACHE" \
  --cached-parent-outputs "$REBOUND/canonical_parent_outputs_bridge_bound.json" \
  --cached-parent-outputs-sha256 fa97d9a8d70f2974832c4971f517eb64d00bd62095946f80ad2f0db2b696cafa \
  --on-policy-round-manifest "$ROUND_MANIFEST" \
  --on-policy-max-rounds 2 \
  --carrier-route-publication "$CARRIER" \
  --carrier-route-artifact-sha256 860bccfe6be46e543f67c8514f3f3ff7dc0d729eb82267f66a9ce1766e43d7bb \
  --carrier-span-evidence "$SPAN" \
  --carrier-span-evidence-file-sha256 082ef4c782b818f31f72629c71de576cef1e1e5a1f9dcaea28bd2287dec9e6f2 \
  --carrier-span-evidence-sha256 134f1ccb0ac2b2b69829ea483263bb4d92c1783e9b809f5c4a87c4ff10528cf0 \
  --semantic-source-tokenizer-directory "$DONOR" \
  --routed-manual-prompts "$PROMPTS" \
  --routed-manual-prompts-sha256 0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b \
  --routed-manual-dossier "$RUN/routed-manual-dossier.json" \
  --routed-manual-checkpoint-directory "$RUN/routed-manual-checkpoints" \
  --counted-target-total-parameters 57420144 \
  --counted-dense-body-parameters 53552368 \
  --carrier-expected-target-rows 4096 \
  --carrier-expected-hidden-size 816 \
  --routed-smoke-maximum-prompt-tokens 512 \
  --routed-smoke-max-new-tokens 2 \
  --routed-manual-maximum-prompt-tokens 8192 \
  --routed-manual-initial-max-new-tokens 1024 \
  --routed-manual-retry-max-new-tokens 2048 \
  --cache-root "$RUN/cache" \
  --derived-global-topology-cache-root "$DERIVED" \
  --balanced-basis-phase-cache-root "$BALANCED_BASIS" \
  --balanced-basis-phase-event-log "$BALANCED_BASIS_EVENTS" \
  --balanced-basis-phase-hmac-key-file "$BALANCED_BASIS_HMAC_KEY" \
  --balanced-basis-phase-hmac-key-id lfm2-circuit-balanced-residual-basis-v1 \
  --autoregressive-hankel-cache-root "$AHB_ROOT" \
  --autoregressive-hankel-source-cache-directory "$AHB_SOURCE" \
  --dense-autoregressive-hankel-future-lags 0,1,2,4,8,16 \
  --dense-autoregressive-hankel-state-ridge-fraction 1e-8 \
  --dense-autoregressive-hankel-future-ridge-fraction 1e-8 \
  --dense-autoregressive-hankel-maximum-condition-number 1e10 \
  --snapshot-root "$RUN/snapshots" \
  --publish-destination "$RUN/published-child" \
  --ledger "$RUN/ledger.jsonl" \
  --receipt "$RUN/receipt.json" \
  --diagnostics "$RUN/diagnostics.json" \
  --parent-name parent \
  --child-name "$CHILD_NAME" \
  --device cuda:0 \
  --dtype float32 \
  --compression-ratio 4 \
  --dense-minimum-hidden-size 32 \
  --required-target-hidden-size 816 \
  --dense-heterogeneous-layout \
  --dense-heterogeneous-global-hidden-frontier \
  --dense-heterogeneous-factor-writer-rank-discovery \
  --dense-heterogeneous-global-functional-rows-per-role 2048 \
  --dense-heterogeneous-target-parameters 53552368 \
  --dense-residual-basis-mode circuit_balanced \
  --dense-run-closed-loop \
  --dense-body-reclosure-mode moving_gauge_v3 \
  --dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement \
  --dense-moving-gauge-boundary-regression-tolerance 1e-6 \
  --dense-moving-gauge-install-verification fingerprint \
  --dense-closed-loop-replay-mode prefix_by_prefix_canonical \
  --dense-closed-loop-performance-metrics-diagnostic-only \
  --dense-closed-loop-merge-adjacent-layer-captures \
  --dense-closed-loop-terminal-composition-floor \
  --dense-closed-loop-terminal-regression-tolerance 0.25 \
  --dense-closed-loop-terminal-absolute-floor-minimum-layer 0 \
  --dense-closed-loop-terminal-free-run-tripwire \
  --dense-closed-loop-input-error-multiplier 0 \
  --dense-closed-loop-solver-conditioning-diagnostics \
  --dense-closed-loop-gate-operating-point-diagnostics \
  --dense-closed-loop-depth-telemetry \
  --dense-replay-devices cuda:0,cuda:4 \
  --dense-restart-capsule-root "$RUN/restart-capsules" \
  --dense-restart-capsule-keep 2 \
  --dense-attention-consumer-mode coupled-value-output \
  --dense-attention-geometry-risk \
  --dense-vocabulary-margin-observability \
  --dense-streaming-vocabulary-margin-v2 \
  --dense-streaming-vocabulary-margin-pair-chunk-size 256 \
  --sample-last-tokens 256 \
  --max-decode-steps 8 \
  --max-capture-bytes 536870912 \
  --child-decode-trace-mode full_prefix_canonical \
  --capture-logit-rtol 0.2 \
  --capture-logit-atol 1 \
  --max-selection-rows 13728 \
  --max-fit-rows 40656 \
  --max-validation-rows 13199 \
  --max-endpoint-relative-rmse 1000000 \
  --min-endpoint-cosine-similarity -1 \
  --min-teacher-top1 0 \
  --min-free-run-match 0 \
  --min-residual-cosine -1 \
  --whole-output-max-new-tokens 64 \
  --whole-output-maximum-prompt-tokens 512 \
  --whole-output-fragment-minimum-characters 12 \
  --whole-output-maximum-new-empty-responses 28 \
  --whole-output-maximum-new-fragment-responses 28 \
  --whole-output-minimum-parent-nonempty-retention 0 \
  --whole-output-minimum-task-score-retention 0 \
  --whole-output-maximum-parent-task-score-regressions 28 \
  --whole-output-maximum-new-generation-limit-saturations 28 \
  --whole-output-maximum-repeated-ngram-share-regressions 28 \
  --whole-output-maximum-repeated-bigram-share-delta 1 \
  --final-performance-metrics-diagnostic-only \
  > "$RUN/run.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$RUN/run.pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$pid" "$RUN/run.log"
