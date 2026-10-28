set -euo pipefail
set -o noclobber

cd /home/cereal/model-surgery-lab/neural-forge-src

OUT=/home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_folds_20260717/lfm2_230m_to_v4096_100x_cpcf_v3
LOG=${OUT}.log
RECEIPT=${OUT}.receipt.json

[ ! -e "$OUT" ] && [ ! -L "$OUT" ]
[ ! -e "$LOG" ] && [ ! -L "$LOG" ]
[ ! -e "$RECEIPT" ] && [ ! -L "$RECEIPT" ]

if find "$(dirname "$OUT")" -maxdepth 1 \
  -name ".$(basename "$OUT").stage-*" -print -quit | grep -q .; then
  printf 'REFUSE_STALE_STAGE\n'
  exit 41
fi

[ "$(sha256sum src/neural_forge/hf_lfm2_cpcf_model_compiler.py | cut -d' ' -f1)" = \
  "7fb5f8d83e9e9ac59dc2e29bd8c00003bf48391bbad8a9ecfa043aa477044e3b" ]
[ "$(sha256sum src/neural_forge/hf_lfm2_cpcf_layer_compiler.py | cut -d' ' -f1)" = \
  "6b3c0722e0a57139cbc9d5c81619d2248d2f334675d902ab18a2f3e63dfa4456" ]
[ "$(sha256sum src/neural_forge/hf_lfm2_cpcf_model_fold.py | cut -d' ' -f1)" = \
  "0cd2873bf75324d2add7f0eb5798128a35c7cbf692ba6a70774cbec631fdd5ce" ]
[ "$(sha256sum src/neural_forge/hf_lfm2_structural_residual_attention_fold.py | cut -d' ' -f1)" = \
  "b53999d8b68b07d3200fdb56389e65ca350cb3151bbad0c3b53aef0779b03165" ]
[ "$(sha256sum experiments/run_lfm2_cpcf_blank_fold.py | cut -d' ' -f1)" = \
  "bec1b0226cdbe1ac942b7c825d8e45d8d90e28b1780fd95fb4aca908e78f503a" ]

! pgrep -f '[r]un_lfm2_cpcf_blank_fold.py' >/dev/null

nohup env \
  CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/home/cereal/model-surgery-lab/neural-forge-src/src \
  /home/cereal/model-surgery-lab/venv/bin/python -u \
  experiments/run_lfm2_cpcf_blank_fold.py \
  --parent-model-directory \
  /home/cereal/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots/37b30cce3446f3f2e26a0d3f8c67c9167f5079d7 \
  --donor-tokenizer-directory \
  /home/cereal/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots/37b30cce3446f3f2e26a0d3f8c67c9167f5079d7 \
  --target-tokenizer-directory \
  /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --bridge-directory \
  /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --trace-cache-directory \
  /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --trajectory-corpus \
  /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_folding_corpora_20260717/v4096_broad256_v1/trajectories.json \
  --output-directory "$OUT" \
  --receipt "$RECEIPT" \
  --compression-ratio 100 \
  --device cuda \
  --dtype bfloat16 \
  --blank-seed 0 \
  --minimum-hidden-size 32 \
  --maximum-candidates 8 \
  --selected-candidate-rank 0 \
  --ridge 1e-8 \
  --boundary-state-reconstruction-weight 0.05 \
  --boundary-predictive-energy-floor 0.90 \
  --boundary-validation-retention-floor 0.0 \
  --evaluation-limit 50 \
  --evaluation-max-new-tokens 128 \
  >"$LOG" 2>&1 </dev/null &

printf '%s\n' "$!"
