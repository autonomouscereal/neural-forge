set -eu
cd /home/cereal/model-surgery-lab/neural-forge-src

OUT=/home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_structural_compression_20260717/h640_weightcomplete_v5_closedloop_l2_attention_ropefixed_smoke_v2
LOG=/home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_structural_compression_20260717/h640_weightcomplete_v5_closedloop_l2_attention_ropefixed_smoke_v2.log

if [ -e "$OUT" ] || [ -e "$LOG" ]; then
  echo "Refusing to overwrite existing smoke path" >&2
  exit 2
fi

nohup env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/home/cereal/model-surgery-lab/neural-forge-src/src:/home/cereal/model-surgery-lab/neural-forge-src \
  /home/cereal/model-surgery-lab/venv/bin/python \
  experiments/run_lfm2_weight_complete_closed_loop.py \
  --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --baseline-child /home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_structural_compression_20260717/h640_q10_kv5_ff1920_weightcomplete_v5_base \
  --trace-cache /home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache/parent/5e29a198be7b1716c45014ec24a16e528a5c56408ed869687d7ae8313cab7656 \
  --output "$OUT" \
  --layers 2 \
  --device cuda:0 \
  --dtype bfloat16 \
  --validation-fraction 0.25 \
  --split-seed lfm2-weightcomplete-v5-l2-attention-ropefixed-smoke-v2 \
  --minimum-output-kl-improvement -1000000 \
  --minimum-output-known-sequence-nll-improvement -1000000 \
  --maximum-output-eos-margin-rmse-regression 1000000 \
  > "$LOG" 2>&1 &

PID=$!
echo "pid=$PID"
echo "log=$LOG"
echo "output=$OUT"
