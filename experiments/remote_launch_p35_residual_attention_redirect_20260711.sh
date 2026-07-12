#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-src
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python
TOKENIZER=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
INCUMBENT=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child
OUTDIR=/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711
TAG=p35_redirect_L19G4_to_L21G6_h4_p2
REPORT="$OUTDIR/$TAG.json"
LOG="$OUTDIR/$TAG.log"
PIDFILE="$OUTDIR/$TAG.pid"

mkdir -p "$OUTDIR"
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cut -d' ' -f1 "$PIDFILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "probe already active pid=$old_pid" >&2
    exit 2
  fi
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 \
  PYTHONPATH="$ROOT/src" nohup "$PYTHON" \
  "$ROOT/experiments/run_qwen_residual_attention_probe.py" \
  --codebook-checkpoint "$INCUMBENT" \
  --tokenizer "$TOKENIZER" \
  --candidate 19:4=21:6 \
  --prompt "The capital of France is" \
  --prompt "Complete the Python function: def add(a, b):" \
  --output "$REPORT" \
  --device cuda:0 \
  --dtype float16 \
  --horizon 4 \
  --compensation both \
  --ridge 0.001 \
  --max-prompts 2 \
  --max-affected-sites 16 \
  --max-affected-layers 8 \
  --logit-rtol 0.01 \
  --logit-atol 0.04 \
  --context-rtol 0.008 \
  --context-atol 0.008 \
  --local-files-only \
  >"$LOG" 2>&1 &

pid=$!
start_ticks="$(awk '{print $22}' "/proc/$pid/stat")"
printf '%s %s\n' "$pid" "$start_ticks" >"$PIDFILE"
echo "pid=$pid start_ticks=$start_ticks log=$LOG report=$REPORT"
