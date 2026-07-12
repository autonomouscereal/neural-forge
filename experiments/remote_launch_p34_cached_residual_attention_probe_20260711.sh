#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-src
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python
DENSE=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
CHILD=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa34_gpu6_r1_20260711/compact_child
OUTDIR=/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711
REPORT="$OUTDIR/p34_forced_cache_smoke.json"
LOG="$OUTDIR/p34_forced_cache_smoke.log"
PIDFILE="$OUTDIR/p34_forced_cache_smoke.pid"

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
  "$ROOT/experiments/run_qwen_cached_residual_attention_probe.py" \
  --dense-model "$DENSE" \
  --codebook-checkpoint "$CHILD" \
  --output "$REPORT" \
  --prompt "The capital of France is" \
  --continuation " Paris. It is a major city in Europe." \
  --prompt "Complete the Python function: def add(a, b):" \
  --continuation $'\n    return a + b\n' \
  --horizon 4 \
  --device cuda:0 \
  --dtype float16 \
  --logit-rtol 0.01 \
  --logit-atol 0.04 \
  >"$LOG" 2>&1 &

pid=$!
start_ticks="$(awk '{print $22}' "/proc/$pid/stat")"
printf '%s %s\n' "$pid" "$start_ticks" >"$PIDFILE"
echo "pid=$pid start_ticks=$start_ticks log=$LOG report=$REPORT"
