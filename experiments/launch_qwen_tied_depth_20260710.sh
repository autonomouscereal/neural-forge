#!/usr/bin/env bash
set -euo pipefail

python=/home/cereal/model-surgery-lab/venv/bin/python
repo=/home/cereal/model-surgery-lab/neural-forge-src
source=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
tokenizer=/mnt/ai_data/neural-forge/tokenizers/qwen3_4b_bytebpe_8192_20260710
suite=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json
parent=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json
gpu=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
root=/mnt/ai_data/neural-forge/tied_depth_controls_20260710
tag=qwen3_4b_to_tieddepth122m_a18_m18_gpu4_r1_20260710
output="$root/$tag"
log="$root/$tag.log"
pid_file="$root/$tag.pid"

cd "$repo"
PYTHONPATH="$repo/src" "$python" -m pytest tests/test_hf_tied_depth.py -q

if [[ -e "$output" || -e "$pid_file" ]]; then
  echo "Refusing to reuse existing run: $tag" >&2
  exit 3
fi
mkdir -p "$root"

CUDA_VISIBLE_DEVICES="$gpu" \
PYTHONPATH="$repo/src" \
PYTHONUNBUFFERED=1 \
nohup "$python" "$repo/experiments/run_qwen_tied_depth_in_memory.py" \
  --source-model "$source" \
  --target-tokenizer "$tokenizer" \
  --output-dir "$output" \
  --eval-suite "$suite" \
  --cached-parent-eval "$parent" \
  --attention-source-layer 18 \
  --mlp-source-layer 18 \
  --dtype float16 \
  --device cuda:0 \
  --max-new-tokens 96 \
  --max-text-evals 1000 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000 \
  --manual-sample-count 64 \
  --local-files-only \
  </dev/null >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Run exited during launch: $tag" >&2
  tail -80 "$log" >&2
  exit 4
fi
printf 'launched pid=%s tag=%s fd0=%s\n' \
  "$pid" "$tag" "$(readlink "/proc/$pid/fd/0")"
