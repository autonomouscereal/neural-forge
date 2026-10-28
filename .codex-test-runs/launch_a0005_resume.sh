set -euo pipefail

CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-38c6d3c31e93b81793592ff38e44e760005cf11a
SCRIPT=/tmp/resume_a0005_frontier_review.py
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0005
PROMPTS=$CODE/research/lfm2_semantic_whole_response_prompts_v1.json

test -f "$SCRIPT"
test -f "$PROMPTS"

launch_resume() {
  local lane=$1
  local gpu=$2
  local run=$BASE/$lane
  local log=$BASE/logs/$lane.resume.log
  local pidfile=$BASE/logs/$lane.resume.pid
  test -d "$run/preterminal-capsule"
  test -d "$run/terminal-frontier-bank"
  test -f "$run/postfold-receipt.json"
  test ! -e "$run/materializations"
  test ! -e "$run/frontier-receipt.json"
  test ! -e "$run/whole-response-dossier.json"
  test ! -e "$log"
  test ! -e "$pidfile"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$CODE:$CODE/src" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    /home/cereal/model-surgery-lab/venv/bin/python -u \
    "$SCRIPT" "$run" --device cuda:0 --prompts-json "$PROMPTS" \
    >"$log" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$pidfile"
  sleep 2
  kill -0 "$pid"
  printf '%s gpu=%s pid=%s\n' "$lane" "$gpu" "$pid"
}

launch_resume source_producer_residual_writer_anchor_0 4
launch_resume full_strength_structural_anchor_0 0
