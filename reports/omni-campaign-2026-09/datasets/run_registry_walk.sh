#!/bin/bash
# Detached driver for run-book item 0.5 (bounded registry checksum walk + Qwen
# token census).  CPU only, no GPU, no `find`, no `du`, no directory walks.
#
#   cd /home/cereal/omni-campaign/datasets && setsid nohup \
#     bash run_registry_walk.sh > runner.log 2>&1 < /dev/null &
#
# Stage 1 (plan) reads the 631 pointers + shards.jsonl and writes the declared
# payload byte total -- the denominator for the completion projection.
# Stage 2 (walk) hashes every shard file and tokenizes every row.  Both stages
# are resumable: re-running this script continues from the progress JSONL, and
# BOTH pass --resume -- neither stage will silently merge a progress file it was
# not told to continue.
set -u

D=/home/cereal/omni-campaign/datasets
PY=/home/cereal/model-surgery-lab/venv/bin/python
STAMP=2026-09-02
LOG=$D/registry-walk-$STAMP.log

# The tokenizer's rust backend will otherwise take every core on the box; the
# GPU lanes need their feeder threads.
export RAYON_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

echo "=== plan  start $(date -Is) ===" >> "$LOG"
timeout 10800 "$PY" "$D/registry_walk.py" --mode plan \
    --out "$D/registry-walk-plan-$STAMP.json" \
    --progress "$D/registry-walk-plan-$STAMP.progress.jsonl" \
    --resume >> "$LOG" 2>&1
echo "=== plan  exit $? $(date -Is) ===" >> "$LOG"

echo "=== walk  start $(date -Is) ===" >> "$LOG"
timeout 604800 "$PY" "$D/registry_walk.py" --mode walk \
    --out "$D/registry-walk-$STAMP.json" \
    --progress "$D/registry-walk-$STAMP.progress.jsonl" \
    --plan "$D/registry-walk-plan-$STAMP.json" \
    --tokenizer /home/cereal/models/Qwen3.5-4B \
    --chunk-timeout-s 300 --file-timeout-s 3600 \
    --checkpoint-every 5 --resume >> "$LOG" 2>&1
echo "=== walk  exit $? $(date -Is) ===" >> "$LOG"
