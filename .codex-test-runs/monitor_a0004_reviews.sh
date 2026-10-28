#!/usr/bin/env bash
set -euo pipefail

BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0004

for lane in fit_guided full_strength_structural residual_writer; do
  root="$BASE/$lane/terminal-frontier"
  progress="$root/whole-response-dossier.progress.json"
  pid=$(cat "$root/review.pid")
  state=exited
  if kill -0 "$pid" 2>/dev/null; then
    state=running
  fi
  completed=0
  if [[ -s "$progress" ]]; then
    completed=$(python3 -c \
      'import json,sys; print(len(json.load(open(sys.argv[1], encoding="ascii"))["completed_candidate_ids"]))' \
      "$progress")
  fi
  dossier=no
  if [[ -s "$root/whole-response-dossier.json" ]]; then
    dossier=yes
  fi
  printf '%s %s candidates=%s/36 responses=%s/576 dossier=%s\n' \
    "$lane" "$state" "$completed" "$((completed * 16))" "$dossier"
done

tail -n 6 "$BASE/pipeline-watch-v3.log"
