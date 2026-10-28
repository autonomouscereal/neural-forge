#!/usr/bin/env bash
set -euo pipefail

RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260720/961c957-c61-4x
mkdir -p "$RUN"
if [[ -e "$RUN/pid" ]]; then
  echo "pid file already exists" >&2
  exit 1
fi
nohup bash /home/cereal/model-surgery-lab/launch-semantic-foundation-961c957.sh \
  >"$RUN/stdout.log" 2>"$RUN/stderr.log" &
pid=$!
printf '%s\n' "$pid" >"$RUN/pid"
echo "launched $pid"
