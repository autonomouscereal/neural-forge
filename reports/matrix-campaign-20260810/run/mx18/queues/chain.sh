#!/bin/bash
# Wait for a marker line in a log, then run the review queue on one GPU.
set -u
. /home/cereal/nf-mx18/env.sh
WAITLOG="$1"; MARKER="$2"; GPU="$3"; shift 3
for i in $(seq 1 400); do
  grep -q "$MARKER" "$WAITLOG" 2>/dev/null && break
  sleep 15
done
exec /home/cereal/nf-mx18/reviewq.sh $GPU "$@"
