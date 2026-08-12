#!/bin/bash
# Serial child queue on one GPU.  Each entry: CELL:RECIPE:CORRECTIONS(0|1)
set -u
cd /home/cereal/nf-mxq
for spec in "$@"; do
  CELL="${spec%%:*}"; rest="${spec#*:}"
  RECIPE="${rest%%:*}"; CORR="${rest#*:}"
  echo "=== $(date -u +%H:%M:%S) queue start $CELL corrections=$CORR"
  CORRECTIONS=$CORR ./drive.sh "$CELL" M-A /home/cereal/nf-mxq/recipes/$RECIPE 4
  echo "=== $(date -u +%H:%M:%S) queue done  $CELL"
done
