#!/bin/bash
# lane gpu6: after the queue writes the B2 verdict -- PASS: exit; FAIL: run the staged per-layer sha bisect
# (bisect_k3_gpu6.sh waits until the queue has released GPU6) and record start/end in gpu6.status.
Q=/home/cereal/omni-campaign/distill/queues
until grep -qE " b2 (PASS|FAIL) " $Q/gpu6.status; do sleep 120; done
if grep -q " b2 FAIL " $Q/gpu6.status; then
  echo "$(date -u +%FT%TZ) b2-bisect START auto after b2 FAIL: $Q/logs/gpu6/bisect.log" >> $Q/gpu6.status
  bash $Q/bisect_k3_gpu6.sh 60 > $Q/logs/gpu6/bisect.log 2>&1
  echo "$(date -u +%FT%TZ) b2-bisect END $(grep "^BISECT" $Q/logs/gpu6/bisect.log | tail -1)" >> $Q/gpu6.status
else
  echo "$(date -u +%FT%TZ) trigger: b2 PASS, no bisect"
fi
