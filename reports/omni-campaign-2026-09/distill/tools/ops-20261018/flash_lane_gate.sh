#!/bin/bash
# flash_lane_gate.sh -- start flash_lane.sh (P40 tool-call generation, then Flash Next logits) only when (1) the GGUF is
# verified on root (flash_gguf_stage.sh wrote the OK mark) and (2) run 1's current segment has logged >= 2 steps with no
# REFUSED/Traceback.  Cyril 2026-10-18: Flash Next "[Cyril quote withheld]".
set -u
OC=/home/cereal/omni-campaign; T=$OC/distill/tools
MARK=/home/cereal/cleanup-20261017/flash_root.sha256.ok; SHA=cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6
SEG=${SEG:-/mnt/lexar/omni-campaign/distill/fullkd/run1/train-003.log}
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*"; }
while :; do
  ok_gguf=0; grep -q "$SHA" "$MARK" 2> /dev/null && ok_gguf=1
  if [ $ok_gguf = 0 ] && ! pgrep -f flash_gguf_stage.sh > /dev/null; then say "STOP: staging ended without the OK mark: $(tail -1 /home/cereal/cleanup-20261018/flash_gguf_stage.log)"; exit 2; fi
  grep -q -E "REFUSED|Traceback|OutOfMemory" "$SEG" 2> /dev/null && { say "STOP: run 1 segment failed: $(grep -m1 -E 'REFUSED|Traceback|OutOfMemory' "$SEG" | cut -c1-200)"; exit 3; }
  steps=$(grep -c "^STEP" "$SEG" 2> /dev/null || echo 0)
  [ $ok_gguf = 1 ] && [ "$steps" -ge 2 ] && break
  say "waiting: gguf_ok=$ok_gguf run1_steps=$steps"; sleep 300
done
say "GGUF verified and run 1 training ($steps steps): starting flash_lane.sh"
cd "$T" && setsid nohup bash flash_lane.sh > $OC/distill/flash_lane.log 2>&1 < /dev/null &
sleep 60; say "lane: $(tail -2 $OC/distill/flash_lane.log | tr '\n' ' ')"
