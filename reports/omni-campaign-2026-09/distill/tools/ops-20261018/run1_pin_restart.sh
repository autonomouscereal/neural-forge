#!/bin/bash
# run1_pin_restart.sh -- restart full-KD run 1 on trainer 1.0.1 with --pin-weights right after its next safety checkpoint
# (2026-10-18, Claude).  Loses only the minutes between that checkpoint and the TERM.
set -u
OC=/home/cereal/omni-campaign; T=$OC/distill/tools; RUN=/mnt/lexar/omni-campaign/distill/fullkd/run1
LOG=$RUN/train-001.log; NEW=/home/cereal/cleanup-20261018/streamed_full_kd_trainer.1.0.1.py; QNEW=/home/cereal/cleanup-20261018/fullkd_queue.patched.sh
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*"; }
base=$(grep -c "checkpoint /" "$LOG")
say "waiting for checkpoint number $((base + 1)) in $LOG"
until [ "$(grep -c "checkpoint /" "$LOG")" -gt "$base" ]; do
  pgrep -f "streamed_full_kd_trainer.py train" > /dev/null || { say "STOP: trainer gone before the checkpoint"; exit 3; }
  sleep 20
done
say "new checkpoint: $(grep "checkpoint /" "$LOG" | tail -1)"
touch "$RUN/STOP"
P=$(pgrep -f "streamed_full_kd_trainer.py train" | head -1); say "TERM trainer $P"; kill -TERM "$P"
for i in $(seq 1 60); do pgrep -f fullkd_queue.sh > /dev/null || break; sleep 5; done
pgrep -f "fullkd_queue.sh|streamed_full_kd_trainer.py train" > /dev/null && { say "STOP: queue or trainer still alive"; exit 4; }
say "queue exited: $(tail -2 "$RUN/queue.status" | tr '\n' ' ')"
[ -f "$T/streamed_full_kd_trainer.1.0.0.py" ] || cp -p "$T/streamed_full_kd_trainer.py" "$T/streamed_full_kd_trainer.1.0.0.py"
[ -f "$T/fullkd_queue.1.0.sh" ] || cp -p "$T/fullkd_queue.sh" "$T/fullkd_queue.1.0.sh"
cp "$NEW" "$T/streamed_full_kd_trainer.py"; cp "$QNEW" "$T/fullkd_queue.sh"
say "installed trainer $(grep -o 'TOOL_VERSION = "[^"]*"' $T/streamed_full_kd_trainer.py) and the EXTRA-aware queue"
rm -f "$RUN/STOP"
cd "$T" && EXTRA="--pin-weights --allow-config-change" setsid nohup bash fullkd_queue.sh >> "$RUN/queue.log" 2>&1 < /dev/null &
sleep 30
say "relaunched: $(tail -2 "$RUN/queue.status" | tr '\n' ' ')"
