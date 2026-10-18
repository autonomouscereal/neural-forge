#!/bin/bash
# media_m1_switch.sh NEXT_TRAINER DATA_CONFIG -- move the running media M1 job onto another media_full_kd_trainer build and
# data config at its NEXT checkpoint, losing no checkpointed work:
#   wait for a checkpoint newer than the current newest -> touch STOP (>= 1.2.0 checkpoints again on STOP) -> wait for
#   the trainer and its queue to exit -> keep the running trainer as media_full_kd_trainer.<version>.py -> install NEXT ->
#   restart the queue with --resume (new trainables keep their base values) -> rm STOP.
#   setsid nohup bash media_m1_switch.sh $T/media_full_kd_trainer.d3d.py $M/media_data_v4.json > $M/switch_v4.log 2>&1 &
set -u
NEXT=${1:?next trainer file}; DATA=${2:?data config}
T=/home/cereal/omni-campaign/distill/tools
M=/mnt/lexar/omni-campaign/distill/media_m1
RUN=${RUN:-$M/run1}
utc() { date -u +%FT%TZ; }
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$NEXT" || { echo "$(utc) STOP: $NEXT does not parse"; exit 2; }
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$DATA" || { echo "$(utc) STOP: $DATA is not JSON"; exit 2; }
newest() { ls -d $RUN/out/checkpoints/step-* 2> /dev/null | tail -1; }
start=$(newest)
echo "$(utc) waiting for a checkpoint newer than ${start:-none}"
while [ "$(newest)" = "$start" ]; do sleep 30; done
echo "$(utc) checkpoint $(newest); STOP"
touch $RUN/out/STOP
while pgrep -f "media_full_kd_trainer.py train" > /dev/null; do sleep 15; done
while pgrep -f media_m1_queue.sh > /dev/null; do sleep 5; done
echo "$(utc) trainer exited: $(tail -1 $RUN/queue.status)"
old=$(grep -m1 -o 'TOOL_VERSION = "[0-9.]*"' $T/media_full_kd_trainer.py | grep -o '[0-9.]*[0-9]')
cp -p $T/media_full_kd_trainer.py $T/media_full_kd_trainer.$old.py
cp -p "$NEXT" $T/media_full_kd_trainer.py
rm -f $RUN/out/STOP
cd $T
MODE=train STEPS=${STEPS:-300} LR=${LR:-2e-5} GROUP_TOKENS=${GROUP_TOKENS:-30000} DATA=$DATA EXTRA="--head-device cuda:0" RUN=$RUN \
  setsid nohup bash media_m1_queue.sh >> $M/run1-train-queue.log 2>&1 < /dev/null &
sleep 5
echo "$(utc) restarted ($old -> $(grep -m1 -o 'TOOL_VERSION = "[0-9.]*"' $T/media_full_kd_trainer.py)) on $DATA: $(tail -1 $RUN/queue.status)"
