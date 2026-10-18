#!/bin/bash
# media_m1_switch_v3.sh -- move the running media M1 job onto media_full_kd_trainer 1.2.0 + media_data_v3.json (adds the
# D3c music_codes route) at its first checkpoint, with no lost checkpointed work:
#   wait for a checkpoint -> touch STOP -> wait for the trainer to exit (1.1.0 stops at the next group) -> swap the trainer
#   (1.1.0 kept as media_full_kd_trainer.1.1.0.py) -> restart the queue with --resume (1.2.0 keeps the checkpoint's
#   tensors and starts the music band rows from the base child) -> rm STOP.
set -u
T=/home/cereal/omni-campaign/distill/tools
M=/mnt/lexar/omni-campaign/distill/media_m1
RUN=$M/run1
utc() { date -u +%FT%TZ; }
echo "$(utc) waiting for the first M1 checkpoint"
until ls -d $RUN/out/checkpoints/step-* > /dev/null 2>&1; do sleep 30; done
echo "$(utc) checkpoint $(ls -d $RUN/out/checkpoints/step-* | tail -1); STOP"
touch $RUN/out/STOP
while pgrep -f "media_full_kd_trainer.py train" > /dev/null; do sleep 15; done
while pgrep -f media_m1_queue.sh > /dev/null; do sleep 5; done
echo "$(utc) trainer exited: $(tail -1 $RUN/queue.status)"
cp -p $T/media_full_kd_trainer.py $T/media_full_kd_trainer.1.1.0.py
cp -p $T/media_full_kd_trainer.next.py $T/media_full_kd_trainer.py
rm -f $RUN/out/STOP
cd $T
MODE=train STEPS=300 LR=2e-5 GROUP_TOKENS=30000 DATA=$M/media_data_v3.json EXTRA="--head-device cuda:0" RUN=$RUN \
  setsid nohup bash media_m1_queue.sh >> $M/run1-train-queue.log 2>&1 < /dev/null &
sleep 5
echo "$(utc) restarted on 1.2.0 + v3: $(tail -1 $RUN/queue.status)"
