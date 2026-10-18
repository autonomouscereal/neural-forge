#!/bin/bash
# flash_mix_queue.sh -- while flash_score_queue.sh scores the corpora, build each Q38 + Flash Next mixture cache as soon
# as its Flash Next cache seals (fullkd_data_flash.py mix-ready, CPU only, idempotent).  Ends when the Flash Next score
# queue has finished and every sealed Flash Next cache has its mixture.
#   setsid nohup bash flash_mix_queue.sh > /home/cereal/omni-campaign/distill/flash_score/mix_queue.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
export CUDA_VISIBLE_DEVICES= PYTHONPATH=$T OMP_NUM_THREADS=4
utc() { date -u +%FT%TZ; }
while :; do
  nice -n 15 "$PY" "$T/fullkd_data_flash.py" mix-ready 2>&1 | tail -3
  if ! pgrep -f flash_score_queue.sh > /dev/null && ! pgrep -f flash_lane.sh > /dev/null; then
    nice -n 15 "$PY" "$T/fullkd_data_flash.py" mix-ready 2>&1 | tail -2
    echo "$(utc) Flash Next lane finished; mixtures done"; break
  fi
  sleep 900
done
