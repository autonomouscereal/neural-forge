#!/bin/bash
# media_m1_queue.sh -- Chimera Omni media full-KD stage M1 on GPU0 (media_full_kd_trainer.py): every weight of the five
# charted units + the native media flow head, trunk frozen at OMNI1D1 (run 1 owns the trunk on GPU4).
#   MODE=probe setsid nohup bash media_m1_queue.sh > $RUN/queue.log 2>&1 < /dev/null &     (lr 0, 3 groups)
#   MODE=train setsid nohup bash media_m1_queue.sh > $RUN/queue.log 2>&1 < /dev/null &     (resumable)
# Holds $Q/gpu0.lock (flock) and writes CLAIM/RELEASE lines in $Q/p40.claims.  Waits for MemAvailable >= NEED_GIB.
# Stop: touch $RUN/STOP (checkpoint + exit at the next group), or kill -TERM the trainer (resume from the last checkpoint).
set -u
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
MODE=${MODE:-probe}
RUN=${RUN:-/mnt/lexar/omni-campaign/distill/media_m1/run1}
DATA=${DATA:-/mnt/lexar/omni-campaign/distill/media_m1/media_data_v1.json}
NEED_GIB=${NEED_GIB:-24}
STEPS=${STEPS:-3}
LR=${LR:-1e-5}
GROUP_TOKENS=${GROUP_TOKENS:-30000}
EXTRA=${EXTRA:-}   # more trainer args, e.g. "--head-device cuda:0" when speech_in is a route
mkdir -p "$RUN"
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*" | tee -a "$RUN/queue.status"; }
avail_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
# OMP threads: the P1 reader composes the frozen D1 overlay on the CPU every layer load (py-spy 10-18: 65 % of a group);
# 8 threads left ~80 free cores idle, so the default is now 32 (OMP_THREADS overrides)
export CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src:$T OMP_NUM_THREADS=${OMP_THREADS:-32}
exec 9>> "$Q/gpu0.lock"
flock -w 600 9 || { say "STOP: gpu0.lock is held"; exit 5; }
echo "$(utc) CLAIM GPU0 owner=claude lane=media-m1 job=media-$MODE pid=$$ log=$RUN preempt: touch $RUN/STOP" >> "$Q/p40.claims"
trap 'echo "$(utc) RELEASE GPU0 owner=claude lane=media-m1 job=media-$MODE released" >> "$Q/p40.claims"' EXIT
while [ "$(avail_gib)" -lt "$NEED_GIB" ]; do say "WAIT MemAvailable $(avail_gib) GiB < $NEED_GIB"; sleep 300; done
say "start $MODE: data $DATA ($(sha256sum "$DATA" | cut -c1-16)), trainer $(sha256sum $T/media_full_kd_trainer.py | cut -c1-16), steps $STEPS, lr $LR"
n=$(ls "$RUN"/$MODE-*.log 2>/dev/null | wc -l)
extra=()
[ "$MODE" = train ] && extra=(--resume --lr "$LR")
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 \
  "$PY" "$T/media_full_kd_trainer.py" "$MODE" \
  --omni1 /mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1 \
  --parent-config /home/cereal/models/Qwen3.8-27B-Uncensored --plan /mnt/lexar/omni-campaign/distill/d2/p1_plan.json \
  --data-config "$DATA" --device cuda:0 --dtype bfloat16 --kernels auto --group-tokens "$GROUP_TOKENS" --steps "$STEPS" \
  --out "$RUN/out" --receipt "$RUN/receipt-$MODE-$(printf %03d "$n").json" "${extra[@]}" $EXTRA \
  > "$RUN/$MODE-$(printf %03d "$n").log" 2>&1 9>&-
rc=$?
say "$MODE exit $rc: $(grep -E 'MEDIA (PROBE|TRAIN)|REFUSED|Traceback|Error' "$RUN/$MODE-$(printf %03d "$n").log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
exit $rc
