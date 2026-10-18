#!/bin/bash
# fullkd_queue.sh -- the full-parameter Chimera Omni KD run (streamed_full_kd_trainer.py) as one unattended queue.
# Rulings: cyril-full-distill-no-lora-2026-10-17 (every weight trained, init from the folds, K3 primary, Q38 small weight),
# cyril-drop-selfgen-refill-2026-10-17.  Plan: reports/omni-campaign-2026-09/REALIGNMENT_2026-10-17.md.
#   setsid nohup bash fullkd_queue.sh > /mnt/lexar/omni-campaign/distill/fullkd/run1/queue.log 2>&1 < /dev/null &
# Loop: take $Q/gpu$GPU.lock (flock) -> wait MemAvailable >= NEED_GIB and Lexar >= LEX_MIN_GB -> train --resume auto.
# ONE rolling 47 GB checkpoint (a second would hold Lexar under the 200 GB the K3 queue needs before each pass).  Exit 0 (completed: student exported) -> done.  Exit 4 (a deferred K3 pass is not
# READY at its token threshold; the trainer checkpointed) -> wait for that marker, then resume.  Exit 3 (halted) or 2
# (refused) -> stop for a ruling.  Any other exit (crash) -> retry from the newest checkpoint, at most RETRIES times.
# Stop for good: touch $RUN/STOP (checked between segments); kill -TERM the trainer pid to interrupt a segment (it resumes
# from its last checkpoint, up to SAFETY_MINUTES of work lost).
set -u
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=${GPU:-4}
RUN=${RUN:-/mnt/lexar/omni-campaign/distill/fullkd/run1}
DATA=${DATA:-/mnt/lexar/omni-campaign/distill/fullkd/fullkd_data_v3.json}
NEED_GIB=${NEED_GIB:-85}
LEX_MIN_GB=${LEX_MIN_GB:-110}
RETRIES=${RETRIES:-3}
TOKENS=${TOKENS:-20000000}
MICRO=${MICRO:-8}
LR=${LR:-1e-5}
# more trainer args (2026-10-18: EXTRA="--pin-weights --allow-config-change" -- the page-locked host master; py-spy put
# 73 % of a run-1 step in pageable host<->device weight copies.  --allow-config-change because --pin-weights is part of
# the config digest; base and data identity stay guarded)
EXTRA=${EXTRA:-}
STATUS=$RUN/queue.status
mkdir -p "$RUN"
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*" | tee -a "$STATUS"; }
avail_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
lex_gb() { df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9; }
ARGS=(train --p1-chimera /mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1
  --p1-parent-config /home/cereal/models/Qwen3.8-27B-Uncensored --p1-plan /mnt/lexar/omni-campaign/distill/d2/p1_plan.json
  --device cuda:0 --data-config "$DATA" --seq-len 4096 --max-width 5282 --micro-batch "$MICRO" --token-budget "$TOKENS"
  --count-real-tokens --lr "$LR" --warmup-steps 20 --min-lr-ratio 0.1
  --out "$RUN/out" --safety-dir "$RUN/safety" --safety-minutes 60 --safety-keep 1 --resume auto
  --no-preregistration "full-parameter KD of Chimera Omni (ruling cyril-full-distill-no-lora-2026-10-17; plan REALIGNMENT_2026-10-17.md)")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)
exec 9>> "$Q/gpu$GPU.lock"
say "queue start: GPU$GPU, data $DATA ($(sha256sum "$DATA" | cut -c1-16)), tokens $TOKENS, micro $MICRO, lr $LR, extra [$EXTRA], trainer $(sha256sum $T/streamed_full_kd_trainer.py | cut -c1-16)"
flock 9
say "gpu$GPU.lock held"
crashes=0
while :; do
  [ -e "$RUN/STOP" ] && { say "STOP file present: exit"; exit 0; }
  while [ "$(avail_gib)" -lt "$NEED_GIB" ] || [ "$(lex_gb)" -lt "$LEX_MIN_GB" ]; do
    say "WAIT MemAvailable $(avail_gib) GiB (need $NEED_GIB), Lexar $(lex_gb) GB (need $LEX_MIN_GB)"; sleep 300
  done
  seg=$(ls "$RUN"/train-*.log 2>/dev/null | wc -l)
  log=$RUN/train-$(printf %03d "$seg").log
  say "segment $seg start, log $log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nice -n 5 "$PY" "$T/streamed_full_kd_trainer.py" "${ARGS[@]}" --receipt "$RUN/receipt-$(printf %03d "$seg").json" > "$log" 2>&1 9>&-
  rc=$?
  say "segment $seg exit $rc: $(grep -E '^TRAIN|REFUSED' "$log" | tail -1 | cut -c1-300)"
  case $rc in
    0) say "DONE: student in $RUN/out/student"; exit 0 ;;
    4) marker=$(grep -o 'sealed marker [^ ]*' "$log" | tail -1 | awk '{print $3}')
       say "waiting for K3 marker ${marker:-?}"
       while [ -n "$marker" ] && [ ! -e "$marker" ]; do sleep 300; done ;;
    2|3) say "STOPPED for a ruling (exit $rc); resume: bash $0 after the fix"; exit "$rc" ;;
    *) crashes=$((crashes + 1)); [ $crashes -gt "$RETRIES" ] && { say "crashed $crashes times: stop"; exit 1; }
       say "crash $crashes of $RETRIES: retrying from the newest checkpoint in 120 s"; sleep 120 ;;
  esac
done
