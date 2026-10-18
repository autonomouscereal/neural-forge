#!/bin/bash
# lexar_relief.sh -- keep the K3 queue's Lexar floor (>= 200 GB free before every pass; queue_gpu6 WAITS, never kills)
# while the full-KD trainer holds a rolling 47 GB checkpoint there.  After flash_stage_chain.sh (the only other NAS
# writer) ends, move the D1 transient scratch (scratch/d1_boundary 77 GB, scratch/d1seal 52 GB; D1 sealed 2026-10-15,
# nothing reads them since; p1_plan.json reads only clones/) to the NAS with verified_move.sh -- verified byte-equal,
# symlinked back, nothing destroyed.
#   setsid nohup bash lexar_relief.sh > /home/cereal/cleanup-20261017/lexar_relief.log 2>&1 < /dev/null &
set -u
T=/home/cereal/omni-campaign/distill/tools
NAS=/mnt/ai_data/backups/lexar-relief-20261018/omni-campaign/scratch
utc() { date -u +%FT%TZ; }
while pgrep -f flash_stage_chain.sh > /dev/null; do sleep 300; done
echo "$(utc) flash_stage_chain.sh ended: $(tail -1 /home/cereal/cleanup-20261017/flash_stage_chain.log)"
echo "$(utc) Lexar free $(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9) GB"
for d in d1_boundary d1seal; do
  bash "$T/verified_move.sh" "/mnt/lexar/omni-campaign/scratch/$d" "$NAS/$d"
  rc=$?
  echo "$(utc) $d: exit $rc; Lexar free $(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9) GB"
  [ $rc -eq 0 ] || exit $rc
done
