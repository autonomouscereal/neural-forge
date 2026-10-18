#!/bin/sh
# S5 finish (2026-10-12, Claude): everything after the 64-layer pass, in order, each step resumable.
# The 64 layers are persisted under $S/child.partial (verified as a virtual child: vcheck_full_partial.json).
# Root must have >= ~51 GiB for the child PLUS the 60 GB floor (the install stops itself otherwise, exit 3).
#   STEP=write      sh finish_s5.sh   detached: hardlink snapshot of the partial (no extra space), then
#                                     launch_s5.sh --resume --min-free-gib 56 on GPU$GPU (re-hash ~25 min, reload, write)
#   STEP=counts     sh finish_s5.sh   s5_register_counts.py (headers of the HF parent, the child, the S3 trunk)
#   STEP=written    sh finish_s5.sh   detached: vcheck_s5.sh MODE=written on GPU$GPU (written child as --reference)
#   STEP=metrics    sh finish_s5.sh   s5_metrics.py for both checks
#   STEP=register   sh finish_s5.sh [--confirm]   s5_register.sh dry run, then --confirm
#   STEP=cleanup    sh finish_s5.sh   removes ONLY this lane's scratch: $S/partial_snapshot (after the written check)
OC=/home/cereal/omni-campaign
S=$OC/composite/c10/S5-C2K4352K3U
TRUNK=/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native
cd $S
case "${STEP:?set STEP}" in
  write)
    df -B1G / | tail -1
    mkdir -p partial_snapshot
    for f in child.partial/*; do ln -f "$f" "partial_snapshot/$(basename "$f")"; done
    ls partial_snapshot | wc -l
    GPU=${GPU:?set GPU} setsid nohup sh launch_s5.sh --resume --min-free-gib 56 > logs/s5_write.log 2>&1 < /dev/null &
    echo "write launched" ;;
  counts)
    python3 s5_register_counts.py --parent /home/cereal/models/Qwen3.8-27B-Uncensored --child $S/child --trunk $TRUNK \
      --out $S/register_counts.json ;;
  written)
    GPU=${GPU:?set GPU} MODE=written setsid nohup sh vcheck_s5.sh > logs/vcheck_written.log 2>&1 < /dev/null &
    echo "written check launched" ;;
  metrics)
    python3 s5_metrics.py --vcheck vcheck_full_partial.json --layers-dir partial_snapshot --out s5_metrics_full.json
    python3 s5_metrics.py --vcheck vcheck_written_child.json --union-receipt child/union_receipt.json --out s5_metrics_written.json ;;
  register)
    bash s5_register.sh "$@" ;;
  cleanup)
    [ -f vcheck_written_child.json ] && rm -r partial_snapshot && echo "partial_snapshot removed" ;;
esac
