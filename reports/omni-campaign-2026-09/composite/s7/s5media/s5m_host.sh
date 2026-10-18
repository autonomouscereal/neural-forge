#!/bin/sh
# S7-on-S5 media lane (Claude, 2026-10-13): trunk host rows with s7_host_rows 1.3.0 --stream-blocks, one unit at a
# time on ONE card (refuses to start a unit while the card holds >= 2000 MiB), an allocator record per unit.
#   usage: s5m_host.sh <gpu index> <mode: s5 | s3equiv> <unit> [<unit> ...]
#   s5      : model = the written S5 child, out = composite/s7/tap_rows_s5, units root = composite/s7/units_s5
#   s3equiv : model = the S3 native trunk, out = composite/s7/s5media/scratch_equiv (lane scratch): the streamed path on
#             the trunk the S3 rows were captured on, compared with composite/s7/tap_rows_s3 (resident path)
set -u
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
L=$S7/s5media/logs
EF=$OC/import/rq27b/tree_ef1ffb9
S3N=/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native
TOK=$(dirname $S3N)/tokenizer/tokenizer.json
S5=$OC/composite/c10/S5-C2K4352K3U/child
PY=/home/cereal/model-surgery-lab/venv/bin/python
IDX=$1; MODE=$2; shift 2
G=$(nvidia-smi -i $IDX --query-gpu=uuid --format=csv,noheader)
NAME=$(nvidia-smi -i $IDX --query-gpu=name --format=csv,noheader)
case $MODE in
  s5) MODEL=$S5; OUT=$S7/tap_rows_s5; UR=$S7/units_s5 ;;
  s3equiv) MODEL=$S3N; OUT=$S7/s5media/scratch_equiv; UR=$S7/units ;;
  *) echo "mode?"; exit 2 ;;
esac
mkdir -p $OUT $L $OC/composite/receipts/allocator
cd $OC
for u in "$@"; do
  used=$(nvidia-smi -i $IDX --query-gpu=memory.used --format=csv,noheader,nounits)
  while [ "${used:-99999}" -ge 2000 ]; do echo "[$u] GPU$IDX at $used MiB, waiting"; sleep 30; used=$(nvidia-smi -i $IDX --query-gpu=memory.used --format=csv,noheader,nounits); done
  START=$(date -u +%FT%TZ)
  A=$OC/composite/receipts/allocator/s5media-hostrows-$MODE-$u-gpu$IDX-claude.json
  cat > $A <<J
{"name": "s5media-hostrows-$MODE-$u-gpu$IDX-claude", "owner": "claude (S7-on-S5 media lane, workflow subagent)", "gpu_index": $IDX,
 "gpu_uuid": "$G", "gpu_name": "$NAME", "memory_used_mib_at_start": $used, "started_utc": "$START",
 "job": "s7_host_rows.py 1.3.0 --stream-blocks --unit $u --model $MODEL (prefix 0..31, bf16)",
 "log": "$L/host_${MODE}_$u.log", "cpu_threads": 2, "nice": 15, "status": "running"}
J
  echo "[$u] $START start on GPU$IDX ($used MiB)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True \
  nice -n 15 $PY composite/tools/s7_host_rows.py --unit $u --model $MODEL --tokenizer $TOK \
    --interface $S7/interface_latent --interface-c11 $S7/interface_latent/c11 --pairs $S7/pairs --out $OUT \
    --units-root $UR --chimera-src $EF --device cuda:0 --dtype bfloat16 --stream-blocks \
    > $L/host_${MODE}_$u.log 2>&1 < /dev/null
  CODE=$?
  END=$(date -u +%FT%TZ)
  python3 - "$A" "$END" "$CODE" <<'P'
import json, sys
p, end, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = json.load(open(p)); r.update({"ended_utc": end, "exit": code, "status": "released"}); json.dump(r, open(p, "w"), indent=1)
P
  echo "[$u] $END exit $CODE"
  tail -2 $L/host_${MODE}_$u.log
done
echo "[drive GPU$IDX $MODE] done $(date -u +%FT%TZ)"
