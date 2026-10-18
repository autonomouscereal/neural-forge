#!/bin/sh
# OMNI1 load / identity / forward check (S7 step 2), OMNI1 assembly lane, Claude, 2026-10-13.
#   s7_omni1_check.py 1.0.0 on GPU ${GPU:-6} (nvidia-smi index; refuses to start above 2000 MiB), tree ef1ffb9, bf16:
#   strict structural load in pieces (every tensor sha-verified on read), trunk identity vs the S5 child, accounting,
#   the 8 R-Q prompts through OMNI1 / S5 / the 27B parent S0 import, one held native media segment per charted block.
#   Records the GPU in $OC/composite/receipts/allocator/omni1-check-gpu$GPU-claude.json.
set -u
OC=/home/cereal/omni-campaign
EF=$OC/import/rq27b/tree_ef1ffb9
GPU=${GPU:-6}
S=$OC/composite/s7/omni1
OUT=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
A=$OC/composite/receipts/allocator/omni1-check-gpu$GPU-claude.json
PY=/home/cereal/model-surgery-lab/venv/bin/python
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ')
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)
echo "start $(date -u +%FT%TZ) gpu $GPU ($name) used ${used} MiB"
if [ "$used" -gt 2000 ]; then echo "REFUSED: GPU $GPU uses ${used} MiB > 2000"; exit 3; fi
START=$(date -u +%FT%TZ)
cat > $A <<EOF
{
 "name": "omni1-check-gpu$GPU-claude",
 "owner": "claude (OMNI1 assembly lane, workflow subagent)",
 "gpu_index": $GPU,
 "gpu_uuid": "$uuid",
 "gpu_name": "$name",
 "memory_used_mib_at_start": $used,
 "started_utc": "$START",
 "job": "s7_omni1_check.py 1.0.0 --omni1 $OUT (layer-major, one block on the GPU at a time, bf16)",
 "log": "$S/logs/omni1_check.log",
 "cpu_threads": 8,
 "nice": 15,
 "status": "running"
}
EOF
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC_LEDGER_ROOT=$OC/ledger
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
sha256sum $OC/composite/tools/s7_omni1_check.py $0
cd $OC/composite/tools
nice -n 15 /usr/bin/time -v $PY s7_omni1_check.py --omni1 $OUT --s5 $OC/composite/c10/S5-C2K4352K3U/child \
  --parent /mnt/ai_data/omni-campaign/s0-probe-sep04 --tap-rows $OC/composite/s7/tap_rows_s5 \
  --tokenizer-dir /mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07 --chimera-src $EF \
  --parent-parameter-count 27781427952 --device cuda:0 --dtype bfloat16 --out $S/omni1_check.json
rc=$?
END=$(date -u +%FT%TZ)
python3 - "$A" "$END" "$rc" <<'EOF'
import json, sys
p, end, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(p)); d.update({"status": "released", "ended_utc": end, "exit": rc})
open(p, "w").write(json.dumps(d, indent=1) + "\n")
EOF
echo "check_exit=$rc $END"
