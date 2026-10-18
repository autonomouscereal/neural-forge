#!/bin/sh
# S8-MEDIA host pass (OMNI1 forward per modality + QVL answers), S8-MEDIA lane, Claude, 2026-10-13.
#   s8_media_smoke.py on GPU ${GPU:-0} (nvidia-smi index; refuses to start above 2000 MiB), tree ef1ffb9, bf16.
#   Usage: GPU=0 NAME=s8-media-forward sh launch_s8_media_host.sh [s8_media_smoke.py args...]
#   Records the GPU in $OC/composite/receipts/allocator/$NAME-gpu$GPU-claude.json (released with the exit code).
set -u
OC=/home/cereal/omni-campaign
EF=$OC/import/rq27b/tree_ef1ffb9
GPU=${GPU:-0}
NAME=${NAME:-s8-media-host}
A=$OC/composite/receipts/allocator/$NAME-gpu$GPU-claude.json
PY=/home/cereal/model-surgery-lab/venv/bin/python
if [ "$GPU" = "5" ] || [ "$GPU" = "6" ]; then echo "REFUSED: GPU $GPU is not this lane's"; exit 3; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ')
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)
echo "start $(date -u +%FT%TZ) gpu $GPU ($name) used ${used} MiB"
if [ "$used" -gt 2000 ]; then echo "REFUSED: GPU $GPU uses ${used} MiB > 2000"; exit 3; fi
START=$(date -u +%FT%TZ)
cat > $A <<EOF
{
 "name": "$NAME-gpu$GPU-claude",
 "owner": "claude (S8-MEDIA lane, workflow subagent)",
 "gpu_index": $GPU,
 "gpu_uuid": "$uuid",
 "gpu_name": "$name",
 "memory_used_mib_at_start": $used,
 "started_utc": "$START",
 "job": "s8_media_smoke.py $* (OMNI1 trunk blocks in host RAM streamed to the GPU per pass, charted blocks resident, bf16)",
 "cpu_threads": 8,
 "nice": 15,
 "status": "running"
}
EOF
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC_LEDGER_ROOT=$OC/ledger MEDIAFOLD_ROOT=$OC/media OC=$OC
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
sha256sum $OC/composite/tools/s8_media_smoke.py $OC/composite/tools/s7_omni1_check.py $OC/composite/tools/s7_pairs_parent.py $0
cd $OC/composite/tools
nice -n 15 /usr/bin/time -v $PY s8_media_smoke.py --device cuda:0 --chimera-src $EF "$@"
rc=$?
END=$(date -u +%FT%TZ)
python3 - "$A" "$END" "$rc" <<'EOF'
import json, sys
p, end, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(p)); d.update({"status": "released", "ended_utc": end, "exit": rc})
open(p, "w").write(json.dumps(d, indent=1) + "\n")
EOF
echo "s8_media_exit=$rc $END"
