set -u
OC=/home/cereal/omni-campaign
PB=$OC/composite/s7/prep_b
G=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582   # GPU4 RTX 3090
EF=$OC/import/rq27b/tree_ef1ffb9
used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$G" '$1==g{print $2}')
if [ "${used:-99999}" -ge 2000 ]; then echo "GPU4 at ${used} MiB -- refusing to start"; exit 2; fi
START=$(date -u +%FT%TZ)
cat > $OC/composite/receipts/allocator/prepb-hy3d-timerow-hostrows-gpu4-claude.json <<J
{"name": "prepb-hy3d-timerow-hostrows-gpu4-claude", "owner": "claude (PREP-B workflow lane)", "gpu_index": 4,
 "gpu_uuid": "$G", "gpu_name": "NVIDIA GeForce RTX 3090", "memory_used_mib_at_start": ${used},
 "started_utc": "$START", "job": "s7_host_rows.py --unit hy3d_dit_L10 (S3 trunk prefix 0..31, bf16; ~21.5 GiB peak per M1)",
 "log": "$PB/logs/host_rows_hy3d.log", "cpu_threads": 2, "nice": 15, "status": "running"}
J
cd $OC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True \
nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python composite/tools/s7_host_rows.py --unit hy3d_dit_L10 --device cuda:0 \
  > $PB/logs/host_rows_hy3d.log 2>&1 < /dev/null
CODE=$?
END=$(date -u +%FT%TZ)
python3 - <<P
import json
p="$OC/composite/receipts/allocator/prepb-hy3d-timerow-hostrows-gpu4-claude.json"
r=json.load(open(p)); r.update({"ended_utc": "$END", "exit": $CODE, "status": "released"}); json.dump(r, open(p,"w"), indent=1)
P
echo "host_rows_exit=$CODE end=$END" >> $PB/logs/host_rows_hy3d.log
