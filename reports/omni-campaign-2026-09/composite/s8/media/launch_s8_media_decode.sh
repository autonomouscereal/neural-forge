#!/bin/sh
# S8-MEDIA decode (parent VAEs inside the ComfyUI runtime image), S8-MEDIA lane, Claude, 2026-10-13.
#   Usage: GPU=0 MODS=ace,h3,qimg,hy3d sh launch_s8_media_decode.sh [extra s8_media_decode.py args]
#   One container, exactly one GPU visible (by UUID; the r5 image bakes CUDA_VISIBLE_DEVICES="" so it is re-set to 0),
#   nice 15, 8 CPUs.  Refuses to start on a card above 2000 MiB.  Allocator record released with the exit code.
set -u
OC=/home/cereal/omni-campaign
GPU=${GPU:-0}
MODS=${MODS:-ace,h3,qimg,hy3d}
NAME=s8-media-decode-$(echo $MODS | tr ',' '-')
A=$OC/composite/receipts/allocator/$NAME-gpu$GPU-claude.json
IMG=comfyui:codex-minimax-h3-runtime-env-r5-20260808
if [ "$GPU" = "5" ] || [ "$GPU" = "6" ]; then echo "REFUSED: GPU $GPU is not this lane's"; exit 3; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ')
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)
echo "start $(date -u +%FT%TZ) gpu $GPU ($name, $uuid) used ${used} MiB image $IMG"
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
 "job": "docker $IMG s8_media_decode.py --modalities $MODS (parent VAE decodes of OMNI1 emissions)",
 "cpu_threads": 8,
 "nice": 15,
 "status": "running"
}
EOF
sha256sum $OC/composite/tools/s8_media_decode.py $OC/media/mediafold/vae_adapters.py $0
docker run --rm --name $NAME --gpus "device=$uuid" -e NVIDIA_VISIBLE_DEVICES=$uuid -e CUDA_VISIBLE_DEVICES=0 \
  --cpus 8 --shm-size 8g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e MPLCONFIGDIR=/tmp/.mpl -e OC=$OC -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $OC:$OC:rw -v /home/cereal/comfyui/models:/home/cereal/comfyui/models:ro \
  -v /mnt/lexar/omni-campaign:/mnt/lexar/omni-campaign:ro \
  --workdir /tmp --entrypoint /usr/bin/nice $IMG -n 15 /opt/environments/python/comfyui/bin/python -u \
  $OC/composite/tools/s8_media_decode.py --modalities $MODS "$@"
rc=$?
END=$(date -u +%FT%TZ)
python3 - "$A" "$END" "$rc" <<'EOF'
import json, sys
p, end, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(p)); d.update({"status": "released", "ended_utc": end, "exit": rc})
open(p, "w").write(json.dumps(d, indent=1) + "\n")
EOF
echo "s8_media_decode_exit=$rc $END"
