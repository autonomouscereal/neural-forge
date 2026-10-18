#!/bin/bash
# Lane E3 (2026-10-09): trunk FFN-input taps of the S3 native child for C17.
# Stream mode only (run_omni_frozen_review.py refuses --split with --stream; evidence taps exist only in stream mode),
# so this is ONE card: GPU0 (RTX 3090). Host-venv job; allocator receipt written below.
set -u
OC=/home/cereal/omni-campaign
NAT=/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native
C17=$OC/composite/c17
T=$C17/trunk_taps/s3native-c2k4352-held-all
PY=/home/cereal/model-surgery-lab/venv/bin/python
NAME=c17-taps-s3native-oct09-claude
GPU=0
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
uuid=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader | tr -d ' ')
if [ "$used" -gt 2000 ]; then echo "REFUSE: GPU$GPU has $used MiB used"; exit 3; fi
free_g=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_g" -lt 150 ]; then echo "REFUSE: root has only ${free_g}G free (taps need ~33G, floor 80G, K3 run +100G)"; exit 4; fi
if [ -e "$T/taps_receipt.json" ]; then echo "REFUSE: $T already holds a taps receipt"; exit 5; fi
mkdir -p $C17/logs $C17/dossier $OC/composite/receipts/allocator
LOG=$C17/logs/taps_s3native.log
setsid nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU /usr/bin/time -v $PY -u $OC/composite/tools/run_omni_frozen_review.py \
  --model $NAT --tokenizer-dir /home/cereal/models/Qwen3.8-27B-Uncensored \
  --chimera-src $OC/import/rq27b/tree_ef1ffb9 \
  --label E3-TAPS-C2K4352-NATIVE --stage fold --stream --teacher-forced --maximum-new-tokens 0 \
  --dtype bfloat16 --device cuda:0 \
  --input $OC/composite/corpus/omni-evidence-27b-v1.json --held-only --micro-batch 1 \
  --tap-residuals --tap-normed --tap-max-rows 0 --tap-out $T \
  --bank $OC/calibration/banks/lfm2_semantic_whole_response_prompts_v1.json \
  --probe-prompts $OC/calibration/banks/matrix_campaign_probe_prompts_v1.json --skip-smoke \
  --out $C17/dossier/oc-dossier-E3-TAPS-C2K4352-NATIVE.json > $LOG 2>&1 < /dev/null &
PID=$!
cat > $OC/composite/receipts/allocator/$NAME.json <<JSON
{"schema": "omni-campaign.host-venv-gpu-claim.v1", "name": "$NAME", "owner": "claude", "kind": "host-venv",
 "gpu_index": $GPU, "gpu_uuid": "$uuid", "gpu_used_mib_at_start": $used, "pid": $PID,
 "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "log": "$LOG", "tap_out": "$T",
 "command": "run_omni_frozen_review.py --stream --teacher-forced --held-only --tap-residuals --tap-normed --tap-max-rows 0 on $NAT",
 "release": "process exit"}
JSON
echo "launched pid $PID on GPU$GPU ($uuid), used ${used} MiB, root free ${free_g}G"
