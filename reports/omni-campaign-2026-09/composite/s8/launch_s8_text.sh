#!/bin/bash
# S8-TEXT capability review (2026-10-13, Claude, S8 lane): the frozen bank v1 on the readable (text) surface of
#   omni1 = OC02-F-Q38-27BU-OMNI1 (the S7 composite, 69 blocks, 5 charted)   split 6,33,30 (GPU4 holds blocks 6-38 incl. charted 32-36)
#   s5    = OC02-F-Q38-27BU-C2K4352K3U (the S5 union child, 64 blocks)        split 6,28,30 (same trunk placement as omni1)
# v2 (2026-10-13 ~01:10 MDT): GPU6 sits in SW Thermal Slowdown (SM 300 MHz at 82 C, nvidia-smi -q); the v1 split 45,24 on
#   GPU6+GPU4 ran 1,689-2,089 s per 1,024-token case, so v2 keeps only the head + 6 blocks on GPU6 and spreads the trunk
#   over the two RTX 3090s (GPU4, GPU0).  v1 was stopped after 4/16 cases; its log/guard receipt are kept under s8/scratch.
# Same runner and flags as the parent (E2) and S3 dossiers: run_omni_frozen_review.py (1.2.0: charted blocks accepted on
# the free-run TEXT path, identity on text, tree rule) under run_review_finite_guard.py (1.2.0, read-only hooks,
# --host-staged-moves), tree ef1ffb9, bank v1, thinking off, --ledger-band-logit-mask (REQUIRED for OMNI1: the C18 rows
# in lm_head win the argmax at 146/210 R-Q positions unmasked), probe + smoke packs, cap 1024, greedy, bf16,
# CUDA_VISIBLE_DEVICES=6,4,0 (cuda:0 = Quadro RTX 8000 = head device, cuda:1/cuda:2 = RTX 3090).  Tokenizer = the S3 dossier's.
# Refuses to start if GPU6 or GPU4 is above 2000 MiB; records the GPUs in $OC/composite/receipts/allocator/.
# usage: launch_s8_text.sh omni1|s5      (run detached: setsid nohup ... &)
set -u
OC=/home/cereal/omni-campaign
EF=$OC/import/rq27b/tree_ef1ffb9
PY=/home/cereal/model-surgery-lab/venv/bin/python
WHICH=${1:?omni1|s5}
case "$WHICH" in
  omni1) MODEL=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1; LABEL=OC02-OMNI1-NATIVE-NOTHINK; SPLIT=6,33,30 ;;
  s5)    MODEL=$OC/composite/c10/S5-C2K4352K3U/child;            LABEL=OC02-C2K4352K3U-NATIVE-NOTHINK; SPLIT=6,28,30 ;;
  *) echo "usage: $0 omni1|s5"; exit 2 ;;
esac
GPUS="6 4 0"
S=$OC/composite/s8
OUT=$S/review/oc-dossier-$LABEL.json
LOG=$S/logs/review_$WHICH.log
A=$OC/composite/receipts/allocator/s8-text-$WHICH-gpu6-4-0-claude.json
mkdir -p $S/review $S/logs
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists"; exit 3; }
for g in $GPUS; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
  echo "GPU$g $(nvidia-smi --query-gpu=name --format=csv,noheader -i $g) used ${u} MiB"
  if [ "$u" -gt 2000 ]; then echo "REFUSED: GPU$g uses ${u} MiB > 2000"; exit 4; fi
done
START=$(date -u +%FT%TZ)
cat > $A <<EOF
{
 "name": "s8-text-$WHICH-gpu6-4-0-claude",
 "owner": "claude (S8-TEXT review lane, workflow subagent)",
 "gpu_indices": [6, 4, 0],
 "gpu_uuids": ["$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 6 | tr -d ' ')", "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 4 | tr -d ' ')", "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | tr -d ' ')"],
 "gpu_names": ["$(nvidia-smi --query-gpu=name --format=csv,noheader -i 6)", "$(nvidia-smi --query-gpu=name --format=csv,noheader -i 4)", "$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"],
 "started_utc": "$START",
 "job": "run_review_finite_guard.py -> run_omni_frozen_review.py 1.2.0 free-run bank v1 on $MODEL, split $SPLIT, bf16",
 "log": "$LOG",
 "cpu_threads": 8,
 "nice": 15,
 "status": "running"
}
EOF
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC_CALIBRATION_DIR=$OC/calibration
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,4,0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd $OC
sha256sum $OC/composite/tools/run_omni_frozen_review.py $OC/composite/tools/run_review_finite_guard.py $0
ARGS=(--host-staged-moves --model "$MODEL" --tokenizer-dir /mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07
  --chimera-src $EF --label $LABEL --parent-code Q38-27BU --stage fold --split $SPLIT --dtype bfloat16 --device cuda:0
  --thinking off --ledger-band-logit-mask --maximum-new-tokens 1024
  --bank calibration/banks/lfm2_semantic_whole_response_prompts_v1.json
  --probe-prompts calibration/banks/matrix_campaign_probe_prompts_v1.json
  --smoke-prompts calibration/banks/matrix_campaign_smoke_prompts_v1.json
  --out "$OUT")
echo "argv: $PY $OC/composite/tools/run_review_finite_guard.py ${ARGS[*]}"
nice -n 15 /usr/bin/time -v $PY $OC/composite/tools/run_review_finite_guard.py "${ARGS[@]}"
rc=$?
END=$(date -u +%FT%TZ)
python3 - "$A" "$END" "$rc" <<'EOF'
import json, sys
p, end, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(p)); d.update({"status": "released", "ended_utc": end, "exit": rc})
open(p, "w").write(json.dumps(d, indent=1) + "\n")
EOF
echo "review_exit=$rc $END"
