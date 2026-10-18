#!/bin/bash
# E2 parent review band launcher (P40 GPU1,2,3; Pascal venv torch 2.6 cu118; fp16).
# usage: launch_review.sh <mode: smoke|full>   (v2: --host-staged-moves; P40 peer copies are corrupt on this host)
set -u
OC=/home/cereal/omni-campaign
PV=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
MODE=${1:?smoke|full}
cd $OC
COMMON=(--model $OC/composite/s0/q38-27bu-chimera-bf16-sep04 --tokenizer-dir /home/cereal/models/Qwen3.8-27B-Uncensored
  --chimera-src $OC/import/rq27b/tree_ef1ffb9 --parent-code Q38-27BU --stage parent --split 16,24,24 --dtype float16
  --device cuda:0 --thinking off --ledger-band-logit-mask
  --bank calibration/banks/lfm2_semantic_whole_response_prompts_v1.json
  --probe-prompts calibration/banks/matrix_campaign_probe_prompts_v1.json
  --smoke-prompts calibration/banks/matrix_campaign_smoke_prompts_v1.json)
if [ "$MODE" = smoke ]; then
  mkdir -p $OC/composite/scratch/e2_smoke
  EXTRA=(--label PARENT-Q38-27BU-NATIVE-NOTHINK-SMOKE --max-cases 2 --maximum-new-tokens 24 --skip-smoke
    --out composite/scratch/e2_smoke/oc-dossier-PARENT-SMOKE.json)
else
  mkdir -p $OC/composite/s0/review
  EXTRA=(--label PARENT-Q38-27BU-NATIVE-NOTHINK --maximum-new-tokens 1024
    --out composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json)
fi
for i in 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i); if [ "$u" -gt 2000 ]; then echo "GPU$i busy $u MiB -- not starting"; exit 4; fi; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=8
echo "argv: $PV $OC/composite/tools/run_review_finite_guard.py --host-staged-moves ${COMMON[*]} ${EXTRA[*]}"
exec $PV $OC/composite/tools/run_review_finite_guard.py --host-staged-moves "${COMMON[@]}" "${EXTRA[@]}"
