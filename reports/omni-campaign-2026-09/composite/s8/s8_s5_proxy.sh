#!/bin/bash
# S8-TEXT, S5 column (2026-10-13, Claude, S8 lane).  The S5 FREE-RUN review (launch_s8_text.sh s5) died at load with a
# CUDA OOM on GPU0: the Stage-D wave-2 queues (distill/queues/gpu0.sh, gpu6.sh, gpu4 p11) were written to start the
# moment the OMNI1 review pid 917640 exited and took GPU0 (16.4 GB) and GPU6 (19.8 GB) during S5's load.  Those
# queues own the cards for hours, so the S5 column is measured here WITHOUT a GPU:
#   run_omni_frozen_review.py 1.2.0 --stream --teacher-forced on the S5 child, CPU, float32, every bank-v1 row scored
#   teacher-forced on OMNI1's own generated tokens (--reference-dossier = the OMNI1 S8 dossier).  Per row: argmax
#   agreement with OMNI1's stream and whether S5 predicts a stop right after it.  100 % agreement on every row means
#   S5's greedy decode reproduces OMNI1's (up to bf16-vs-fp32 near-ties), i.e. the S5 column equals OMNI1's.
# CPU only: nice 15, 16 threads.  usage: setsid nohup s8_s5_proxy.sh > logs/s5_proxy.out 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
EF=$OC/import/rq27b/tree_ef1ffb9
PY=/home/cereal/model-surgery-lab/venv/bin/python
S=$OC/composite/s8
OUT=$S/review/oc-dossier-OC02-C2K4352K3U-PROXY-vs-OMNI1-CPU-FP32.json
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists"; exit 3; }
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC_CALIBRATION_DIR=$OC/calibration CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
cd $OC
sha256sum $OC/composite/tools/run_omni_frozen_review.py $S/review/oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json $0
nice -n 15 /usr/bin/time -v $PY $OC/composite/tools/run_omni_frozen_review.py \
  --model $OC/composite/c10/S5-C2K4352K3U/child --tokenizer-dir /mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07 \
  --chimera-src $EF --label OC02-C2K4352K3U-PROXY-vs-OMNI1 --parent-code Q38-27BU --stage fold \
  --stream --teacher-forced --maximum-new-tokens 0 --reference-dossier $S/review/oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json \
  --device cpu --dtype float32 --micro-batch 4 --thinking off --ledger-band-logit-mask \
  --bank calibration/banks/lfm2_semantic_whole_response_prompts_v1.json \
  --probe-prompts calibration/banks/matrix_campaign_probe_prompts_v1.json --skip-smoke \
  --out "$OUT"
echo "proxy_exit=$? $(date -u +%FT%TZ)"
