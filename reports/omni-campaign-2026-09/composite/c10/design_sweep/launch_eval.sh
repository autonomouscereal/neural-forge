#!/bin/sh
# C10 D4 design sweep, evaluation (2026-10-11, Claude): c10_virtual_child_check.py >= 1.2.0.
# Every sweep variant is a VIRTUAL 27B child (S3 trunk + the variant's union FFNs, never written);
# all ten run in ONE layer-major bf16 pass beside the S3 trunk and the S0 import of the 27B parent
# (reference, read lazily from the NAS).  Scored on the 8 R-Q prompts and the 6 held rows of the
# 27B evidence manifest.  GPU (default 6) is the nvidia-smi index; check < 2000 MiB first.
#   GPU=6 setsid nohup sh launch_eval.sh > logs/eval.log 2>&1 < /dev/null &
# Split runs (2026-10-11: the one-pass run measured 591 s per layer for 12 models -> ~10 h; the forward is
# host-bound, so parallel processes scale): PART=<name> VARIANTS="<n1> <n2> ..." writes
# eval/virtual_check_<PART>.json; every part re-runs the trunk and the parent beside its own children.
#   GPU=6 PART=p1 VARIANTS="legacy a0 a003" THREADS=6 setsid nohup sh launch_eval.sh > logs/eval_p1.log 2>&1 < /dev/null &
OC=/home/cereal/omni-campaign
D=$OC/composite/c10/design_sweep
EF=$OC/import/rq27b/tree_ef1ffb9
V=""
for n in ${VARIANTS:-legacy a0 a003 a01 a03 b01 b025 b05 g01 g03}; do V="$V --variant $n=$D/sweep/variants/$n"; done
T=${THREADS:-8}
mkdir -p $D/eval
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CHIMERA_SRC=$EF/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-6}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $OC
exec nice -n 15 /usr/bin/time -v /home/cereal/model-surgery-lab/venv/bin/python composite/tools/c10_virtual_child_check.py \
  --trunk /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native \
  --k3-channels 12288 --parent-parameter-count 27781427952 --media-units-parameters 591437129 \
  --tokenizer-dir /mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07 \
  --reference parent=/mnt/ai_data/omni-campaign/s0-probe-sep04 \
  --evidence $OC/qwen-fold/evidence/omni-evidence-27bu-v1.json \
  --device cuda:0 --dtype bfloat16 --position-chunk 512 \
  $V --out $D/eval/virtual_check_${PART:-variants}.json "$@"
