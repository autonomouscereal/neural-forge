#!/bin/sh
# M4 load/identity check of the c10 child DEFINED by child.partial (c10_virtual_child_check.py >= 1.1.0).
# Full child: sh vcheck_launch.sh            Prefix rehearsal: sh vcheck_launch.sh --pad-missing-with-trunk --out <json>
# GPU (env, default 0) is the nvidia-smi index; check it is < 2000 MiB first.
OC=/home/cereal/omni-campaign
R=$OC/composite/c10/rehearsal-k3-1-60
export CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd $OC
exec nice -n 19 /usr/bin/time -v /home/cereal/model-surgery-lab/venv/bin/python composite/tools/c10_virtual_child_check.py \
  --trunk /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native \
  --partial $R/child.partial --k3-channels 12288 \
  --parent-parameter-count 27781427952 --media-units-parameters 591437129 \
  --tokenizer-dir /mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07 \
  --device cuda:0 --dtype bfloat16 --out $R/virtual_child_check.json "$@"
