#!/bin/sh
# S5 checks with c10_virtual_child_check.py 1.2.0 (tree ef1ffb9; GPU = nvidia-smi index, check < 2000 MiB first).
#   MODE=mid  : HYBRID prefix check on a hardlinked snapshot of child.partial L00..L$LAST (default 23), trunk FFN
#               zero-padded above it (R-Q top-1 / KL / residual relL2 vs the S3 trunk); verdict rule: switch to g01
#               if R-Q top-1 vs trunk < 0.80 or the residual norm ratio > 1.5x (bounded by 1 + residual relL2).
#   MODE=full : every layer of child.partial as a virtual child (--variant a003=...) beside the S3 trunk and the
#               27B parent S0 import (--reference), R-Q prompts + the 6 held evidence rows: the sweep's metrics.
#   MODE=written : the WRITTEN child as --reference beside the virtual child from $S/partial_snapshot: its own
#               forward (finite logits, held NLL, top-1 vs trunk) and its agreement with the virtual child.
OC=/home/cereal/omni-campaign
S=$OC/composite/c10/S5-C2K4352K3U
EF=$OC/import/rq27b/tree_ef1ffb9
S3N=/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native
T=${THREADS:-8}
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CHIMERA_SRC=$EF/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--trunk $S3N --k3-channels 12288 --parent-parameter-count 27781427952 --media-units-parameters 591437129 \
  --tokenizer-dir /mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07 --device cuda:0 --dtype bfloat16"
PY=/home/cereal/model-surgery-lab/venv/bin/python
cd $OC
case "${MODE:?set MODE}" in
  mid)
    LAST=${LAST:-23}; M=$S/mid_L00-$LAST; P=$S/child.partial
    mkdir -p $M; cp -p $P/partial_inputs.json $M/
    for k in $(seq -w 0 $LAST); do cp -p $P/L$k.json $M/; ln -f $P/L$k.safetensors $M/L$k.safetensors; done
    exec nice -n 15 /usr/bin/time -v $PY composite/tools/c10_virtual_child_check.py $COMMON \
      --partial $M --pad-missing-with-trunk --out $S/vcheck_mid_L00-$LAST.json ;;
  full)
    exec nice -n 15 /usr/bin/time -v $PY composite/tools/c10_virtual_child_check.py $COMMON \
      --variant a003=$S/child.partial --reference parent=/mnt/ai_data/omni-campaign/s0-probe-sep04 \
      --evidence $OC/qwen-fold/evidence/omni-evidence-27bu-v1.json --position-chunk 512 \
      --out $S/vcheck_full_partial.json ;;
  written)
    exec nice -n 15 /usr/bin/time -v $PY composite/tools/c10_virtual_child_check.py $COMMON \
      --variant a003=$S/partial_snapshot --reference written=$S/child \
      --evidence $OC/qwen-fold/evidence/omni-evidence-27bu-v1.json --position-chunk 512 \
      --out $S/vcheck_written_child.json ;;
esac
