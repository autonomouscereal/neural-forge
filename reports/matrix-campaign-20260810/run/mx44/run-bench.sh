#!/bin/bash
# ADDENDUM 8 benchmark column for one model directory (parent baseline or child).
# Wave 8's `run/mxl/run-bench.sh` verbatim except for the output root, because
# every number it produces is only comparable to the parent's if nothing about
# the invocation moved: same tasks, same --limit, same --seed, same batch size,
# same max_length, same dtype (float32, matching the frozen review's own
# reconstitution), same GPU architecture.
#
#   $1  label        e.g. PARENT | ARCH.R0
#   $2  model dir
#   $3  gpu index    0 | 4  (both SM86 RTX 3090)
#
# --limit is a SAMPLE, so these are PARTIAL evaluations by construction; lm-eval
# prints its own warning to that effect and the log keeps it.  max_length=2048 is
# NOT a default: LFM2.5-230M declares max_position_embeddings=128000 and lm-eval
# sizes the wikitext rolling window from it, which OOMs a 24 GB 3090.
set -u
LABEL="${1:?label}"; DIR="${2:?model dir}"; GPU="${3:?gpu}"
LAB=/home/cereal/ai-benchmark-lab
OUT=/mnt/ai_data/neural-forge-matrix/wave44/bench
PY=$LAB/envs/lm-eval/bin/python

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU -> $CAP" >&2; exit 3
fi

mkdir -p $OUT/logs
cd $LAB
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/home/cereal/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m lm_eval \
  --model hf \
  --model_args pretrained=$DIR,dtype=float32,max_length=2048 \
  --tasks arc_easy,hellaswag,piqa,winogrande,lambada_openai,wikitext \
  --limit 200 --batch_size 4 --seed 1234 \
  --device cuda:0 \
  --output_path $OUT/$LABEL \
  > $OUT/logs/bench-$LABEL.log 2>&1
rc=$?
echo "bench $LABEL rc=$rc"
tail -14 $OUT/logs/bench-$LABEL.log
exit $rc
