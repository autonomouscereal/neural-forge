#!/bin/bash
# ADDENDUM 8 benchmark column for one model directory (parent baseline or child).
#
#   $1  label        e.g. PARENT | COMP.MA.MXL5
#   $2  model dir
#   $3  gpu index    0 | 4  (both SM86 RTX 3090)
#
# Why these tasks: ADDENDUM 8 wants a CONTINUOUS intelligence-per-compression
# measure that the 15-prompt band cannot give.  arc_easy / hellaswag / piqa /
# winogrande are zero-shot multiple-choice tasks a ~230M model scores meaningfully
# above chance on; lambada_openai and wikitext give a perplexity, which degrades
# smoothly rather than in steps and is therefore the measure that can rank two
# children the 15-prompt review calls the same thing.
#
# Everything that could make two runs incomparable is pinned: the same --limit,
# the same --seed, the same batch size, the same max_length, the same dtype
# (float32, matching the frozen review's own reconstitution), and the same GPU
# architecture.  --limit is a SAMPLE, so these are PARTIAL evaluations by
# construction; lm-eval prints its own warning to that effect and the log keeps it.
#
# max_length=2048 is NOT a default and is reported with every number it produces.
# LFM2.5-230M declares max_position_embeddings=128000, and lm-eval sizes the
# rolling window for wikitext word_perplexity from that, which asks for a 67.86
# GiB attention allocation and OOMs a 24 GB 3090 on the first request.  2048 is
# the window every model in this comparison is measured at; it changes the
# absolute perplexity and does not change the parent-child comparison, because
# both sides are measured the same way.
set -u
LABEL="${1:?label}"; DIR="${2:?model dir}"; GPU="${3:?gpu}"
LAB=/home/cereal/ai-benchmark-lab
OUT=/mnt/ai_data/neural-forge-matrix/wave11/bench
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
