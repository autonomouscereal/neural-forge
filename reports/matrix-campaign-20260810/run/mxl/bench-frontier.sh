#!/bin/bash
# Materialize each coherent frontier child and benchmark it, deepest first.
#
# ADDENDUM 8 ranks the frontier by net stored ratio among COHERENT children, so
# the deepest child is benchmarked first and the shallow ones are the ones that
# get dropped if the wave runs out of GPU.  Dead children are skipped: a child
# that loops a token to the cap has no intelligence-per-compression to measure,
# and its perplexity would be a number about a collapse rather than about a fold.
set -u
GPU="${1:?gpu}"; shift
W=/home/cereal/nf-mxl
HF=/mnt/ai_data/neural-forge-matrix/wave8/hf-children
PY=/home/cereal/model-surgery-lab/venv/bin/python
mkdir -p $HF
for spec in "$@"; do
  wave="${spec%%:*}"; cell="${spec##*:}"
  dir=/mnt/ai_data/neural-forge-matrix/$wave/cells/$cell
  echo "=========== $cell ($wave) ==========="
  if [ ! -d "$HF/$cell" ]; then
    $PY $W/mxl_materialize_child.py --cell-directory "$dir" --out "$HF/$cell" || { echo "MATERIALIZE FAILED $cell"; continue; }
  else
    echo "already materialized"
  fi
  $W/run-bench.sh "$cell" "$HF/$cell" "$GPU"
  # A float32 230M child is ~920 MB; keep only the manifest once it is measured.
  rm -f $HF/$cell/model.safetensors
  echo "removed $HF/$cell/model.safetensors (re-derivable from the sealed receipt)"
done
echo "BENCH LANE COMPLETE"
