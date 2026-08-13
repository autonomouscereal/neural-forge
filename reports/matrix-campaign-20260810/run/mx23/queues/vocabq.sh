#!/bin/bash
# Wave 23: the vocabulary stack.  Every rung is a POST-PROCESS on one sealed
# child's own bytes -- a row gather on `model.embed_tokens.weight` and a
# closure-corrected tokenizer remap -- so it costs no GPU, solves nothing, and
# leaves the source child's body bit-identical.  That is why the whole ladder is
# built in one CPU pass and only the reviews are rationed.
#
# The source of every rung is named here rather than rebuilt, because a rebuilt
# body is not the sealed body and the campaign has paid for that lesson twice.
# `--source-cell-receipt` carries the sealed cell receipt forward so the
# composed child's provenance names the row it is standing on.
#
#   vocabq.sh <K> [tag ...]     (no tags = the whole ladder)
set -u
. /home/cereal/nf-mx23/env.sh
K="${1:?K}"; shift
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs

# tag | source child dir | sealed cell receipt | axes carried from the source
run () {
  TAG="$1"; SRC="$2"; CELL="$3"; shift 3
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; return; fi
  $PY $MX23/mx23_vocab.py --mode build --k $K --tag $TAG \
    --source "$SRC" --source-cell-receipt "$CELL" \
    --child $W/children/MX23-$TAG "$@" \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
}

want () { [ $# -eq 0 ] && return 0; for t in "$@"; do [ "$t" = "$TAGSEL" ] && return 0; done; return 1; }

for TAGSEL in ${@:-V16 V16X13 V16W896 V16W896X13 V16W896LS V16W896LSX13 V16W896KV6JS V16X07KV6JQC V16CL896 V16W768 V16W736}; do
case $TAGSEL in
  V16)
    run V16 /home/cereal/model-surgery-lab/models/LFM2.5-230M "" ;;
  V16X13)
    run V16X13 $M/wave16/children/MX16-X13A $M/wave16/cells/ARCH.X13A/cell-receipt.json \
      --excised 13 \
      --axis '{"axis":"num_hidden_layers","parent":14,"child":13,"mechanism":"block 13 (conv) excised, rank-by-solo; the sealed ARCH.X13A body verbatim","solved":true}' ;;
  V16W896)
    run V16W896 $M/wave19/children/MX19-W896 $M/wave19/cells/ARCH.W896/cell-receipt.json \
      --axis '{"axis":"hidden_size","parent":1024,"child":896,"mechanism":"pooled reader-PCA residual basis, conditioned-local sequential re-solve; the sealed ARCH.W896 body verbatim","solved":true}' ;;
  V16W896X13)
    run V16W896X13 $M/wave19/children/MX19-W896X13 $M/wave19/cells/ARCH.W896X13/cell-receipt.json \
      --excised 13 \
      --axis '{"axis":"hidden_size","parent":1024,"child":896,"mechanism":"the sealed ARCH.W896X13 body verbatim","solved":true}' \
      --axis '{"axis":"num_hidden_layers","parent":14,"child":13,"mechanism":"block 13 excised inside the sealed pair","solved":true}' ;;
  V16W896LS)
    run V16W896LS $M/wave21/children/MX21-W896LS $M/wave21/cells/ARCH.W896LS/cell-receipt.json \
      --axis '{"axis":"hidden_size","parent":1024,"child":896,"mechanism":"the sealed ARCH.W896LS body verbatim (writers lambda on an internal fit split)","solved":true}' ;;
  V16W896LSX13)
    run V16W896LSX13 $M/wave21/children/MX21-W896LSX13 $M/wave21/cells/ARCH.W896LSX13/cell-receipt.json \
      --excised 13 \
      --axis '{"axis":"hidden_size","parent":1024,"child":896,"mechanism":"the sealed ARCH.W896LSX13 body verbatim","solved":true}' \
      --axis '{"axis":"num_hidden_layers","parent":14,"child":13,"mechanism":"block 13 excised inside the sealed pair","solved":true}' ;;
  V16W896KV6JS)
    run V16W896KV6JS $M/wave19/children/MX19-W896KV6JS $M/wave19/cells/ARCH.W896KV6JS/cell-receipt.json \
      --axis '{"axis":"hidden_size","parent":1024,"child":896,"mechanism":"the sealed ARCH.W896KV6JS body verbatim","solved":true}' \
      --axis '{"axis":"num_attention_heads","parent":16,"child":12,"mechanism":"KV groups merged 8 to 6 inside the sealed pair","solved":true}' ;;
  V16X07KV6JQC)
    run V16X07KV6JQC $M/wave18/children/MX18-X07KV6JQC $M/wave18/cells/ARCH.X07KV6JQC/cell-receipt.json \
      --excised 7 \
      --axis '{"axis":"num_hidden_layers","parent":14,"child":13,"mechanism":"the sealed ARCH.X07KV6JQC body verbatim -- the best two-axis child in the campaign","solved":true}' \
      --axis '{"axis":"num_attention_heads","parent":16,"child":12,"mechanism":"KV groups merged 8 to 6 inside the sealed pair","solved":true}' ;;
  V16CL896)
    run V16CL896 $M/wave15/children/MX15-CL896 $M/wave15/cells/ARCH.CL896/cell-receipt.json \
      --axis '{"axis":"intermediate_size","parent":2560,"child":896,"mechanism":"the sealed ARCH.CL896 body verbatim -- the campaign frontier, uniform inner coreset, conditioned-local","solved":true}' ;;
  V16W768)
    run V16W768 $M/wave19/children/MX19-W768 $M/wave19/cells/ARCH.W768/cell-receipt.json \
      --axis '{"axis":"hidden_size","parent":1024,"child":768,"mechanism":"the sealed ARCH.W768 body verbatim","solved":true}' ;;
  V16W736)
    run V16W736 $M/wave21/children/MX21-W736 $M/wave21/cells/ARCH.W736/cell-receipt.json \
      --axis '{"axis":"hidden_size","parent":1024,"child":736,"mechanism":"the sealed ARCH.W736 body verbatim","solved":true}' ;;
  *) echo "$TAGSEL: no such rung" ;;
esac
done
echo "VOCABQ DONE"
