#!/bin/bash
# Wave 23: the vocabulary stack.  Every rung is a POST-PROCESS on one sealed
# child's own bytes -- a row gather on `model.embed_tokens.weight` and a
# closure-corrected tokenizer remap -- so it costs no GPU, solves nothing, and
# leaves the source child's body bit-identical.  That is why the whole ladder is
# built in one CPU pass and only the reviews are rationed.
#
# The source of every rung is NAMED here rather than rebuilt, because a rebuilt
# body is not the sealed body and the campaign has paid for that lesson twice.
# Its axes are not restated either: `mx23_vocab.py` carries them verbatim out of
# the source's own build receipt -- basis digest, retained energies, full
# spectrum and all -- because a composed child that re-describes its parent's
# provenance in a driver script is a child whose provenance is that script's
# opinion.  `--source-cell-receipt` names the sealed row it stands on.
#
#   vocabq.sh <K> [tag ...]     (no tags = the whole ladder)
set -u
. /home/cereal/nf-mx23/env.sh
K="${1:?K}"; shift
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/children $W/logs

run () {
  TAG="$1"; SRC="$2"; CELL="$3"
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; return; fi
  rm -rf $W/children/MX23-$TAG
  $PY $MX23/mx23_vocab.py --mode build --k $K --tag $TAG \
    --source "$SRC" --source-cell-receipt "$CELL" \
    --child $W/children/MX23-$TAG \
    > $W/logs/vocabbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/vocabbuild-$TAG.log | tail -1)"
}

for TAGSEL in ${@:-V16 V16X13 V16W896 V16W896X13 V16CL896 V16W896LS V16W896LSX13 V16W896KV6JS V16X07KV6JQC V16W768 V16W736}; do
case $TAGSEL in
  # the pure axis: the vocabulary fold alone, which is what makes every rung
  # below it attributable
  V16)          run V16          /home/cereal/model-surgery-lab/models/LFM2.5-230M "" ;;
  # 0.7284x -- the campaign's HIGHEST scoring child (13/15) with the vocabulary
  # folded on top, and the cleanest test L3 has ever been given: the two folds
  # share no tensor
  V16X13)       run V16X13       $M/wave16/children/MX16-X13A     $M/wave16/cells/ARCH.X13A/cell-receipt.json ;;
  # 0.6673x -- the multiplying pair.  The width fold narrows the table's columns
  # and the vocabulary fold cuts its rows: 16,384 x 896 against 65,536 x 1,024
  V16W896)      run V16W896      $M/wave19/children/MX19-W896     $M/wave19/cells/ARCH.W896/cell-receipt.json ;;
  V16W896X13)   run V16W896X13   $M/wave19/children/MX19-W896X13  $M/wave19/cells/ARCH.W896X13/cell-receipt.json ;;
  # 0.4693x -- the campaign frontier body with the vocabulary folded on it, and
  # the first standard-shape child ever built under half the parent that was not
  # built by narrowing the residual to 512
  V16CL896)     run V16CL896     $M/wave15/children/MX15-CL896    $M/wave15/cells/ARCH.CL896/cell-receipt.json ;;
  V16W896LS)    run V16W896LS    $M/wave21/children/MX21-W896LS   $M/wave21/cells/ARCH.W896LS/cell-receipt.json ;;
  V16W896LSX13) run V16W896LSX13 $M/wave21/children/MX21-W896LSX13 $M/wave21/cells/ARCH.W896LSX13/cell-receipt.json ;;
  V16W896KV6JS) run V16W896KV6JS $M/wave19/children/MX19-W896KV6JS $M/wave19/cells/ARCH.W896KV6JS/cell-receipt.json ;;
  V16X07KV6JQC) run V16X07KV6JQC $M/wave18/children/MX18-X07KV6JQC $M/wave18/cells/ARCH.X07KV6JQC/cell-receipt.json ;;
  V16W768)      run V16W768      $M/wave19/children/MX19-W768     $M/wave19/cells/ARCH.W768/cell-receipt.json ;;
  V16W736)      run V16W736      $M/wave21/children/MX21-W736     $M/wave21/cells/ARCH.W736/cell-receipt.json ;;
  *) echo "$TAGSEL: no such rung" ;;
esac
done
echo "VOCABQ DONE"
