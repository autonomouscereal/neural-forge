#!/bin/bash
# The wave-19 human-review package: every graded child rendered verbatim, plus
# the receipts the rendering is bound to.  Wave 15 built the same package and
# left it on the server; wave 16 committed its copy and so does this one, under
# reports/matrix-campaign-20260810/wave20/human-review/.
set -u
. /home/cereal/nf-mx20/env.sh
NF=/home/cereal/nf-mx20
H=$W/human-review
mkdir -p $H
for d in $W/cells/ARCH.*; do
  tag=$(basename $d | sed 's/ARCH\.//')
  [ -f $d/frozen-review-dossier.json ] || continue
  cp -f $d/frozen-review-dossier.json $H/dossier-$tag.json
  cp -f $d/cell-receipt.json $H/receipt-$tag.json
  [ -f $d/cell-receipt-behavior-reviewed.json ] && cp -f $d/cell-receipt-behavior-reviewed.json $H/verdict-$tag.json
  [ -f $d/per-case.json ] && cp -f $d/per-case.json $H/per-case-$tag.json
  $PY $NF/mx20_read_dossier.py --dossier $d/frozen-review-dossier.json --full > $H/READ-$tag.txt 2>/dev/null
  $PY $NF/mx20_read_dossier.py --dossier $d/frozen-review-dossier.json --which parent --full > $H/READ-PARENT-$tag.txt 2>/dev/null
done
for c in $W/children/MX20-*; do
  for r in $c/mx20-build-receipt.json; do
    [ -f $r ] && cp -f $r $H/build-receipt-$(basename $c).json
  done
done
ls $H | wc -l
du -sh $H
