#!/bin/bash
# The wave-20 human-review package: every graded child rendered verbatim, plus
# the receipts the rendering is bound to, plus the artifacts this wave's claims
# are read off.  Wave 15 built the same package and left it on the server; every
# wave since has committed its copy, under
# reports/matrix-campaign-20260810/wave20/human-review/.
set -u
. /home/cereal/nf-mx20/env.sh
H=$W/human-review
mkdir -p $H
for d in $W/cells/ARCH.*; do
  tag=$(basename $d | sed 's/ARCH\.//')
  [ -f $d/frozen-review-dossier.json ] || continue
  cp -f $d/frozen-review-dossier.json $H/dossier-$tag.json
  cp -f $d/cell-receipt.json $H/receipt-$tag.json
  [ -f $d/cell-receipt-behavior-reviewed.json ] && cp -f $d/cell-receipt-behavior-reviewed.json $H/verdict-$tag.json
  [ -f $d/per-case.json ] && cp -f $d/per-case.json $H/per-case-$tag.json
  $PY $MX20/mx20_read_dossier.py --dossier $d/frozen-review-dossier.json --full > $H/READ-$tag.txt 2>/dev/null
  $PY $MX20/mx20_read_dossier.py --dossier $d/frozen-review-dossier.json --which parent --full > $H/READ-PARENT-$tag.txt 2>/dev/null
done
for c in $W/children/MX20-*; do
  r=$c/mx20-build-receipt.json
  [ -f $r ] && cp -f $r $H/build-receipt-$(basename $c).json
done
# The basis is the wave's independent variable, so its censuses travel with the
# dossiers: without them "attribution-weighted beta 0.25" is a name, not a number.
for b in $W/basis/*.census.json $W/basis/selector-retention-*.json $W/basis/subspace-*.json; do
  [ -f "$b" ] && cp -f "$b" $H/basis-$(basename $b)
done
# The two probes are not ledger rows and their evidence would otherwise live
# nowhere a reader looks.
for p in $W/probes/*-receipt.json $W/probes/*-dossier.json; do
  [ -f "$p" ] && cp -f "$p" $H/probe-$(basename $p)
done
for p in $W/probes/*-dossier.json; do
  [ -f "$p" ] || continue
  tag=$(basename $p | sed 's/-dossier.json//')
  $PY $MX20/mx20_read_dossier.py --dossier $p --full > $H/READ-PROBE-$tag.txt 2>/dev/null
done
[ -f $W/bench/benchmark-table.json ] && cp -f $W/bench/benchmark-table.json $H/
ls $H | wc -l
du -sh $H
