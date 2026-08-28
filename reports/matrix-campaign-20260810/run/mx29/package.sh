#!/bin/bash
# The wave-29 human-review package: every graded child rendered verbatim, plus
# the receipts the rendering is bound to, plus the artifacts this wave's claims
# are read off.  Wave 15 built the same package and left it on the server; every
# wave since has committed its copy, under
# reports/matrix-campaign-20260810/wave29/human-review/.
set -u
. /home/cereal/nf-mx29/env.sh
H=$W/human-review
mkdir -p $H
for d in $W/cells/ARCH.*; do
  tag=$(basename $d | sed 's/ARCH\.//')
  [ -f $d/frozen-review-dossier.json ] || continue
  cp -f $d/frozen-review-dossier.json $H/dossier-$tag.json
  cp -f $d/cell-receipt.json $H/receipt-$tag.json
  [ -f $d/cell-receipt-behavior-reviewed.json ] && cp -f $d/cell-receipt-behavior-reviewed.json $H/verdict-$tag.json
  [ -f $d/per-case.json ] && cp -f $d/per-case.json $H/per-case-$tag.json
  $PY $MX29/mx29_read_dossier.py --dossier $d/frozen-review-dossier.json --full > $H/READ-$tag.txt 2>/dev/null
  $PY $MX29/mx29_read_dossier.py --dossier $d/frozen-review-dossier.json --which parent --full > $H/READ-PARENT-$tag.txt 2>/dev/null
done
for c in $W/children/MX29-*; do
  r=$c/mx29-build-receipt.json
  [ -f $r ] && cp -f $r $H/build-receipt-$(basename $c).json
done
# The basis is the wave's independent variable, so its censuses travel with the
# dossiers: without them "attribution-weighted beta 0.25" is a name, not a number.
for b in $W/basis/*.census.json $W/basis/endpoint-retention.json; do
  [ -f "$b" ] && cp -f "$b" $H/basis-$(basename $b)
done
# The wave's own tables, and the campaign-wide one the consolidated report is
# generated from -- the artifact, not the prose, is what a reader should be able
# to disagree with.
for t in $W/wave29-child-index.json $W/wave29-child-comparison.json          $W/campaign-frontier-table.json $W/ledger-merge-report.json          $W/children-retention-manifest-wave29.json          $W/deployments-prune-manifest-wave29.json; do
  [ -f "$t" ] && cp -f "$t" $H/
done
[ -f $W/bench/benchmark-table.json ] && cp -f $W/bench/benchmark-table.json $H/

# WAVE 29 ADDS FOUR CLASSES OF ARTIFACT, and each one is here because a claim in
# this wave's note is unreadable without it.
#
# * the EXTENSION surface.  ADDENDUM 16.1's second review surface grades nothing,
#   but the retro-score is the wave's first result and a reader cannot check it
#   from the sealed dossiers alone.  The bank and its seal travel too, so the
#   thirty prompts a child was asked can be read beside its thirty answers.
# * the PREREADS.  The emission forensic, the LIST-to-SET autopsy and the KKT
#   receipts carry measurements that no child's dossier contains -- the massive-
#   activation channel loss, the flip position, and the rung-0 discriminator that
#   explains the KKT null.
# * the MASK.  It was written before the build that cites it, and "pre-registered"
#   is a claim about a file, so the file ships.
# * the REPORTS.  The new results matrix and the refreshed scorecard are what
#   Cyril asked this wave for.
for f in $W/extension/*.json; do
  [ -f "$f" ] && cp -f "$f" $H/extension-$(basename $f | sed s/^extension-//)
done
[ -f $W/extension-score.json ] && cp -f $W/extension-score.json $H/
[ -f $NF_DEPLOY/research/lfm2_extension_bank_prompts_v1.json ] && cp -f $NF_DEPLOY/research/lfm2_extension_bank_prompts_v1.json $H/
[ -f $NF_DEPLOY/reports/matrix-campaign-20260810/extension-bank-seal.json ] && cp -f $NF_DEPLOY/reports/matrix-campaign-20260810/extension-bank-seal.json $H/
for f in $W/prereads/*.json; do [ -f "$f" ] && cp -f "$f" $H/preread-$(basename $f); done
for f in $W/report/*.md; do [ -f "$f" ] && cp -f "$f" $H/; done
[ -f $W/sealed-counts.json ] && cp -f $W/sealed-counts.json $H/
[ -f $W/mechanisms.json ] && cp -f $W/mechanisms.json $H/
[ -f $W/scorecard-characters.json ] && cp -f $W/scorecard-characters.json $H/

ls $H | wc -l
du -sb $H | cut -f1
