#!/bin/bash
# Wave 22: seal one ADDENDUM 6 verdict per graded child.  Every count here is a
# reading of the frozen dossier text, taken case by case; the per-case file is
# written first and the seal is refused if it disagrees with the totals.
#
# The reading standard is waves 19, 20 and 21's, calibrated against them
# deliberately, because every comparison this wave makes is against a child one
# of those waves sealed.  Two of those calibrations decide cases in this wave as
# well and are restated so a reader can disagree with them here rather than
# reconstruct them from three notes:
#
# * `ARCH.W896`'s sealed Partial on `instruction-three-actions` is the text
#   "- Organize - Review - Update - Review - Review-" -- THREE distinct bare
#   verbs, no detail.  One action repeated three times is a miss (wave 21).
# * `ARCH.W896`'s sealed Partial on `writing-concise-email` is the ASK with no
#   email form; wave 19 credited content over form, so form with no ask is a
#   miss (wave 21).
#
# **The verdicts are appended below when the dossiers have been read, and not
# before.**  This file ships in the builder commit with the seal function only,
# because a driver that carries last wave's verdicts is a driver that can seal
# last wave's reading against this wave's receipts.
set -u
. /home/cereal/nf-mx22/env.sh
seal () {
  TAG="$1"; U="$2"; P="$3"; M="$4"; SAFETY="$5"; NOTES="$6"; CASES="$7"; LANES="$8"
  mkdir -p $W/cells/ARCH.$TAG
  $PY $MX22/mx22_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX22/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}

echo "SEALQ DONE"
