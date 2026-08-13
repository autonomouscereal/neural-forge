#!/bin/bash
# Wave 25: seal one ADDENDUM 6 verdict per graded child.  Every count here is a
# reading of the frozen dossier text, taken case by case; the per-case file is
# written first and the seal is refused if it disagrees with the totals.
#
# NOTE ON THIS FILE'S SYNTAX, because wave 23 lost text to it: a backtick inside
# a double-quoted bash string opens a command substitution and EATS everything to
# the next backtick.  Wave 23's sealed notes have holes in them for exactly that
# reason.  There is not one backtick anywhere below and there never will be.
#
# THE READING STANDARD.  Waves 19-24's, restated where it decides a case here.
#
#   A lane is CHARGED (degenerate) if it hits the generation cap, returns the
#   empty string, returns the PROMPT ITSELF, or carries the word-substitution
#   signature.  A CASE is held (Partial) if the lane's substantive asks are
#   answered in part and Useful if they are answered in full.  The two readings
#   are independent: a lane can carry a substitution and still answer, and wave
#   23's own V16X13 sealed a Partial on a lane it described as garbled.  What is
#   NOT allowed is to let a charged lane pass unnamed, which is why every child
#   below names its charged set explicitly.
#
# Calibrations carried from the sealed record and applied in BOTH directions:
#
# * writing-concise-email.  Wave 21 sealed ARCH.W896's Partial as the ASK with no
#   email form and W896LS's MISS as the form with no ask, so THE ASK is the
#   discriminating criterion; wave 23 sealed V16 Useful for carrying both asks
#   inside a salutation and a sign-off under the word limit, and sealed V16X13 a
#   MISS for inverting the prompt's own dates.  Wave 24 read V8-family children
#   Useful for carrying all three named items even where the prose is telegraphic,
#   and read the V16-family children misses for inverting Friday to Monday.
# * instruction-three-actions.  Wave 23 sealed V16 Useful for three distinct
#   detailed actions DESPITE an introduction the prompt forbids, so the sealed
#   criterion on this lane is the three actions and not the preamble; a child
#   whose three items are two items repeated is read a miss (wave 24's V8X13G).
# * planning-dependencies.  Wave 20 sealed a Partial for four of the five named
#   work items and wave 23 read V16 a MISS for naming one of five in six
#   well-formed paragraphs, so the lane keys on the ITEMS.  Wave 24's V32 named
#   all five in six steps and is the only child read Useful here.
# * arithmetic-multi-step.  The PARENT gets this lane wrong (its sealed lane
#   boxes "3 kits per box, 94 kits left over"), so a child that gets it right is
#   getting it right on its own.
#
# THE SEALS THEMSELVES ARE WRITTEN AFTER THE REVIEWS AND NEVER BEFORE.  This file
# ships to the server as the wave's reading, one seal call per graded child; it
# is deliberately EMPTY in the deployment cut before the reviews exist, because a
# seal body inherited from last wave is last wave's reading of this wave's text.
set -u
. /home/cereal/nf-mx25/env.sh
seal () {
  TAG="$1"; U="$2"; P="$3"; M="$4"; SAFETY="$5"; NOTES="$6"; CASES="$7"; LANES="$8"
  mkdir -p $W/cells/ARCH.$TAG
  $PY $MX25/mx25_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX25/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}

echo "SEALQ: no seals in this cut -- the wave's readings are uploaded to /home/cereal/nf-mx25/sealq.sh after the dossiers exist, and the final version is committed with the wave."
