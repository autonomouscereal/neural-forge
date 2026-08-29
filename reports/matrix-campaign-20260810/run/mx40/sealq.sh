#!/bin/bash
# Wave 29: seal one ADDENDUM 6 verdict per graded child.  Every count here is a
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
# THE CALIBRATION THIS WAVE RAN BEFORE IT GRADED ANYTHING.  Wave 29 stacks off
# ARCH.V19 -- the 14/15 summit wave 27 sealed at 0.794570x -- and off ARCH.V18
# and ARCH.V20, the two 13/15 shoulders.  So the reader that produced the summit
# number is the reader that has to be shown to still be the same reader, and it
# is shown on THAT row: ARCH.V19's own wave-27 dossier was re-graded blind
# against the standard below before any wave-29 child was opened, and the parent
# baseline was re-read on the same pass.  Both reproduce the sealed record.  If
# either had not, nothing below would have been graded.
#
# THE COMPOSITION RULE, carried from wave 27 unchanged, because this wave grades
# more composed children than wave 27 did.  A composed child is graded on its OWN
# text and never against what its axes "should" cost: a reader who knows V19X13
# "ought" to read three below V19 will find three.  Every seal below was written
# from the READ- file before its arithmetic was computed.
#
# THE RULE THIS WAVE ADDS, for the bottom of the spectrum.  Wave 27 sealed the
# campaign's first 0/15 and observed that summarisation is the last lane to go.
# That observation is now a THING TO CHECK rather than a thing to expect: on
# every child under 0.60x below, the summarisation lane is read FIRST and its
# reading is written down before the other fifteen are opened, so the claim
# cannot be confirmed by a reader who already believes it.
#
# Carried calibrations, applied in BOTH directions:
#
# * writing-concise-email.  Wave 21 sealed W896's Partial as the ASK with no
#   email form and W896LS's MISS as the form with no ask, so THE ASK is the
#   discriminating criterion; wave 23 sealed V16 Useful for both asks inside a
#   salutation and a sign-off; wave 24 read the V16 family a MISS for inverting
#   Friday to Monday.  This wave adds one distinction it had to make eight times
#   and states it once: a child that carries all three named items and REQUESTS
#   the tracking number is Useful, and a child that carries all three and offers
#   to SUPPLY the tracking number has inverted the ask and is Partial.
# * instruction-three-actions.  Wave 23 sealed V16 Useful for three distinct
#   detailed actions DESPITE a forbidden preamble, so the criterion is the three
#   actions; a child whose three items are two items repeated is a miss.
# * planning-dependencies.  Wave 20 sealed a Partial for four of the five named
#   work items and wave 23 read a MISS for one of five, so the lane keys on the
#   ITEMS: five in six steps is Useful, four is Partial, three or fewer is a miss.
# * arithmetic-multi-step.  The PARENT gets this lane wrong (its sealed lane
#   boxes 3 kits per box and 94 left over), so a child that gets it right is
#   getting it right on its own.  A child that carries the chain correctly to the
#   division and then botches the remainder is Partial; a child that never forms
#   17 times 6 is a miss.
# * coding-deduplicate, which no child in 682 rows has held.  Wave 24 read V32's
#   attempt a miss and the reason is in the code: it accumulates into a SET and
#   returns it, so the one thing the lane asks for -- first-seen ORDER -- is the
#   one thing it does not do.  A child that accumulates into a LIST with append
#   and returns it has the algorithm right and is Partial even when the call
#   parentheses are dropped, because a dropped token is a dropped token and a
#   wrong container is a wrong answer.  Two children this wave clear that line.
# * creative-continuity, likewise never held.  Wave 24 read V32's 1,163-character
#   continuation a miss because it invented a new character and then contradicted
#   the scene.  A continuation that carries the fog, the keeper and the question
#   motif forward in coherent prose is Partial even if it renames the keeper;
#   a continuation that does neither is a miss.
set -u
. /home/cereal/nf-mx40/env.sh
seal () {
  TAG="$1"; U="$2"; P="$3"; M="$4"; SAFETY="$5"; NOTES="$6"; CASES="$7"; LANES="$8"
  # LANES is ONE argument.  This function reads "$8" and nothing after it, so a
  # call that passes charged lanes as separate bare words seals only the FIRST
  # of them and lets the rest pass unnamed -- which is the one thing the reading
  # standard at the top of this file forbids.  Wave 29 did exactly that on its
  # first pass, caught it in the sealer's own "1 of 16 lanes charged" line
  # BEFORE the ledger merge, deleted the seven receipts and shards, and re-ran.
  # The refusal that caught it was the sealer printing what it had actually been
  # given rather than what it had been meant to be given.
  if [ "$#" -ne 8 ]; then
    echo "SEAL REFUSED: $TAG passed $# arguments, not 8 -- the charged-lane list must be ONE quoted argument" >&2
    return 3
  fi
  mkdir -p $W/cells/ARCH.$TAG
  $PY $MX40/mx40_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX40/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}



# --- wave 40 seals are appended below, AFTER the dossiers are read ---
#
# THE PROTECTION IS EXERCISED FOR THE EIGHTH TIME, and what it refused on this
# time is NEW.  Wave 32 found its own cut had carried wave 31's nineteen seal
# calls forward with the substitution applied and every COUNT intact, added this
# file to HAND_EDIT and made the audit assert that the marker names THIS wave.
# Wave 33 deleted fifteen calls by hand, wave 34 six, wave 35 eight, wave 36
# eight, wave 37 six, wave 38 six, wave 39 six.  Wave 40 is the eighth -- and it
# is the FIRST cut where the file arrived carrying a marker and NO seal calls,
# because wave 39 built and sealed two children and then never graded either of
# them.  So the marker was the only thing to replace, and the three path
# resolutions a verbatim copy cannot fix for itself -- last wave's env.sh, last
# wave's per-case writer and last wave's seal-verdict script -- were retargeted
# by hand at pinned counts of one each.
#
# WAVE 39'S TWO CHILDREN ARE NOT GRADED HERE, AND THE REASON CHANGED MID-WAVE.
# Wave 40's pre-registration says it would read the two ungraded wave-39 dossiers
# on the share and seal their verdicts under wave 39's own receipts.  While wave
# 40 was building, WAVE 39 SEALED AND MERGED THEM ITSELF: the ledger moved from
# 988 rows to 992 and both children read 1/15.  So wave 40 grades neither, claims
# neither, and its merge starts from 992 rather than 988.  The two seal calls
# this file arrived carrying are wave 39's OWN work, now landed by wave 39, which
# is exactly why leaving them here would have been a duplicate seal and not a
# rescue.
#
# THE CALIBRATION THIS WAVE RUNS BEFORE IT GRADES ANYTHING is the parent lane,
# unchanged: the parent's own responses are re-read blind from this wave's first
# dossier against the standard at the top of this file and must reproduce the
# sealed reference of 5 Useful / 7 Partial / 3 misses = 12/15.  If it does not,
# nothing below is graded.  TWELFTH consecutive calibration.
#
# AND IT WAS RUN AND IT IS EXACT.  The parent's own responses were re-read blind
# from ARCH.V14CL896X13's dossier -- the first dossier of the wave -- against the
# standard at the top of this file: 5 Useful (instruction-three-actions,
# summarization-priorities, coding-deduplicate, writing-concise-email,
# uncertainty-insufficient-data), 7 Partial (greeting, explanation-seasons,
# arithmetic-multi-step, context-retrieval, debugging-off-by-one,
# planning-dependencies, creative-continuity) and 3 misses (logic-ordering,
# comparison-ram-rom, multi-turn-correction) = 12/15, cap-hits 0/16.  Both
# signature defects are present verbatim: the arithmetic Partial boxes 3 kits per
# box with 94 left over, and the multi-turn miss offers Vegan Peanut Butter and
# Jelly Bars to a table with two severe peanut allergies.  TWELFTH consecutive
# calibration and the twelfth exact one.
#
# THE RUNNER CONTROL IS NOW EIGHT WAVES AND TWENTY-THREE DOSSIERS WIDE, the
# strongest it has ever been.  mx40_parent_identity.py reads
# 865f006e3813712ec202cb23 on both wave-39 dossiers, and
# wave39/lane-byte-comparison-wave39.json computes the same digest over the
# sealed dossiers of waves 15, 23, 34, 35, 36, 37 and 38 as well -- twenty-three
# dossiers, EIGHT waves, ONE digest, on the same protocol e5e377688597 and prompt
# pack 0ee6bf14cef5, with exactly one review surface seen across all of them.
#
# THE BYTE COMPARISON RAN BEFORE ANY COUNT BELOW WAS WRITTEN AND IT IS THE WHOLE
# REASON THIS WAVE'S COMPARISON IS A MEASUREMENT:
# * ARCH.V14CL896X13 shares ZERO of sixteen lanes with ARCH.V14CL896X09 and
#   ARCH.V10CL896X13 shares ZERO of sixteen with ARCH.V10CL896X09.  Those are the
#   two pairs this wave exists for -- the same K, the same parameter count TO THE
#   BYTE, one conv block index apart -- so a count difference between them is two
#   independent decodes disagreeing and not one recipe read twice.
# * ARCH.V10CL896X13 shares ZERO with ARCH.CL896X13 and ZERO with
#   ARCH.V12CL896X13G03, the two children whose sealed summarisation lane is the
#   precedent its own summarisation lane is graded on.  So that Partial is an
#   independent decode landing on the same content, not a re-read of one decode.
# * ARCH.V14CL896X13 and ARCH.V10CL896X13 share exactly ONE of sixteen lanes with
#   each other (arithmetic-multi-step), and each shares THREE with
#   ARCH.V12CL896X13.  Nothing that decides a case below is shared.
#
# THE READING ORDER THIS FILE REQUIRES FOR CHILDREN UNDER 0.60x WAS FOLLOWED: the
# summarisation lane of both children was read FIRST and written down before any
# other lane was opened, and on this body that is the lane that decides one of
# the two seals.


echo "SEALQ DONE"
