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
. /home/cereal/nf-mx43/env.sh
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
  $PY $MX43/mx43_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX43/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}



# --- wave 43 seals are appended below, AFTER the dossiers are read ---
#
# THE PROTECTION IS EXERCISED FOR THE TENTH TIME.  Wave 32 found its own cut had
# carried wave 31's nineteen seal calls forward with the substitution applied and
# every COUNT intact, added this file to HAND_EDIT and made the audit assert that
# the marker names THIS wave.  Wave 33 deleted fifteen calls by hand, wave 34
# six, wave 35 eight, wave 36 eight, wave 37 six, wave 38 six, wave 39 six, wave
# 40 none -- and wave 40's cut then discovered why none was the wrong answer: it
# had carried TWO of wave 39's calls under a rewritten marker and the audit said
# nothing, because the marker is one line at the top and the calls are anywhere
# below it.  Wave 40 added the pin that fixes that -- audit() counts ^seal <TAG>
# and requires ZERO at cut time.  Wave 41 was the first cut where it had work to
# do from the start and deleted eleven; wave 42 is the second and deleted TEN.
# WAVE 43 IS THE THIRD AND DELETED THREE.
#
# AND WAVE 43 IS THE FIRST CUT TO EXERCISE WAVE 42'S GENERAL PIN, WHICH FIRED.
# `NOTES-MX-AT` section 11.4 predicted it would report 78 residue lines against
# `run/mx42/sealq.sh` as shipped.  It reported EXACTLY 78, the first of them the
# sentence that begins AND IT WAS RUN AND IT IS EXACT ON THE COUNT THAT MATTERS,
# and the block it named carried wave 42's calibration result, wave 42's nine-
# dossier byte comparison, wave 42's reading-order finding and wave 42's three
# seal calls with every count intact.  All 78 were deleted by hand.  The
# prediction and the count agree, which is the only reason this paragraph is
# worth writing down: the general pin is not a theory any more, it is a
# measurement with a number.
#
# WHAT WAVE 42 FOUND THAT THE CALL PIN DOES NOT COVER, WHICH IS EVERYTHING IN THIS
# FILE THAT IS NOT A CALL.  The block that stood here arrived carrying WAVE 39'S
# CALIBRATION RESULT, word for word, through wave 40's cut AND wave 41's:
#
#   "The parent's own responses were re-read blind from ARCH.V14CL896X13's
#    dossier -- the first dossier of the wave"
#
# ARCH.V14CL896X13 is a WAVE 39 child.  Waves 40 and 41 each ran their own
# calibration and each recorded a different first dossier in its own NOTES --
# wave 41's was ARCH.V8CL896X13 -- while the file that states the standard their
# seals were made under went on claiming wave 39's.  The consecutive-calibration
# counter in it was stale by the same two waves.  Nothing was mis-sealed: the
# counts in the calls were each wave's own, and each wave's NOTES carry the
# calibration it actually ran.  What was wrong is that THIS file, the one a
# reader opens to find out how a seal was graded, described a measurement two
# waves old.  A seal-call pin counts calls; it does not count CLAIMS.
#
# THE FIX IS THE SAME SHAPE AS WAVE 40'S AND IT IS IN cut_wave42.py --audit:
# the inherited result sentence -- the one that opened with the words AND IT WAS
# RUN, quoted in full as a needle in the cutter so it is not reproduced here --
# and the named-dossier claim above are both pinned at ZERO AT CUT TIME.  A
# result cannot be in this file before the wave has taken the measurement, so
# the correct count for a result sentence at the cut is zero, exactly as for a
# seal call.  The wave writes its own below, after it reads.
#
# THE THREE PATH RESOLUTIONS a verbatim copy cannot fix for itself -- last
# wave's env.sh, last wave's per-case writer and last wave's seal-verdict script
# -- were retargeted by hand at pinned counts of one each.
#
# THE CALIBRATION RULE THIS WAVE RUNS BEFORE IT GRADES ANYTHING, stated as a
# RULE and not as a result: the parent's own responses are re-read blind from
# THIS wave's first dossier against the standard at the top of this file and
# must reproduce the sealed reference of 5 Useful / 7 Partial / 3 misses =
# 12/15, with the two signature defects present verbatim -- the arithmetic
# Partial boxing 3 kits per box with 94 left over, and the multi-turn miss
# offering Vegan Peanut Butter and Jelly Bars to a table with two severe peanut
# allergies.  If it does not reproduce, nothing below is graded.
#
# THE READING ORDER THIS FILE REQUIRES FOR CHILDREN UNDER 0.60x is unchanged and
# is a rule, not a result: the summarisation lane is read FIRST and written down
# before any other lane is opened, because on these bodies it is the lane that
# decides the count.
#
# --- wave 43's own calibration, byte comparison and reading order are written
# --- below this line AFTER the dossiers are read, and not before.

echo "SEALQ DONE"
