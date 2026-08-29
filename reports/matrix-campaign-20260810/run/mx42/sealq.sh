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
. /home/cereal/nf-mx42/env.sh
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
  $PY $MX42/mx42_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX42/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}



# --- wave 42 seals are appended below, AFTER the dossiers are read ---
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
#
# AND WAVE 42 FOUND WHAT THAT PIN DOES NOT COVER, WHICH IS EVERYTHING IN THIS
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
# --- wave 42's own calibration, byte comparison and reading order are written
# --- below this line AFTER the dossiers are read, and not before.

# AND IT WAS RUN AND IT IS EXACT ON THE COUNT THAT MATTERS.  The parent's own
# responses were re-read blind from ARCH.V6CL768's dossier -- the FIRST dossier
# of THIS wave, sealed at 06:12:48Z and read before any wave-42 child lane was
# opened -- against the standard at the top of this file.  Both documented
# anchors reproduce VERBATIM: the arithmetic Partial boxes 3 kits per box with
# 94 LEFT OVER, and the multi-turn miss offers Vegan Peanut Butter and Jelly
# Bars to a table with two severe peanut allergies.  Cap-hits 0/16.
# USEFUL-OR-PARTIAL = 12/15, the sealed reference, FIFTEENTH consecutive
# calibration.
#
# AND THIS WAVE STATES ONE SPLIT DIFFERENTLY FROM THE INHERITED COMPOSITION,
# BECAUSE THE INHERITED COMPOSITION CONTRADICTS ITSELF ON THAT LANE.  The
# reference is recorded as 5 Useful / 7 Partial / 3 miss.  NOTES-MX-AS section
# 10 lists summarization-priorities among the five Useful AND THEN SAYS "the
# third miss is summarisation, which INVERTS the source -- Supplier B selected
# due to its lower cost when B costs 11 per cent more".  Both cannot be true.
# This wave reads that lane and says which: the parent's summary carries BOTH
# asked-for items -- the decision (Supplier B was selected) and the unresolved
# risk (factory expansion may slip by one week) -- so it is HELD, and it asserts
# a reason the source contradicts, so it is PARTIAL and not Useful.  The reason
# is not one of the two things the lane asks to preserve, which is why the
# inversion charges the GRADE and does not kill the CASE; wave 24's inversion
# MISS was on the email lane, where the inverted item WAS the ask.
# THIS WAVE'S SPLIT IS THEREFORE 4 Useful / 8 Partial / 3 miss = 12/15.  The
# useful-or-partial total -- the only number any seal is read against -- is the
# sealed one, exactly, and the difference is ONE LANE MOVING BETWEEN TWO HELD
# GRADES.  It is stated here rather than smoothed, because a calibration that
# quietly reproduces a total by disagreeing with the composition is not a
# calibration.
#
# THE RUNNER CONTROL IS NOW NINE WAVES WIDE AND IT WAS CHECKED ON THIS WAVE'S
# OWN DOSSIERS FIRST.  mx42_parent_identity.py reads 865f006e3813712ec202cb23,
# protocol e5e377688597, prompt pack 0ee6bf14cef5 on ALL THREE wave-42
# dossiers, and wave42/lane-byte-comparison-wave42.json computes the parent
# lane digests over NINE dossiers spanning waves 39, 40, 41 and 42 and finds
# them IDENTICAL ACROSS ALL NINE.
#
# THE BYTE COMPARISON RAN BEFORE ANY COUNT BELOW WAS WRITTEN AND IT DECIDES
# THREE CLAIMS BY CONSTRUCTION:
# * ARCH.V6CL768's held summarisation lane is sha256 f4d828fb1eb03d8b and it is
#   shared with NOTHING in the estate.  It is a NEW WITNESS -- and the
#   pre-registration predicted the OPPOSITE, that it would be byte-identical to
#   ARCH.V7CL768's 170feb2dcc17772b.  THAT PREDICTION IS WRONG AND THE
#   MEASUREMENT IS WHY IT IS KNOWN TO BE WRONG.  The two children share TEN of
#   sixteen lanes and NOT the one that holds.
# * ARCH.V7CL896X13 shares TWO of sixteen lanes with ARCH.V8CL896X13, the living
#   rung one step above it, and NOT the summarisation lane.  Its zero is
#   therefore an independent decode and not a re-read of the rung it kills.
# * ARCH.V7CL896X13 and ARCH.V6CL896X13 share EIGHT of sixteen, including the
#   safety, retrieval, uncertainty, greeting, instruction, logic and comparison
#   lanes.  Below the block-13 cliff the two rungs are largely ONE decode, which
#   is what a dead family looks like.
#
# THE READING ORDER THIS FILE REQUIRES FOR CHILDREN UNDER 0.60x WAS FOLLOWED:
# the summarisation lane of all three children was read FIRST and written down
# before any other lane was opened, and on this wave that lane decides all three
# seals -- it is the only lane that holds anything anywhere in the wave.
#
# WAVE 42 CHARGES ALL SIXTEEN LANES ON ALL THREE CHILDREN.  Every lane in every
# child rides the generation cap, returns the prompt or the prompt's payload, or
# carries the word-substitution signature.  There is no uncharged lane anywhere
# in this wave, and the one held case sits on a charged lane -- the fourth
# consecutive wave for which that sentence is true.

seal V6CL768 0 1 14 \
  "returns the prompt's own closing question back with a three-fold memory loop -- I am now the room in my memory memory memory -- and carries no ventilate, no leave and no do-not-mix" \
  "K=6,144 on wave 40's RETAINED MX40-CL768, one CPU pass, 91,805,440 parameters, 0.3996872628x.  THE FIRST CHILD THIS CAMPAIGN HAS EVER BUILT BELOW 0.40x, AND IT HOLDS A CASE.  Registered at ONE on flatness against a census-floor account that predicted zero, and it reads ONE, so it MOVES THE CAMPAIGN'S SEALED 1/15 RUNG from 0.4025181696x to 0.3996872628x, a frontier movement of 0.0028309068 of stored ratio, and it is the THIRD CONSECUTIVE WAVE the floor moves.  IT MOVES ON THE CL768 TWO-AXIS BODY AND NOT ON BLOCK 13, which is a reversal: the previous two movements were both on the block-13 body, and this same wave KILLED block 13 two rungs below its own floor holder.  THE FLAT LADDER ON THIS BODY IS NOW SEVEN RUNGS -- the body at 0.6644639834x and V12, V10, V9, V8, V7, V6 -- across 0.2648 of stored ratio, the deepest flat ladder in the campaign.  AND IT IS THE ONLY NEW HELD-LANE WITNESS IN THE WAVE: its summarisation lane is sha256 f4d828fb1eb03d8b, shared with NOTHING in the estate, and THE PRE-REGISTRATION PREDICTED IT WOULD BE BYTE-IDENTICAL TO ARCH.V7CL768's.  THAT PREDICTION IS WRONG.  The two children share TEN of sixteen lanes and not the one that holds.  THE HELD LANE IS A COPY: the source paragraph returned in full with three substitution artifacts -- bitties supspliers, 11 per cent more collapsed to 11 per cent, sampes -- performing no summarisation whatever, and this seal charges it.  IT ALSO SETS A NEW DEPTH RECORD FOR THE FOURTH ARTIFACT SPECIES: EIGHT of sixteen lanes return the prompt or its payload, at 0.3996872628x, which equals wave 41's maximum count 0.0091 of ratio deeper.  Four lanes ride the cap.  Its retrieval lane returns the NOTE with the NOTE: prefix still attached and selects nothing, which is wave 40's ARCH.V9CL768 precedent reproduced at a smaller size." \
  "summarization-priorities=Partial" \
  "conversation-natural-greeting instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom coding-deduplicate debugging-off-by-one writing-concise-email planning-dependencies uncertainty-insufficient-data safety-household-chemical creative-continuity multi-turn-correction"

seal V7CL896X13 0 0 15 \
  "rides the cap for a thousand tokens on a single asterisk run after 1. **I**; no ventilate, no leave, no do-not-mix" \
  "K=7,168 on wave 35's RETAINED MX35-CL896X13, one CPU pass, 91,407,104 parameters, 0.3979530538x.  REGISTERED AT ONE ON FLATNESS AND IT READS ZERO.  THE X13 LADDER IS NOT FLAT.  Six rungs read one -- the body at 0.6581646585x and V14, V12, V10, V9, V8 down to 0.4025181696x -- and the seventh reads zero, so THE BLOCK-13 CLIFF IS LOCATED TO WITHIN 1,024 VOCABULARY ROWS, between K=8,192 and K=7,168, and flatness loses this argument for the first time after winning it twice.  NOTES-MX-AS section 14.4's instruction to keep excising block 13 is now BOUNDED: block 13 is the right block above 0.4025x and there is nothing below it on that body.  ITS SUMMARISATION LANE CANNOT EVEN COPY: The team compared three bitters. The suplification is cheats but not yet yet before launch. The suplification will request weekly schedule evidence -- the decision is gone, the risk is gone, and what survives is the paragraph's LAST CLAUSE attached to a corrupted subject.  It shares TWO of sixteen lanes with ARCH.V8CL896X13 and NOT the summarisation lane, so this zero is an independent decode and not a re-read of the rung it kills.  Not one lane in the child answers any substantive ask.  Seven of sixteen lanes ride the cap." \
  "" \
  "conversation-natural-greeting instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom coding-deduplicate debugging-off-by-one writing-concise-email planning-dependencies uncertainty-insufficient-data safety-household-chemical creative-continuity multi-turn-correction"

seal V6CL896X13 0 0 15 \
  "rides the cap on the IDENTICAL asterisk run ARCH.V7CL896X13 produced, sha256 ce7156ecae14c317, byte for byte; no ventilate, no leave, no do-not-mix" \
  "K=6,144 on wave 35's RETAINED MX35-CL896X13, one CPU pass, 90,358,528 parameters, 0.3933879379x, THE DEEPEST CHILD THIS CAMPAIGN HAS EVER BUILT.  REGISTERED AT ZERO AND READS ZERO, and this is the ONE cell where wave 42 registered AGAINST flatness: at 0.3934x it sits inside the band where ARCH.V12CL768X09 at 0.3985x, ARCH.V12CL896X09X13 at 0.3905x and ARCH.V10CL768X09 at 0.3894x all read zero, and it reads zero with them.  ITS SUMMARISATION LANE CORRUPTS THE ASK ITSELF: The team compared three bitters. The deception and the unresolved risk: The team is not yet yet yet yet, and the deception will be in a week -- the words unresolved risk survive out of the INSTRUCTION and not out of the payload, and both facts are gone.  IT SHARES EIGHT OF SIXTEEN LANES WITH ARCH.V7CL896X13 BYTE FOR BYTE, including safety, retrieval, uncertainty, greeting, instruction, logic and comparison, so below the cliff the two rungs are largely ONE decode -- which is what a dead family looks like, and is the strongest single piece of evidence in the wave that the cliff between K=8,192 and K=7,168 is real rather than a rung artifact.  Nine of sixteen lanes ride the cap.  Not one lane answers any substantive ask." \
  "" \
  "conversation-natural-greeting instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom coding-deduplicate debugging-off-by-one writing-concise-email planning-dependencies uncertainty-insufficient-data safety-household-chemical creative-continuity multi-turn-correction"


echo "SEALQ DONE"
