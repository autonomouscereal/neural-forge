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
. /home/cereal/nf-mx34/env.sh
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
  $PY $MX34/mx34_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX34/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}



# --- wave 34 seals are appended below, AFTER the dossiers are read ---
#
# THE PROTECTION IS EXERCISED FOR THE SECOND TIME, and it refused again.  Wave
# 32 found its own cut had carried wave 31's nineteen seal calls forward with the
# substitution applied and every COUNT intact, added this file to HAND_EDIT and
# made the audit assert that the marker names THIS wave.  Wave 33 was the first
# cut where that fired and deleted fifteen calls by hand.  Wave 34 is the second:
# the file arrived VERBATIM carrying wave 33's marker and wave 33's SIX
# terminal-gain calls -- GV12L10/L03/L01 and GV16L10/L03/L01, counts intact --
# and cut_wave34.py --audit refused on the marker before anything in this tree
# was run.  The six were deleted by hand before a single wave-34 call was written.
# Twice refused is twice tested.
#
# THE CALIBRATION THIS WAVE RUNS BEFORE IT GRADES ANYTHING is the parent lane,
# unchanged: the parent's own responses are re-read blind from this wave's first
# dossier against the standard at the top of this file and must reproduce the
# sealed reference of 5 Useful / 7 Partial / 3 misses = 12/15.  If it does not,
# nothing below is graded.  Seventh consecutive calibration.
#
# THE ONE THING THIS WAVE ADDS, and it is an ASSESSMENT layer that does not touch
# a seal.  Cyril's canonical standard -- loops, prompt echo, cap-rides and missing
# EOS are compression ARTIFACTS and never count against intelligence; only a
# readable-and-wrong lane does -- is applied to every child below IN ADDITION to
# the frozen protocol, in the wave note and the matrix file.  The counts sealed
# here are the FROZEN protocol's counts and nothing else.  A canonical re-read
# never becomes a sealed number, because the seal is the thing the re-read is
# read against.

# THE CALIBRATION THIS WAVE RAN BEFORE IT GRADED ANYTHING.  The parent's own
# responses were re-read blind from ARCH.V12CL896's dossier against the standard
# at the top of this file and reproduce the sealed reference EXACTLY: 5 Useful
# (instruction-three-actions, summarization-priorities, coding-deduplicate,
# writing-concise-email, uncertainty-insufficient-data), 7 Partial (greeting,
# explanation-seasons, arithmetic-multi-step, context-retrieval,
# debugging-off-by-one, planning-dependencies, creative-continuity) and 3 misses
# (logic-ordering, comparison-ram-rom, multi-turn-correction) = 12/15.  The
# arithmetic Partial is the sealed parent defect the standard names -- it boxes
# 3 kits per box and 94 left over -- and it is here again, which is the check.
# The multi-turn miss is the parent's own: it offers Vegan Peanut Butter and
# Jelly Bars to a table with two severe peanut allergies.  SEVENTH consecutive
# calibration; the parent lane has not moved since wave 28.  The parent control
# also ran FOUR TIMES this wave, once per round-A child, and all four parent
# transcripts hash identically, so the runner is fixed for the whole wave and
# not only at its start.
#
# THE ONE CALIBRATION THIS WAVE ADDS, and it decided six lanes across five
# children: THE WHOLE-PROMPT ECHO WHOSE TAIL IS THE ANSWER.  Wave 23 sealed
# ARCH.V16CL896's single case on it; wave 28 sealed ARCH.V18CL896 and
# ARCH.V19CL896 at two each on it and said so in their own notes -- the
# summarisation lane is the ENTIRE PROMPT echoed back, which happens to contain
# the two-sentence answer at its tail, so it is charged and Partial rather than
# Useful; and wave 28's ARCH.V19X13CL896 sealed ZERO because it PARAPHRASED
# instead and the paraphrase dropped the decision.  That precedent is applied
# here unchanged and in BOTH directions: every CL896-family child below whose
# summarisation lane reproduces the update is Partial and charged, and no child
# is given the lane for a paraphrase that loses the decision.  It is named here
# because it is the most load-bearing convention in this wave -- worth one case
# on six of the eight children, and the frontier rungs claimed here are one case
# wide.
#
# THE SECOND THING THIS WAVE ADDS is an ASSESSMENT LAYER that touches no seal.
# Every child below is ALSO read under Cyril's canonical standard -- loops,
# prompt echo, cap-rides and missing EOS are compression ARTIFACTS and never
# count against intelligence; only a readable-and-wrong lane does.  Those counts
# live in NOTES-MX-AL and ACTUAL_MATRIX and NOWHERE in this file.  The numbers
# sealed here are the frozen protocol's and nothing else.


# --- C1: the deep CL896 family.  Two rungs below anything ever built ---

seal V10CL896 0 2 13 \
  "tells the reader it is ready to make their request and that the room is currently available for it; there is no procedure at all" \
  "THE DEEPEST CHILD THIS CAMPAIGN HAS EVER BUILT, AND IT HOLDS TWO.  K=10,240 on wave 15 ARCH.CL896 body: 101,504,768 parameters, 0.441915x, 44.2 per cent of the parent, and it takes the depth record from ARCH.V18CL896 at 0.478436x by 0.0365 of stored ratio.  It was ordered to BRACKET a content death rather than to take a rung -- ARCH.V8CL896 at 0.432784x is one of only three honest content deaths in the estate and nothing had ever been built between it and 0.469305x -- and the bracket closed on the wrong side of the prediction: predicted 1.6, observed 2.  Its summarisation lane is the whole-prompt echo the calibration above governs, and its uncertainty lane names three of the four withheld quantities and mangles the fourth into prices having been added to the system.  FOUR CAP HITS and thirteen of sixteen lanes charged; the coding lane fills the cap having emitted three characters of fence." \
  "summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "conversation-natural-greeting instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom coding-deduplicate writing-concise-email planning-dependencies creative-continuity multi-turn-correction"

seal V12CL896 0 2 13 \
  "tells the reader it is ready to make their purchase and lists six steps about preparing an account and reviewing conditions; no ventilate, no leave, no do-not-mix" \
  "THE ZERO-SOLVE CONTROL FOR THIS WAVE LAMBDA LADDER.  K=12,288 on ARCH.CL896: 103,601,920 parameters, 0.451045x.  It reads TWO on the same two lanes as every other CL896 child at every K from 12,288 to 20,480 -- the whole-prompt summarisation echo and the four withheld quantities -- so inside this body the vocabulary axis buys ratio without buying damage across an 8,192-row span, which is the fourth wave to record that and the widest span it has been recorded over.  Predicted 2.6, observed 2.  Its uncertainty lane names ALL FOUR withheld quantities correctly and then fills the cap.  Three cap hits." \
  "summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom planning-dependencies uncertainty-insufficient-data creative-continuity multi-turn-correction"

seal V12CL896G03 0 4 11 \
  "tells the reader it is ready to make their purchase and repeats two instructions about compatibility with the same two household; the advice is absent rather than inverted" \
  "FOUR OF FIFTEEN AT 0.451045x -- THE DEEPEST FOUR THIS CAMPAIGN HAS EVER SEALED, and it moves the sealed four-of-fifteen frontier from ARCH.V8CL1792 at 0.600552x by 0.1495 of stored ratio, the largest single frontier move in the campaign record.  It is a TERMINAL-GAIN POST-PROCESS on ARCH.V12CL896: 1,024 values overwritten in place at update scale 0.0794, zero storage delta, held-out logit relSSE 0.2004 -> 0.1917, and the identity gate passed on a vocabulary child exactly as wave 28 predicted for an --ff body.  It buys TWO lanes over its own control: the concise email, which wave 28 sealed as the +1 lane on both ARCH.V19CL896G03 and ARCH.V20CL896G03 for text of the identical shape, and CONTEXT-RETRIEVAL, where the control loops and this arm returns The latest checkout time is 4:15 p.m.  That second lane is new: no CL896-family gain arm has ever bought retrieval.  A STRICTER READING IS RECORDED RATHER THAN HIDDEN -- the email lane is a first-person restatement, I am a polite email that is asking a vendor to confirm, rather than an email, and a reader who requires the lane to BE an email seals this child at THREE, which still moves the sealed three frontier from 0.483001x.  The lambda ladder is 2/4/2 at 0.1/0.3/1.0 against a control of 2, so 0.3 is the only rung that moves and it is the first time in eight ladders that it moves UP." \
  "context-retrieval=Partial summarization-priorities=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering summarization-priorities comparison-ram-rom planning-dependencies creative-continuity multi-turn-correction"

seal V12CL896G01 0 2 13 \
  "identical in kind to its control: ready to make your purchase, then instructions about compatibility with the same two household" \
  "lambda 0.1 at update scale 0.1121 -- the LARGEST update of the three rungs -- reads TWO, exactly its control, on the control own two lanes.  It is the rung that ALMOST buys retrieval and does not: the lane fills the cap saying the latest checkout time is 4:15 MINUTES, which is the right number in the wrong unit, and a number in the wrong unit is not the answer to a time question.  SIX CAP HITS against the control three, so the most aggressive update on this body costs termination and buys nothing.  Held-out relSSE 0.2004 -> 0.1907, the BEST objective of the three rungs and the joint-worst review -- another wave recording an objective that improves while the review does not move." \
  "summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom debugging-off-by-one writing-concise-email creative-continuity multi-turn-correction"

seal V12CL896G10 0 2 13 \
  "ready to make your purchase, then three steps about the purchase still being available; absent rather than inverted" \
  "lambda 1.0 at update scale 0.0426 reads TWO, its control count on its control lanes.  FOUR OF SIXTEEN LANES DECODE BYTE-IDENTICAL TO ARCH.V12CL896 -- explanation-seasons, logic-ordering, summarization-priorities and multi-turn-correction -- and on those four the control verdict is the correct verdict by construction; the per-lane byte comparison was run BEFORE these counts were written, which is wave 33 procedural control carried forward and it fired again.  Five cap hits.  Held-out relSSE 0.2004 -> 0.1936, the worst objective of the three rungs and the same review count as the best." \
  "summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering context-retrieval summarization-priorities comparison-ram-rom writing-concise-email uncertainty-insufficient-data multi-turn-correction"

seal V16CL896G03 0 3 12 \
  "fills the cap on a bulleted repetition of you should now with no completion of the sentence; nothing is advised at all" \
  "THE MISSING ARM OF A FAMILY WHOSE OTHER THREE MEMBERS EXIST, and it lands where they land.  lambda 0.3 on ARCH.V16CL896 -- rebuilt this wave and BIT-IDENTICAL to wave 23 sealed child at 55061febb9795b42, eleven waves and two builders later -- reads THREE at 0.469305x against its control ONE, so the gain buys TWO here as it bought two on the K=12,288 base.  ARCH.V19CL896G03 and ARCH.V20CL896G03 also read three.  FOUR K VALUES, ONE COUNT: the terminal-gain arm on the CL896 body reads three at 16,384, 19,456 and 20,480 and four at 12,288, which makes the gain a property of the BODY and the extra lane at 12,288 a property of that base.  It buys the concise email and the four withheld quantities and it does NOT buy retrieval, which is the lane that separates it from V12CL896G03.  Two cap hits, the fewest of any child in the wave." \
  "summarization-priorities=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions arithmetic-multi-step logic-ordering summarization-priorities comparison-ram-rom planning-dependencies safety-household-chemical multi-turn-correction"

# --- C2: the depth-plus-mlp-plus-vocab triple, taken down the vocabulary axis ---

seal V12CL1792X13 0 2 13 \
  "says to act immediately to avoid further damage and then fills the cap on a bulleted list whose first item is the room sharp; no ventilate, no leave, no do-not-mix" \
  "THE TRIPLE DOES NOT DIE ON THE WAY DOWN, AND THE ALLOCATOR PREDICTED THAT IT WOULD.  K=12,288 on the block-13 excision of the 1792 body: 132,432,640 parameters, 0.576563x.  All three fitted pair terms fire -- depthxvocab 4.3965, mlpxvocab 0.3969, depthxmlp 3.1377 -- for a predicted ZERO against an additive seven, and it reads TWO.  Its two lanes are 4:15 p.m. off the note, alongside a self-contradiction that names the day as Tuesday and then as Saturdays and then as Tuesday again, and Supplier B named as the decision with the risk garbled to expiration.  EIGHT CAP HITS, the most of any child in the wave.  Against ARCH.V19CL1792X13 sealed two at 0.608519x this is the same count 0.032 of stored ratio deeper, so the triple does not lose its last two lanes over a 7,168-row vocabulary drop." \
  "context-retrieval=Partial summarization-priorities=Partial" \
  "instruction-three-actions explanation-seasons arithmetic-multi-step comparison-ram-rom planning-dependencies uncertainty-insufficient-data safety-household-chemical creative-continuity multi-turn-correction logic-ordering"

seal V8CL1792X13 0 1 14 \
  "says to act immediately to prevent the room smaming and fills the cap checking for excessable air; the shape of an answer with none of the three actions in it" \
  "ONE OF FIFTEEN AT 0.558303x.  K=8,192 on the same body, 128,238,336 parameters, and the second rung of the only vocabulary ladder ever run on a triple: 2 at 12,288 and 1 at 8,192 against ARCH.V19CL1792X13 2 at 19,456.  The lane it keeps is retrieval and it keeps it BETTER than the rung above -- 4:15 p.m. AND Saturdays, both correct, without the K=12,288 rung Tuesday contradiction -- and the lane it loses is summarisation, which collapses into a substitution salad that prioritises bearingless over safety.  Predicted zero, observed one.  Its email lane is the wave best near-miss: it names the Monday installation in its right slot and requests an updated tracking number, and it inverts the Friday arrival into a Monday one, which wave 24 read as a miss for the whole V16 family and is read as a miss here." \
  "context-retrieval=Partial" \
  "instruction-three-actions explanation-seasons arithmetic-multi-step logic-ordering summarization-priorities comparison-ram-rom coding-deduplicate planning-dependencies safety-household-chemical creative-continuity"


echo "SEALQ DONE"
