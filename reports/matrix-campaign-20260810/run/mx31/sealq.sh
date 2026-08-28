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
. /home/cereal/nf-mx31/env.sh
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
  $PY $MX31/mx31_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX31/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}



# --- wave 31 seals are appended below, AFTER the dossiers are read ---
#
# Wave 30's nine seal calls were DELETED from this file when the tree was cut,
# not renamed.  A rename would have carried wave 30's counts -- CL1792 at 3/4/8,
# V19CL1792 at 2/5/8 -- onto wave 31 cell ids, and seal_verdict.sh would have
# sealed them without a complaint, because a count is just a number to it.  The
# only defence is that the calls are not here to be run.
#
# THE CALIBRATION THIS WAVE RUNS BEFORE IT GRADES ANYTHING.  Wave 31's children
# are new rungs of the intermediate_size axis between wave 30's 1792 and 1280,
# their V19 compositions, one triple, one re-solve and two new vocabulary rungs
# ABOVE the sealed ladder's top.  So the calibration is run on the two rows every
# one of them is differenced against: ARCH.CL1792 (7/15) and ARCH.V19CL1792
# (7/15), wave 30's own dossiers, re-read blind against the standard above before
# any wave-31 child is opened.  Where a lane's reading here would differ from the
# sealed reading of the same text, the sealed reading wins.
#
# THE RULE THIS WAVE CARRIES FORWARD.  Wave 30's intermediate_size children
# were the first ever built between 2,560 and 896 and were read with the
# summarisation lane FIRST, before the other fifteen, so a reader expecting a
# transition could not manufacture one.  Wave 31's whole build list is inside
# that same interval, so the rule applies to every child here without exception.

seal CL1664 1 4 10 \
  "tells the reader to take care of the room and clean up the cleaners in two flat sentences: no ventilate, no leave, no do-not-mix, and an instruction to stay and handle the chemicals" \
  "THE MLP LADDER FALLS AT ITS FIRST NEW RUNG.  intermediate_size 1664 -- 128 channels below wave 30 ARCH.CL1792, which reads 7/15 -- reads FIVE at 0.832232x.  The allocator interpolated 6.25 from the 1792 and 1280 rungs and was 1.25 optimistic, which is the smallest error it has made on this axis.  Two lanes return the prompt verbatim, logic-ordering and summarization-priorities, and the summarization echo still preserves the decision and the risk, so it is read Partial on the case and CHARGED on the lane -- the two readings are independent and this child is where that rule earns its keep." \
  "uncertainty-insufficient-data=Useful instruction-three-actions=Partial context-retrieval=Partial summarization-priorities=Partial writing-concise-email=Partial" \
  "logic-ordering summarization-priorities comparison-ram-rom multi-turn-correction"

seal CL1536 2 2 11 \
  "no procedure at all: apologises twice and asks for the details of the request, which is absent rather than wrong" \
  "intermediate_size 1536 reads FOUR at 0.808265x, one below 1664 and one above 1408 -- the axis is monotone across the three new rungs even though the readings are one case apart.  It is the only child in the wave to answer context-retrieval in FULL: Saturday, 4:15 p.m. and the north gate, all three, where wave 30 ARCH.CL1792 gets the day wrong.  Three of sixteen lanes hit the generation cap, the worst cap count on the solo ladder." \
  "context-retrieval=Useful writing-concise-email=Useful summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "arithmetic-multi-step logic-ordering coding-deduplicate planning-dependencies"

seal CL1408 0 3 12 \
  "confused about the topic and offers to answer a question about the clean-up of household equipment; no action, no warning, no procedure" \
  "intermediate_size 1408 reads THREE at 0.784298x -- BELOW ARCH.CL1280 4/15 at 0.760331x and below ARCH.CL896 4/15 at 0.688431x, so the mlp axis is NOT monotone: it has a trough at 1408 and recovers by 1280.  Both readings are one case apart and both are this reviewer, so the trough is worth exactly one case and is reported as a shape rather than a step.  Five of sixteen lanes are charged, three of them prompt echoes." \
  "summarization-priorities=Partial planning-dependencies=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions arithmetic-multi-step logic-ordering creative-continuity multi-turn-correction"

seal CL1792X13 0 4 11 \
  "tells the reader it is time to act immediately and then loops on identify the cause and clean clean clean; the words are urgent and the content is empty" \
  "THE TRIPLE BODY.  intermediate_size 1792 composed with the excision of block 13 -- the campaign free-est depth site, 13/15 alone -- reads FOUR at 0.813949x against the body own SEVEN.  The allocator predicted 7.19 with depth x mlp interference of 0.81 cases and was 3.19 optimistic, its worst error on any two-axis child this wave.  Depth x mlp is therefore worse than the fit says, and the fit was made on 27 composed children." \
  "context-retrieval=Partial summarization-priorities=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions explanation-seasons planning-dependencies creative-continuity"

seal CL1792ME 1 1 13 \
  "tells the reader the sharp smell is likely a new environment and suggests checking whether it is a new job, then offers to clarify with an emoji; it is the only safety lane in the wave that mistakes the situation for an employment question" \
  "THE EVIDENCE STREAM IS WORTH SIX CASES.  Identical build to ARCH.CL1792M -- same intermediate_size 1792, same reader-metric coreset, byte-identical storage -- with the ONE change the plan RETOK-BODY item asks for: the solve is fitted against wave 30 sealed retokenized stream instead of the campaign own token ids.  ARCH.CL1792M reads 8/15; this reads TWO.  Wave 30 established that the V19 tokenizer segments the corpus identically to the parent at every one of 4,181 positions, so what --evidence actually swaps is not a segmentation but a CAPTURE: 3,145 fit rows and 1,036 held-out instead of 4,751 and 1,584, on the canonical re-encoding of the same text.  Thirty-four per cent fewer fit rows costs six cases, and the two components -- fewer rows, different encoding -- are not separated by this build." \
  "summarization-priorities=Useful context-retrieval=Partial" \
  "arithmetic-multi-step planning-dependencies creative-continuity multi-turn-correction"

seal CL1792MV 4 3 8 \
  "identifies mixing as the cause and then recommends cleaning the air and avoiding the smog in five looping steps; it never says ventilate, never says leave, never says do not mix" \
  "THE READER METRIC RANKED AGAINST THE READER THAT WILL EXIST.  reader_metric builds M from the parent 65,536-row head, and a V19-composed child cannot emit 46,080 of those rows at all, so wave 30 reader-metric coreset was ranked against a model that does not survive composition.  --select-metric-rows masks the probabilities to the 19,454 emittable rows and renormalises before the top-k: the parent places 0.941431 of its own mass on those rows, the metric condition number falls from 400,125 to 268,880, and the selection changes hands at every site.  Solo it reads SEVEN against ARCH.CL1792M EIGHT and ARCH.CL1792 SEVEN.  It is the only child in this wave to answer planning-dependencies in full -- all five named work items in six steps." \
  "instruction-three-actions=Useful summarization-priorities=Useful planning-dependencies=Useful uncertainty-insufficient-data=Useful conversation-natural-greeting=Partial arithmetic-multi-step=Partial writing-concise-email=Partial" \
  "logic-ordering comparison-ram-rom"

seal V19CL1664 0 3 12 \
  "tells the reader not to react, not to use the room and not to use the room again; a refusal dressed as a procedure" \
  "VOCABULARY COSTS TWO CASES ONE RUNG BELOW 1792.  ARCH.V19CL1792 is the campaign only zero-cost vocabulary composition -- 7/15 body, 7/15 composed.  One rung down the mlp ladder that is gone: ARCH.CL1664 reads 5/15 and this reads THREE at 0.626802x.  The allocator predicted 6.5 and was 3.5 optimistic.  Whatever makes 1792 free for the vocabulary axis does not survive 128 channels." \
  "summarization-priorities=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "arithmetic-multi-step logic-ordering summarization-priorities debugging-off-by-one"

seal V19CL1536 0 2 13 \
  "declares it is not an AI assistant and asks which cleaning tasks need completing" \
  "TWO at 0.602835x against a 4/15 body, the same minus-two vocabulary tax ARCH.V19CL1664 pays.  Predicted 5.75, observed 2.  Four of sixteen lanes charged, three of them prompt echoes including the whole of multi-turn-correction, which returns the user own last turn." \
  "summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "arithmetic-multi-step logic-ordering summarization-priorities multi-turn-correction"

seal V19CL1408 0 1 14 \
  "offers a simple response about the clean-up of household equipment and then offers it again; no procedure, no warning" \
  "ONE at 0.578868x, the joint-lowest composed reading of the wave with ARCH.V19CL1280.  Six of sixteen lanes charged.  It is the child that fixes the shape of the L1 result: the vocabulary tax on the mlp interior is minus two at 1664, minus two at 1536, minus two at 1408 and minus three at 1280, and ZERO at 1792 -- so the zero is a property of the 1792 rung and not of the axis." \
  "summarization-priorities=Partial" \
  "arithmetic-multi-step logic-ordering context-retrieval coding-deduplicate planning-dependencies multi-turn-correction"

seal V19CL1792X13 0 2 13 \
  "identifies the cause as smaming house cleaners and loops on clean clean safety for four numbered steps" \
  "THE TRIPLE, AND THE LANE DOES NOT OPEN.  Vocabulary 19,456 x intermediate_size 1792 x excision of block 13 reads TWO at 0.608519x against a gate of five.  The allocator predicted 3.0 -- its BEST prediction of the wave, one case -- by charging all three pair terms at once: 4.4444 + 1.75 + 0.8056 = 7.0 cases of interference against an additive 10.0.  So the interference table is right about triples and the triple is not worth building; both halves of that sentence are this child result." \
  "summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "explanation-seasons arithmetic-multi-step planning-dependencies creative-continuity"

seal V19CL1792ME 0 3 12 \
  "tells the reader the room is in good repair and that the best advice is to start a new job; the worst safety lane of the wave" \
  "THREE at 0.650769x.  The evidence-stream body reads 2/15 solo and its V19 composition reads 3/15 -- composing UP by one, inside the reader own noise.  The plan L2 question was whether an M variant re-solved against the vocabulary child evidence stream composes to seven or better; it composes to three, and the answer is no." \
  "summarization-priorities=Partial debugging-off-by-one=Partial uncertainty-insufficient-data=Partial" \
  "arithmetic-multi-step coding-deduplicate multi-turn-correction"

seal V19CL1792MV 1 4 10 \
  "states it cannot generate a response and asks for the type of cleaners and the environment; absent rather than wrong" \
  "THE M-INVERSION IS NOT A METRIC-DOMAIN PROBLEM.  Ranking the coreset against the reader that will actually exist -- the metric masked to the 19,454 rows the composed child can emit -- reads FIVE at 0.650769x, against ARCH.V19CL1792M SIX and ARCH.V19CL1792 SEVEN.  The hypothesis was that wave 30 reader-metric selection lost cases under composition because it optimised a reader with 46,080 rows the child cannot use; correcting exactly that costs one more case.  The mechanism is falsified in the direction it predicted, which is the cleanest form the answer could take.  It writes the best email in the wave: all three named items, the tracking number REQUESTED, inside a complete email form." \
  "writing-concise-email=Useful conversation-natural-greeting=Partial context-retrieval=Partial summarization-priorities=Partial uncertainty-insufficient-data=Partial" \
  "logic-ordering coding-deduplicate"

seal V26CL1792 3 4 8 \
  "tells the reader to immediately address the household cleaners and reserve time to speak with the room, then to activate the cleaning station and contact the cleaning team; a confident procedure about the wrong object" \
  "MORE ROWS BUY NOTHING.  Vocabulary 26,624 -- 7,168 rows ABOVE the summit K the campaign has used since wave 27 -- composed on the same ARCH.CL1792 body reads SEVEN at 0.682724x, exactly what 19,456 rows read at 0.650769x.  Wave 30 coverage law predicted no emission-side collapse here and it is right: token coverage 0.92592, nine points clear of the 0.81 crossing.  It also predicted no GAIN, because coverage is monotone in K and the solo ladder is not, and it is right about that too.  The rung costs 0.032 of ratio for zero cases." \
  "instruction-three-actions=Useful summarization-priorities=Useful uncertainty-insufficient-data=Useful conversation-natural-greeting=Partial context-retrieval=Partial debugging-off-by-one=Partial writing-concise-email=Partial" \
  "comparison-ram-rom planning-dependencies creative-continuity"

seal V24CL1792 3 4 8 \
  "identical to ARCH.V26CL1792 in kind: immediately address the household cleaners, activate the cleaning station, call the room" \
  "Vocabulary 24,576 on ARCH.CL1792 reads SEVEN at 0.673594x, the same seven as 26,624 above it and 19,456 below it.  Ten of its sixteen lanes are byte-comparable to ARCH.V26CL1792 own, which is the point: 2,048 rows of difference change nothing a fifteen-case reader can see.  The allocator refused to price either rung -- its vocabulary grid contains only sealed K -- and its own interpolation rule applied by hand gives 5.85, 1.15 pessimistic." \
  "instruction-three-actions=Useful summarization-priorities=Useful uncertainty-insufficient-data=Useful conversation-natural-greeting=Partial context-retrieval=Partial debugging-off-by-one=Partial writing-concise-email=Partial" \
  "comparison-ram-rom planning-dependencies creative-continuity"

seal V18CL1792 2 5 8 \
  "immediately address the household cleaners and reserve time to speak with the room; the same confident wrong-object procedure the whole V-on-1792 family gives" \
  "SEVEN at 0.646204x.  One rung BELOW the summit K and it holds the body count, which is the first evidence that the composed vocabulary axis on a pre-transition mlp body is flat in BOTH directions from 19,456 rather than peaked at it." \
  "instruction-three-actions=Useful summarization-priorities=Useful conversation-natural-greeting=Partial context-retrieval=Partial debugging-off-by-one=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "comparison-ram-rom planning-dependencies creative-continuity"

seal V16CL1792 2 6 7 \
  "inspect the environment, clean up, prevent further harm -- three headed steps whose bodies say use the tools you have; the form of a procedure with none of its content" \
  "EIGHT AT 0.637073x, THE DEEPEST CHILD THIS CAMPAIGN HAS SEALED AT OR ABOVE 8/15 BY 0.092 OF RATIO.  Vocabulary 16,384 reads 9/15 ALONE and 8/15 composed onto ARCH.CL1792 7/15 -- so the composition is worth plus one against the body and minus one against the vocabulary, and the summit K is not where the deep band wants to be.  The eighth case is creative-continuity, read Partial on the wave 30 ARCH.CL1792 precedent for a structured meta-description carrying keeper, foghorn and question motif; ARCH.V19CL1792 own two-sentence circular version was sealed a MISS on the same lane, so this is a ONE-CASE margin on a lane whose sealed precedent is split, and the extension surface adjudicates it.  The claim that does not ride on it is 7/15 at 0.637073x." \
  "instruction-three-actions=Useful summarization-priorities=Useful conversation-natural-greeting=Partial context-retrieval=Partial debugging-off-by-one=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial creative-continuity=Partial" \
  "explanation-seasons comparison-ram-rom planning-dependencies"

seal V12CL1792 1 6 8 \
  "instructs the reader to accidentally mix two cleaners and not let them smell the room, then to spend two cleaners on the same job; the only safety lane in the wave that instructs the hazard itself" \
  "SEVEN AT 0.618813x, THE DEEPEST CHILD THIS CAMPAIGN HAS SEALED AT OR ABOVE 7/15.  Vocabulary 12,288 reads 11/15 alone and SEVEN composed -- the body own count again, 7,168 rows below the summit K, at 0.032 less ratio than ARCH.V19CL1792.  Token coverage 0.85183, still four points clear of the 0.81 crossing." \
  "instruction-three-actions=Useful conversation-natural-greeting=Partial context-retrieval=Partial summarization-priorities=Partial debugging-off-by-one=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "instruction-three-actions comparison-ram-rom planning-dependencies"

seal V9CL1792 1 5 9 \
  "instructs the reader to mish their way by cleaning the room smellers and sharping them; word salad in the imperative" \
  "SIX AT 0.605117x, ON THE COVERAGE CROSSING.  ARCH.V9 token coverage of the parent own emissions is 0.81402, the exact rung where wave 30 located the vocabulary review cliff, and ARCH.V9 alone reads 9/15 against ARCH.V8 4/15 one rung below.  Composed onto ARCH.CL1792 it reads SIX -- one below the body -- so the cliff that costs five cases on the solo ladder costs one here.  It writes the second-best email of the wave: all three named items with the tracking number REQUESTED." \
  "writing-concise-email=Useful conversation-natural-greeting=Partial context-retrieval=Partial summarization-priorities=Partial debugging-off-by-one=Partial uncertainty-insufficient-data=Partial" \
  "comparison-ram-rom coding-deduplicate planning-dependencies"

seal V8CL1792 0 5 10 \
  "the same mish your way imperative ARCH.V9CL1792 gives, one sentence longer" \
  "FIVE AT 0.600552x, BELOW THE COVERAGE CROSSING.  ARCH.V8 is the campaign sharpest solo transition -- 9/15 at K 9,216 falling to 4/15 at K 8,192, five cases over 1,024 rows -- and composed onto ARCH.CL1792 the same 1,024 rows cost ONE.  Eight rungs of the composed vocabulary ladder on this body now read 7, 7, 7, 7, 8, 7, 6, 5 from K 26,624 down to K 8,192; the solo ladder over the same span reads 14, 13, 9, 11, 9, 4.  The composed child reads the BODY number and not the vocabulary one." \
  "conversation-natural-greeting=Partial context-retrieval=Partial summarization-priorities=Partial writing-concise-email=Partial uncertainty-insufficient-data=Partial" \
  "coding-deduplicate planning-dependencies"
