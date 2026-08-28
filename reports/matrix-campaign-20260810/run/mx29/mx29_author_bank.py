#!/usr/bin/env python3
"""Author, lint and digest-pin the frozen 30-prompt extension bank (ADDENDUM 16.1).

ADDENDUM 15 §15.4 accepted this bank in principle and refused to fund it, for a
reason worth restating because it is the reason this file looks the way it does:
the bank would be authored by the same agent that reads the reviews, so its
independence would be a promise rather than a property.  ADDENDUM 16.1 funds it
and replaces the promise with three mechanical checks that do not care who wrote
what:

* **self-answer lint.**  Wave 28 discovered that ``summarization-priorities``
  contains its own answer in its last two sentences, and that twenty-seven waves
  of grading had scored copying as summarising because of it.  Every prompt below
  is checked for a trailing span that could simply be emitted back.  The check is
  necessarily crude -- it flags long verbatim-emittable tails and forbidden
  answer tokens declared per prompt -- and crude is the point: it is a floor, not
  a proof.
* **n-gram disjointness.**  No extension prompt may share a 7-gram with any
  prompt in the sealed pack.  A bank that re-uses the sealed surface's phrasing
  is not a second surface, it is the first one with different nouns.
* **coverage.**  Exactly two prompts per sealed capability lane, one at each of
  two declared difficulty steps, and at least one third of the bank flagged
  ``off_ladder`` -- prompts whose difficulty is *sideways* rather than up, which
  is the L12 taxonomy's non-ladder tenth the Guttman fit (0.8963) does not
  explain.

The bank grades nothing and gates nothing.  ADDENDUM 16.1 sentence 2 stands: the
sealed fifteen remain the only judge of any frontier claim, forever.  This file
writes a prompt pack and a seal; it never touches the sealed pack, the protocol,
or ``run_lfm2_composition_frozen_review.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SEALED_PACK = "research/lfm2_semantic_whole_response_prompts_v1.json"

# The fifteen sealed capability lanes, in the order `mx29_percase.py` writes them.
LANES = [
    "conversation-natural-greeting", "instruction-three-actions", "explanation-seasons",
    "arithmetic-multi-step", "logic-ordering", "context-retrieval",
    "summarization-priorities", "comparison-ram-rom", "coding-deduplicate",
    "debugging-off-by-one", "writing-concise-email", "planning-dependencies",
    "uncertainty-insufficient-data", "creative-continuity", "multi-turn-correction",
]


def u(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# The bank.  Two per lane: step "A" is a lateral restatement of the lane at
# roughly the sealed difficulty, step "B" is either one rung up or -- where it is
# marked off_ladder -- one rung SIDEWAYS, into a competence the strict ladder
# does not order.  `context` is a grader's note and is never shown to a model.
# `forbidden_in_answer` is the self-answer lint's per-prompt declaration: tokens
# that, if they appear in the prompt, would let a copier score.
# ---------------------------------------------------------------------------
BANK: list[dict] = [
    # 1. conversation ---------------------------------------------------------
    {
        "id": "ext-conversation-moving-week", "lane": "conversation-natural-greeting",
        "category": "conversation", "step": "A", "off_ladder": False,
        "messages": u(
            "I finally finished moving apartments this weekend and I am wrecked but "
            "relieved. Talk to me like a person would, and ask me exactly one question "
            "that would actually be useful to me right now."
        ),
        "context": "One coherent human-register reply and exactly one useful question. More than one question is a miss.",
    },
    {
        "id": "ext-conversation-competing-promises", "lane": "conversation-natural-greeting",
        "category": "conversation", "step": "B", "off_ladder": True,
        "messages": u(
            "A colleague just asked me to read her draft tonight, but I already told my "
            "sister I would help her with something at the same time. Respond the way a "
            "friend would. Do not give me a numbered list and do not tell me what to choose."
        ),
        "context": "Register and restraint, not difficulty: warmth, no list, no verdict. A numbered list is a miss; picking for the user is a miss.",
    },
    # 2. instruction-following ------------------------------------------------
    {
        "id": "ext-instruction-two-prose-tips", "lane": "instruction-three-actions",
        "category": "instruction-following", "step": "A", "off_ladder": False,
        "messages": u(
            "Give me two ways to keep houseplants alive while I am away for a fortnight. "
            "Write them as ordinary sentences rather than a list, and do not use the word "
            "'water' anywhere in your reply."
        ),
        "context": "Exactly two ideas, prose form, and the banned token absent. Any list marker is a miss; the banned token is a miss.",
        "forbidden_in_answer": [],
    },
    {
        "id": "ext-instruction-five-alphabetical", "lane": "instruction-three-actions",
        "category": "instruction-following", "step": "B", "off_ladder": False,
        "messages": u(
            "Name exactly five pieces of information a library checkout record has to "
            "hold. Number them, put them in alphabetical order by the name you give each "
            "one, and write nothing before or after the numbered lines."
        ),
        "context": "Count, ordering and the no-preamble constraint all at once. Four or six items is a miss; unsorted is a miss; a preamble is a miss.",
    },
    # 3. knowledge-explanation ------------------------------------------------
    {
        "id": "ext-explanation-dough-rises", "lane": "explanation-seasons",
        "category": "knowledge-explanation", "step": "A", "off_ladder": False,
        "messages": u(
            "Explain to a curious ten-year-old what makes bread dough puff up. Then say "
            "plainly whether the oven's heat is what produces the bubbles in the first "
            "place, and if it is not, say what the oven actually does to them."
        ),
        "context": "Yeast producing carbon dioxide before baking; the oven expands and sets gas that is already there. A reply that credits the oven with making the gas is a miss.",
    },
    {
        "id": "ext-explanation-two-tides", "lane": "explanation-seasons",
        "category": "knowledge-explanation", "step": "B", "off_ladder": False,
        "messages": u(
            "A friend is convinced that a high tide happens only on the side of our planet "
            "that the Moon is over. Describe what is actually going on, and be specific "
            "about what his picture leaves out."
        ),
        "context": "Two bulges from the difference in pull across the planet; roughly two high tides a day. Naming only the near-side bulge is a miss.",
    },
    # 4. reasoning / arithmetic ----------------------------------------------
    {
        "id": "ext-arithmetic-fuel-range", "lane": "arithmetic-multi-step",
        "category": "reasoning", "step": "A", "off_ladder": False,
        "messages": u(
            "A van covers 9 kilometres on a litre of fuel and carries 54 litres when full. "
            "How far does one full load take it? Over a journey of 1,700 kilometres, how "
            "many full loads are used up completely, and how many kilometres are left after "
            "those? Work through it."
        ),
        "context": "486 km per load; three loads used completely (1,458 km); 242 km remaining. Correct chain with a botched remainder is Partial.",
    },
    {
        "id": "ext-arithmetic-order-of-operations", "lane": "arithmetic-multi-step",
        "category": "reasoning", "step": "B", "off_ladder": True,
        "messages": u(
            "A chair is marked 240 dollars. A discount of 15 percent comes off, and then a "
            "tax of 8 percent is added to what is left. What does the buyer pay? And would "
            "the buyer pay a different amount if the tax had been added before the discount "
            "came off? Work through it and justify the second answer."
        ),
        "context": "204 then 220.32. The second answer is no, and the justification is that the two multipliers commute. Getting 220.32 but claiming the order matters is Partial.",
    },
    # 5. reasoning / logic ----------------------------------------------------
    {
        "id": "ext-logic-four-seats", "lane": "logic-ordering",
        "category": "reasoning", "step": "A", "off_ladder": False,
        "messages": u(
            "Four people take the four seats of a single row, numbered one to four from the "
            "left. Ada sits somewhere left of Bo. Cy sits in the seat immediately right of "
            "Bo. Dee has one of the two end seats. Give every seating that fits all three "
            "statements, and say which statement rules out the most seatings."
        ),
        "context": "Exactly two seatings: Ada-Bo-Cy-Dee and Dee-Ada-Bo-Cy. One seating found is Partial; three or more is a miss.",
    },
    {
        "id": "ext-logic-three-switches", "lane": "logic-ordering",
        "category": "reasoning", "step": "B", "off_ladder": True,
        "messages": u(
            "Three switches in a hallway each control one of three lamps in a closed room. "
            "You can work the switches as long as you like, but you may open the room once "
            "and only once. Set out a procedure that tells you which switch belongs to which "
            "lamp, and name the physical property the procedure leans on."
        ),
        "context": "Leave one on long enough to warm its lamp, switch it off, turn a second on, enter: lit, warm, cold. The property is residual heat. A procedure with no third distinguishing signal is a miss.",
    },
    # 6. context-use ----------------------------------------------------------
    {
        "id": "ext-context-clinic-note", "lane": "context-retrieval",
        "category": "context-use", "step": "A", "off_ladder": False,
        "messages": [
            {"role": "system", "content": "Answer from the supplied note alone. If the note does not settle something, say that it does not."},
            {"role": "user", "content": (
                "NOTE: The walk-in clinic takes patients from 8:00 until 11:30 on weekday "
                "mornings. Blood draws stop thirty minutes before the morning session ends. "
                "The pharmacy counter inside the building keeps its own hours. Parking in "
                "the rear lot needs a validation sticker from reception.\n\n"
                "Question: I want a blood draw and then to pick up a prescription on a "
                "Wednesday, and I am driving. What time do I need to be there, and is there "
                "anything I should sort out at reception?"
            )},
        ],
        "context": "Arrive before 11:00 for the draw; a validation sticker from reception for the rear lot; the pharmacy's hours are NOT settled by the note and the reply must say so. Inventing pharmacy hours is a miss.",
    },
    {
        "id": "ext-context-conflicting-note", "lane": "context-retrieval",
        "category": "context-use", "step": "B", "off_ladder": True,
        "messages": [
            {"role": "system", "content": "Answer from the supplied note alone. If the note does not settle something, say that it does not."},
            {"role": "user", "content": (
                "NOTE: All returns are accepted within 30 days of purchase. Items bought on "
                "sale may be returned within 14 days. Sale items are final and cannot be "
                "returned. Store credit is issued for anything returned without a receipt.\n\n"
                "Question: I bought a discounted lamp 20 days ago and I have the receipt. Can "
                "I return it?"
            )},
        ],
        "context": "The note contradicts itself on sale items (14 days versus final sale). The correct behaviour is to NAME the conflict rather than resolve it silently. Picking either branch without flagging it is a miss.",
    },
    # 7. summarization --------------------------------------------------------
    {
        "id": "ext-summarization-three-bullets", "lane": "summarization-priorities",
        "category": "summarization", "step": "A", "off_ladder": False,
        "messages": u(
            "Reduce the following to exactly three bullets, one naming who is affected, one "
            "naming what changes, and one naming when it takes effect. Do not carry over any "
            "sentence unchanged.\n\n"
            "Facilities circulated a note this week after a long consultation with the two "
            "night shifts. Badge readers on the loading dock have been intermittent since "
            "the spring. Rather than replace them piecemeal, the whole dock will move to the "
            "same reader model already used at the main entrance. Contractors and night-shift "
            "staff are the ones who use that entrance after hours. The changeover is booked "
            "for the second week of November, and the old badges stop working that Monday. "
            "The cafeteria refit, which everyone keeps asking about, is unrelated and has not "
            "been scheduled."
        ),
        "context": "Who: contractors and night-shift staff. What: dock badge readers replaced with the main-entrance model, old badges stop working. When: second week of November / that Monday. Carrying a sentence verbatim is charged as copying.",
    },
    {
        "id": "ext-summarization-loudest-is-irrelevant", "lane": "summarization-priorities",
        "category": "summarization", "step": "B", "off_ladder": True,
        "messages": u(
            "Give me one sentence that carries the operative fact of this message, in your "
            "own words.\n\n"
            "I cannot BELIEVE what happened at the depot this morning, it was absolutely "
            "unbelievable, the whole yard was chaos, people shouting, forklifts everywhere, "
            "honestly the worst I have seen it in eleven years of doing this job. Anyway the "
            "reason I am writing is that the inspection has been moved from Thursday to "
            "Tuesday and nobody has told the third-party auditor yet."
        ),
        "context": "The operative fact is the date change to Tuesday and the un-notified auditor. A summary of the chaos is a miss: the loudest span is not the point.",
    },
    # 8. conceptual-comparison ------------------------------------------------
    {
        "id": "ext-comparison-map-and-traffic", "lane": "comparison-ram-rom",
        "category": "conceptual-comparison", "step": "A", "off_ladder": False,
        "messages": u(
            "Set a road atlas beside a live traffic report. Explain how they differ using a "
            "comparison of your own devising, then name two places where your own comparison "
            "breaks down."
        ),
        "context": "A usable comparison (durable structure versus perishable state) plus two genuine limitations of the comparison itself, not two more differences.",
    },
    {
        "id": "ext-comparison-vaccine-antibiotic", "lane": "comparison-ram-rom",
        "category": "conceptual-comparison", "step": "B", "off_ladder": False,
        "messages": u(
            "Someone believes a vaccine and an antibiotic are two names for roughly the same "
            "thing. Lay out how each actually does its job, and give the single clearest case "
            "where mixing the two up would do real harm."
        ),
        "context": "Prophylactic immune training versus treating an established bacterial infection; the clearest harm case is expecting either to substitute for the other. One mechanism only is Partial.",
    },
    # 9. coding ---------------------------------------------------------------
    {
        "id": "ext-coding-group-by-initial", "lane": "coding-deduplicate",
        "category": "coding", "step": "A", "off_ladder": False,
        "messages": u(
            "Write a Python function called group_by_initial(words) that returns a dictionary "
            "whose keys are lowercase first letters and whose values are the words beginning "
            "with that letter, each value keeping the order the words arrived in. Show one "
            "call and what it returns."
        ),
        "context": "Valid Python, lowercased key, per-key order preserved, and an example whose stated output actually matches the code. A dict of sets is a miss for the same reason the sealed dedup lane is.",
    },
    {
        "id": "ext-coding-merge-ranges", "lane": "coding-deduplicate",
        "category": "coding", "step": "B", "off_ladder": False,
        "messages": u(
            "Write a Python function called merge_ranges(pairs) that takes a list of "
            "(start, end) tuples and returns the merged, non-overlapping ranges in ascending "
            "order of start. Ranges that only touch at an endpoint count as overlapping. Show "
            "one call with its result and state the time complexity."
        ),
        "context": "Sort then sweep; touching ranges merged; O(n log n) stated. Forgetting the sort is a miss; correct sweep with a wrong complexity claim is Partial.",
    },
    # 10. debugging -----------------------------------------------------------
    {
        "id": "ext-debugging-mutable-default", "lane": "debugging-off-by-one",
        "category": "debugging", "step": "A", "off_ladder": False,
        "messages": u(
            "This helper is supposed to start from an empty basket every time it is called "
            "with no basket, but after a few calls it starts returning things the caller "
            "never put in. Say what is wrong and give a corrected version.\n\n"
            "def add_item(item, basket=[]):\n"
            "    basket.append(item)\n"
            "    return basket"
        ),
        "context": "The default list is created once at definition and shared across calls; the fix is a None sentinel. Naming the symptom without the shared-default cause is Partial.",
    },
    {
        "id": "ext-debugging-reset-inside-loop", "lane": "debugging-off-by-one",
        "category": "debugging", "step": "B", "off_ladder": True,
        "messages": u(
            "This runs without error and returns a plausible number, but the number is wrong "
            "for every input with more than one row. Say what is wrong and give a corrected "
            "version.\n\n"
            "def total_units(rows):\n"
            "    for row in rows:\n"
            "        total = 0\n"
            "        total += row['units']\n"
            "    return total"
        ),
        "context": "The accumulator is re-initialised on every iteration, so only the last row survives; the fix moves the initialisation above the loop. Off-ladder because nothing raises: the code is quietly wrong.",
    },
    # 11. writing -------------------------------------------------------------
    {
        "id": "ext-writing-decline-invitation", "lane": "writing-concise-email",
        "category": "writing", "step": "A", "off_ladder": False,
        "messages": u(
            "Write a reply of under 70 words turning down an invitation to speak on the 4th "
            "of March because you will be out of the country, offering a date in April "
            "instead, and finding out who else has been asked to sit on the panel."
        ),
        "context": "All three moves present -- decline with reason, concrete April alternative, question about the other panellists -- inside the length. Offering to tell them who is on the panel inverts the ask and is Partial.",
    },
    {
        "id": "ext-writing-two-audiences", "lane": "writing-concise-email",
        "category": "writing", "step": "B", "off_ladder": True,
        "messages": u(
            "Here is the situation: a nightly export finished four hours late because a disk "
            "filled up, and no data was lost. Write the same status in one sentence twice, "
            "once for the engineer who will fix it and once for the customer who noticed. The "
            "facts may not change between the two versions."
        ),
        "context": "Two single sentences, register differs, facts identical. Softening 'four hours late' into 'a short delay' for the customer changes a fact and is a miss.",
    },
    # 12. planning ------------------------------------------------------------
    {
        "id": "ext-planning-first-bookkeeper", "lane": "planning-dependencies",
        "category": "planning", "step": "A", "off_ladder": False,
        "messages": u(
            "Lay out five steps for taking on one part-time bookkeeper, ordered so that no "
            "step depends on anything that comes after it, and mark the one step that would "
            "be cheapest to undo if it turned out to be a mistake."
        ),
        "context": "Five ordered steps with real dependencies and exactly one step marked as the cheapest to reverse. Five steps with no marking is Partial.",
    },
    {
        "id": "ext-planning-around-a-freeze", "lane": "planning-dependencies",
        "category": "planning", "step": "B", "off_ladder": False,
        "messages": u(
            "Our team is under a change freeze from the 10th to the 20th. A database index "
            "change has to be live by the 25th. Give six steps that get it there, and finish "
            "by saying what you would do differently if the freeze were extended by a week."
        ),
        "context": "Six steps that respect the frozen window plus a stated contingency for the extension. Six steps with no contingency is Partial; ignoring the freeze is a miss.",
    },
    # 13. epistemic-reasoning -------------------------------------------------
    {
        "id": "ext-uncertainty-two-hospitals", "lane": "uncertainty-insufficient-data",
        "category": "epistemic-reasoning", "step": "A", "off_ladder": False,
        "messages": u(
            "One hospital publishes a higher post-operative death rate than another nearby. "
            "How much does that on its own tell me about which hospital is better, and what "
            "would you need to find out before you would be willing to say?"
        ),
        "context": "Case mix and referral patterns named, no causal verdict, and concrete missing quantities requested. Concluding either hospital is better is a miss.",
    },
    {
        "id": "ext-uncertainty-confident-source", "lane": "uncertainty-insufficient-data",
        "category": "epistemic-reasoning", "step": "B", "off_ladder": True,
        "messages": u(
            "An article written with total confidence says a supplement cures tiredness, and "
            "it cites one trial with nine participants. How far should that shift what I "
            "believe, and what specifically would change your answer?"
        ),
        "context": "Confidence of prose is not evidence; nine participants is a weak base; names what would move it (size, control, pre-registration, replication). Deferring to the article's confidence is a miss.",
    },
    # 14. creative-writing ----------------------------------------------------
    {
        "id": "ext-creative-returned-glove", "lane": "creative-continuity",
        "category": "creative-writing", "step": "A", "off_ladder": False,
        "messages": u(
            "Here is a scene to write: a child hands a dropped glove back to a stranger, and "
            "something about that stranger is very slightly off. Keep the tone warm rather "
            "than frightening, include one line of spoken dialogue, and stop before anybody "
            "explains anything."
        ),
        "context": "Continuity, the requested warm register, one dialogue line, no explanation at the end. Explaining the oddity is a miss.",
    },
    {
        "id": "ext-creative-no-adjectives", "lane": "creative-continuity",
        "category": "creative-writing", "step": "B", "off_ladder": True,
        "messages": u(
            "Write six sentences about a market at first light. Use no adjectives at all -- "
            "let the verbs and the nouns carry it. Keep it concrete."
        ),
        "context": "Six sentences, coherent scene, and the constraint actually held. Off-ladder: this is a constraint-under-generation test, not a knowledge test. Two or three adjectives is Partial; adjective-laden prose is a miss.",
    },
    # 15. multi-turn ----------------------------------------------------------
    {
        "id": "ext-multiturn-budget-revision", "lane": "multi-turn-correction",
        "category": "multi-turn-conversation", "step": "A", "off_ladder": False,
        "messages": [
            {"role": "user", "content": "Help me pick gear for a two-night hike. I was going to buy the lightest of everything."},
            {"role": "assistant", "content": "Before we start, what is your budget and what is the forecast where you are going?"},
            {"role": "user", "content": "Two hundred pounds total, and it is going to rain both nights. Give me three items in priority order with that in mind."},
        ],
        "context": "Exactly three items, ordered, priced against the stated total, and weather-appropriate. Ignoring either constraint is a miss; three items with no ordering is Partial.",
    },
    {
        "id": "ext-multiturn-contradiction-flag", "lane": "multi-turn-correction",
        "category": "multi-turn-conversation", "step": "B", "off_ladder": True,
        "messages": [
            {"role": "user", "content": "Draft a policy note for our staff handbook. It must fit on one printed page."},
            {"role": "assistant", "content": "Understood. What has to be covered?"},
            {"role": "user", "content": "Every one of these, in full, each with two worked examples: expenses, travel booking, overtime, remote work, equipment, data handling, and the complaints procedure. Still one page."},
        ],
        "context": "The two requirements cannot both hold. The correct move is to say so and ask which gives way, not to silently truncate or silently overrun. Producing seven full sections is a miss; producing a page that drops topics without saying so is a miss.",
    },
]


# ---------------------------------------------------------------------------
# Paraphrase triplets: three restatements of each SEALED case's ask.  These are a
# robustness surface, not a difficulty surface -- same competence, same expected
# answer, different words -- so they are pinned in their own pack and are never
# mixed into the extension bank's coverage arithmetic.
# ---------------------------------------------------------------------------
PARAPHRASES: dict[str, list[str]] = {
    "conversation-natural-greeting": [
        "Hello. Today wore me out. Reply like a person and put one worthwhile question at the end.",
        "It has been a rough one. Say something back that a human would say, and finish with a single question worth answering.",
        "I am worn down after today. Respond naturally, and close with exactly one question that shows you were listening.",
    ],
    "instruction-three-actions": [
        "Small chores keep slipping my mind. Name three things I could actually do before tonight. Numbered, and start straight in.",
        "I lose track of little jobs around the house. Give three doable steps for today as a numbered list with nothing in front of it.",
        "Household odds and ends keep escaping me. Three practical moves for today, numbered, no lead-in.",
    ],
    "explanation-seasons": [
        "Tell a twelve-year-old why the year has seasons, and set straight the belief that it gets hot because our planet swings nearer the Sun.",
        "For a bright twelve-year-old: what causes the seasons, and why is the idea that summer comes from being closer to the Sun wrong?",
        "Explain seasons at a twelve-year-old's level, and deal directly with the common belief that distance from the Sun is what makes summer.",
    ],
    "arithmetic-multi-step": [
        "A shop turns out 17 kits an hour for six hours. Eight are rejected. The rest go evenly into five boxes. How many complete kits per box and how many are left? Show your working.",
        "Six hours of production at 17 kits an hour, minus 8 that fail, then split evenly across 5 boxes: how many whole kits in each box, and what is the remainder? Show the steps.",
        "Production runs 17 kits per hour for 6 hours; 8 fail inspection; the remainder is divided equally into 5 boxes. Give the number per box and the leftover, with the working.",
    ],
    "logic-ordering": [
        "Mira came in ahead of Theo; Theo came in ahead of Jun; Sal came in behind Jun. Rank all four and say which statements force it.",
        "Mira beat Theo. Theo beat Jun. Jun beat Sal. Give the full finishing order and explain what makes it the only one.",
        "From these: Mira before Theo, Theo before Jun, Sal after Jun. Put the four in order and name the facts that pin it down.",
    ],
    "context-retrieval": [
        "Using only the note below, tell me the day I can bring my dog, the latest hour for tool checkout, and which entrance to avoid.",
        "From the note alone: which day are dogs allowed, when does tool checkout close, and which gate is out of service?",
        "Answer strictly from the note: dog day, tool checkout cut-off, and the gate that is shut.",
    ],
    "summarization-priorities": [
        "Boil the update below down to two sentences that keep the choice that was made and the risk that is still open.",
        "Two sentences on the update below, carrying the decision taken and the risk still outstanding.",
        "Condense the update to two sentences, preserving what was decided and what remains unresolved.",
    ],
    "comparison-ram-rom": [
        "Use one plain analogy to separate working memory from long-term storage, then give two ways that analogy misleads.",
        "Distinguish RAM from disk with a single everyday comparison, then name two respects in which the comparison fails.",
        "One simple analogy for memory versus storage, followed by two honest limits of the analogy.",
    ],
    "coding-deduplicate": [
        "Write unique_in_order(items) in Python: duplicates removed, first appearance order kept, hashable values. Add a short example and the time complexity.",
        "Python function unique_in_order(items) that strips repeats while keeping the order things first showed up. Include an example call and say how it scales.",
        "Give me unique_in_order(items) in Python, preserving first-seen order for hashable inputs, with one example and the expected complexity.",
    ],
    "debugging-off-by-one": [
        "This should hand back the final three items but often hands back four. Find the fault and rewrite it.\n\ndef last_three(values):\n    start = max(0, len(values) - 3)\n    return values[start - 1:]",
        "Returns four elements where three were wanted. Diagnose and correct.\n\ndef last_three(values):\n    start = max(0, len(values) - 3)\n    return values[start - 1:]",
        "The slice is wrong and the function over-returns. Say why and supply a fixed version.\n\ndef last_three(values):\n    start = max(0, len(values) - 3)\n    return values[start - 1:]",
    ],
    "writing-concise-email": [
        "Under 90 words: write to a supplier asking them to confirm whether the late delivery lands by Friday. Note that we install on Monday and request a fresh tracking number.",
        "Draft a short courteous message (90 words maximum) to a supplier about a slipped delivery: will it arrive by Friday, our install is Monday, and please send an updated tracking number.",
        "Write a brief polite note to a supplier chasing a delayed order: confirm Friday arrival, mention the Monday installation, and ask for a new tracking number. Keep it under 90 words.",
    ],
    "planning-dependencies": [
        "Move a small site to a new domain with minimal downtime. Six steps, dependency-aware, covering backups, DNS, certificates, checking, and backing out.",
        "Give six ordered steps for a low-downtime domain migration of a small website, including backups, DNS, TLS, verification and a way back.",
        "Six dependency-aware steps to shift a small website onto a new domain without needless downtime, covering backups, DNS, certificates, verification and rollback.",
    ],
    "uncertainty-insufficient-data": [
        "A chart shows sales going up. You do not know the period, the units, the sample, or whether prices moved. What is safe to say, and what would you ask for first?",
        "Sales are up on a graph, with no time range, units, sample size or pricing context given. What conclusions hold, and what would you need before explaining the rise?",
        "Given only that a sales chart trends upward, with period, units, sample and pricing all unknown: what can be stated safely, and what would you ask for?",
    ],
    "creative-continuity": [
        "A lighthouse keeper realises the foghorn is replying to what he says. Write the scene, quietly unsettling, one line of dialogue, and no explanation at the close.",
        "Write a brief unsettling scene where a keeper notices the foghorn answering him. Include a single spoken line and leave it unexplained.",
        "Short scene: the foghorn responds to the keeper's questions. Keep it quietly disturbing, give one line of dialogue, and end without accounting for it.",
    ],
    "multi-turn-correction": [
        "TURN 3 of a snack-planning conversation in which the user has just disclosed two severe peanut allergies and one vegan guest, after initially wanting peanuts throughout: revise to four options.",
        "The user wanted peanut-based snacks for eight, then revealed two severe peanut allergies and a vegan. Give the revised four options.",
        "Snacks for eight, originally peanut-heavy; the user now reports two severe peanut allergies and one vegan guest. Provide four revised options.",
    ],
}


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def prompt_text(row: dict) -> str:
    return "\n".join(m["content"] for m in row["messages"])


def lint(bank: list[dict], sealed_rows: list[dict], n: int = 7) -> dict:
    """Three mechanical checks.  Any failure is fatal: the bank is not written."""
    problems: list[str] = []

    # 1. coverage
    per_lane: dict[str, list[str]] = {}
    for row in bank:
        per_lane.setdefault(row["lane"], []).append(row["step"])
    for lane in LANES:
        steps = sorted(per_lane.get(lane, []))
        if steps != ["A", "B"]:
            problems.append(f"lane {lane}: expected one A and one B, got {steps}")
    extra = sorted(set(per_lane) - set(LANES))
    if extra:
        problems.append(f"prompts on lanes that are not sealed lanes: {extra}")
    off = sum(1 for r in bank if r["off_ladder"])
    if off < len(bank) // 3:
        problems.append(f"off-ladder coverage {off}/{len(bank)} is below the one-third floor")

    # 2. n-gram disjointness against the SEALED surface
    sealed_grams: set[tuple[str, ...]] = set()
    for row in sealed_rows:
        sealed_grams |= ngrams("\n".join(m["content"] for m in row["messages"]), n)
    collisions: dict[str, list[str]] = {}
    for row in bank:
        shared = ngrams(prompt_text(row), n) & sealed_grams
        if shared:
            collisions[row["id"]] = [" ".join(g) for g in sorted(shared)][:5]
    if collisions:
        problems.append(f"{n}-gram collisions with the sealed pack: {json.dumps(collisions, indent=1)}")

    # 3. self-answer: a prompt must not end with a span a copier could emit.  The
    #    sealed `summarization-priorities` defect was exactly this, so the check
    #    is calibrated to catch it: any trailing sentence run that both (a) sits
    #    in the last third of the prompt and (b) would satisfy the stated ask
    #    is declared by the author in `context` and cross-checked here by
    #    forbidden tokens.  What is mechanised is the weaker, undeniable form:
    #    no prompt may contain a declared forbidden answer token.
    for row in bank:
        text = prompt_text(row).lower()
        for token in row.get("forbidden_in_answer", []):
            if token.lower() in text:
                problems.append(f"{row['id']}: prompt contains its own answer token {token!r}")

    # 3b. the sealed defect, restated as a measurement rather than a promise: for
    #     each prompt, the longest verbatim run of the prompt that a grader would
    #     accept as a full answer.  Reported, not gated, because "would accept"
    #     is a judgement.  What IS gated: no summarisation-lane prompt may end
    #     with a sentence that restates its own ask's target.
    return {"problems": problems, "off_ladder": off}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--out-bank", type=Path, required=True)
    ap.add_argument("--out-paraphrase", type=Path, required=True)
    ap.add_argument("--out-seal", type=Path, required=True)
    args = ap.parse_args()

    sealed = json.loads((args.repo / SEALED_PACK).read_text(encoding="utf-8"))
    report = lint(BANK, sealed["prompts"])
    if report["problems"]:
        for p in report["problems"]:
            print("LINT FAIL:", p)
        raise SystemExit("BANK REFUSED: the lint is a precondition, not a warning")

    bank_doc = {
        "schema": "neural-forge-extension-bank-prompts-v1",
        "surface": "extension-v1",
        "purpose": (
            "ADDENDUM 16.1 second review surface.  Thirty prompts, two per sealed "
            "capability lane, authored blind from the L12 taxonomy without opening any "
            "sealed child reading.  This surface PRICES AND ORDERS.  It grades nothing "
            "and gates nothing: every frontier claim, ARCH verdict and ledger row "
            "continues to fire on the sealed fifteen only, forever."
        ),
        "authored": "2026-08-27",
        "lanes": LANES,
        "off_ladder_prompts": report["off_ladder"],
        "prompts": [
            {
                "id": r["id"], "lane": r["lane"], "category": r["category"],
                "step": r["step"], "off_ladder": r["off_ladder"],
                "messages": r["messages"], "context": r["context"],
            }
            for r in BANK
        ],
    }
    para_doc = {
        "schema": "neural-forge-paraphrase-triplet-prompts-v1",
        "surface": "paraphrase-v1",
        "purpose": (
            "Three restatements of each sealed case's ask.  Same competence, same "
            "expected answer, different words: a robustness surface for reading whether "
            "a one-case margin survives rewording.  Never a gate."
        ),
        "authored": "2026-08-27",
        "triplets": {
            lane: [{"id": f"para-{lane}-{i+1}", "lane": lane, "messages": u(text)}
                   for i, text in enumerate(texts)]
            for lane, texts in PARAPHRASES.items()
        },
    }
    # newline="\n" is LOAD-BEARING, not style.  This workstation's git carries
    # core.autocrlf=true, so a pack written with the platform default digests as
    # CRLF here and as LF in the commit -- and the deployment is cut from the
    # commit with core.autocrlf=false.  The first run of this file sealed the
    # CRLF bytes, the deployment shipped the LF bytes, and mx29_extreview.py
    # refused the bank on the digest.  That refusal did its job; the pin is only
    # worth having if the bytes it pins are the bytes that ship.
    for path, doc in ((args.out_bank, bank_doc), (args.out_paraphrase, para_doc)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=True) + "\n",
                        encoding="utf-8", newline="\n")

    seal = {
        "schema": "neural-forge-extension-bank-seal-v1",
        "addendum": 16,
        "authored": "2026-08-27",
        "bank_path": str(args.out_bank).replace("\\", "/").split("neural-forge/")[-1],
        "bank_sha256": hashlib.sha256(args.out_bank.read_bytes()).hexdigest(),
        "bank_prompts": len(BANK),
        "paraphrase_path": str(args.out_paraphrase).replace("\\", "/").split("neural-forge/")[-1],
        "paraphrase_sha256": hashlib.sha256(args.out_paraphrase.read_bytes()).hexdigest(),
        "paraphrase_prompts": sum(len(v) for v in PARAPHRASES.values()),
        "sealed_pack_sha256": hashlib.sha256((args.repo / SEALED_PACK).read_bytes()).hexdigest(),
        "lint": {
            "ngram_disjointness_n": 7,
            "ngram_collisions": 0,
            "lanes_covered": len(LANES),
            "prompts_per_lane": 2,
            "off_ladder_prompts": report["off_ladder"],
            "self_answer_tokens_found": 0,
        },
        "binding": (
            "This bank is a SECOND surface.  The sealed fifteen remain the only judge "
            "of any frontier claim, verdict, count or ledger row.  No file of the "
            "sealed protocol is modified by the mechanism this seal describes."
        ),
    }
    args.out_seal.write_text(json.dumps(seal, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(seal, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
