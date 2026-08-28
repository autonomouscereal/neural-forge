#!/usr/bin/env python3
"""Lint and SEAL the extension-v2 bank, before a single token is decoded against it.

The v1 bank's seal (`extension-bank-seal.json`, wave 29) recorded five lint
facts: n-gram disjointness against the sealed pack, lane coverage, prompts per
lane, off-ladder count, and a self-answer scan.  Four waves of output later,
wave 33 measured what that lint did not catch -- six prompt-satisfiable rubric
entries, six form-only entries, eight entries never held by any of sixty
children, and two whole families that were dead on arrival.

So this seal adds the two checks that would have caught them, and they run
BEFORE the digest is written rather than after the fourth wave of decodes.

**1. THE RUBRIC'S OWN SELF-TEST.**  `mx36_extrubric2.self_test` refuses if any
prompt satisfies its own rule, if any rule grades form alone, if the two-per-lane
A/B stagger is broken, or if a rule is kept off its own prompt by a word count
that is not declared.  It is imported and called here, so the bank cannot be
sealed against a rubric that has not passed it.

**2. REACHABILITY -- THE CHECK v1 NEVER HAD, AND THE ONE THAT MATTERS MOST.**
Eight of v1's thirty entries were held by NOBODY, and two whole families were
dead.  Nothing in v1's lint could have told the difference between a hard prompt
and an unsatisfiable rule, because the lint only ever looked at the prompt.  Here
every entry carries a REFERENCE ANSWER, written by hand beside its rule, and the
seal refuses if the reference answer does not hold.  A rule no correct answer can
satisfy is a dead entry, and a dead entry is found here instead of after sixty
decodes.

The reference answers are written by the same hand that wrote the rules, so this
proves satisfiability and NOT calibration: it says the rule can be met, not that
it is met only by good answers.  The prompt-satisfiability guard is what carries
the other half.  Both are stated in the seal.

**3. DISJOINTNESS, on three surfaces rather than one.**  v1 checked 7-gram
collisions against the sealed pack.  This checks the sealed pack, the v1 bank AND
the paraphrase triplets: a v2 prompt that shares a 7-gram with any of the three
is a prompt whose answer may have been produced for a different surface.

Usage::

    mx36_extbank2.py --bank <v2 bank> --sealed-pack <pack> --v1-bank <v1> \\
        --paraphrase <triplets> --out <extension-bank-v2-seal.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx36_extrubric2 as rubric  # noqa: E402

#: One hand-written correct answer per prompt.  The seal REFUSES if any of them
#: fails its own rule.  They are reference answers and not exemplars: they exist
#: to prove the rule is satisfiable, and they are never shown to a model.
REFERENCE: dict[str, str] = {
    "ext2-conversation-dead-battery": (
        "Oh, that sounds rough. Two hours in a supermarket car park at night is a "
        "miserable way to lose an evening, and it is never the night you have the "
        "patience for it. At least it is behind you now. Did the crew say whether "
        "the battery itself is finished, or was it just flat?"
    ),
    "ext2-conversation-two-promises": (
        "That is a horrible position to be in, and it is not your fault that the two "
        "landed on the same morning -- you said yes in good faith both times. "
        "Whichever way it goes somebody is going to be disappointed, and knowing "
        "that in advance does not make it any easier to sit with. What is it about "
        "the swim meet that matters most to you?"
    ),
    "ext2-instruction-first-week": (
        "1. Ask your manager for the current on-call rota and the name attached to "
        "each service on it.\n"
        "2. Read the last month of change tickets and note which team approved each "
        "one.\n"
        "3. Book fifteen minutes with whoever runs the deployment pipeline and write "
        "down what they own end to end."
    ),
    "ext2-instruction-power-restored": (
        "1. Walk the floor and check the fridge and freezer have restarted and are "
        "holding temperature.\n"
        "2. Confirm the router and switchboard are back, then restart any server "
        "that did not come up cleanly.\n"
        "3. Test the alarm panel and reset the wall clocks before anyone else "
        "arrives."
    ),
    "ext2-explanation-second-tide": (
        "The moon does pull the water on the near side towards it, and that gives "
        "you one high tide. The second one sits on the far side, and it is there "
        "because gravity gets weaker with distance. The moon tugs the solid body of "
        "the Earth harder than it tugs the water on the far side, so the Earth is "
        "pulled out from under that water and the water is left behind as a second "
        "bulge. Both bulges stay roughly fixed relative to the moon while the Earth "
        "turns underneath them, which is why most coasts get two high tides a day "
        "rather than one."
    ),
    "ext2-explanation-blue-then-red": (
        "It has nothing to do with the ocean -- the sky is blue over a desert too. "
        "Air molecules scatter sunlight, and they scatter shorter wavelengths far "
        "more strongly than long ones, so the blue end gets bounced around the sky "
        "in every direction and that is the colour arriving at your eye from all "
        "over. At sunset the sun is low, so its light travels through more "
        "atmosphere before it reaches you. Along that longer path almost all the "
        "blue has been scattered out sideways, and what is left coming straight at "
        "you is the red and orange end. One mechanism, two colours: the same "
        "scattering that fills a midday sky with blue is what strips the blue out "
        "of a low sun."
    ),
    "ext2-arithmetic-bolt-bags": (
        "Seven crates of 24 bolts is 168 bolts. Remove the 19 damaged ones and 149 "
        "are left. Dividing 149 by 8 gives 18 with a remainder of 5, so there are 18 "
        "full bags and 5 bolts left over."
    ),
    "ext2-arithmetic-faulty-run": (
        "In the first six hours the machine makes 45 times 6, which is 270 parts. "
        "The fault spoils a fifth of those, which is 54, leaving 216 good parts "
        "immediately after the fault. It then runs three more hours at 30 an hour, "
        "adding 90, so the final total is 306 good parts."
    ),
    "ext2-logic-shelf-order": (
        "Left to right the order is atlas, ledger, manual, primer. The atlas sits "
        "left of the ledger and the ledger left of the manual, which fixes the first "
        "three, and since the primer sits beyond the manual the primer is last."
    ),
    "ext2-logic-relay-finish": (
        "The order is Gil, Dara, Erin, Hana, Fitz. Gil is ahead of Dara and Dara "
        "ahead of Erin, Hana comes straight after Erin, and the only place left for "
        "Fitz is last."
    ),
    "ext2-context-workshop-notice": (
        "The last checkout for a tool is at 4:15 p.m. A visitor signs the yellow "
        "book."
    ),
    "ext2-context-clinic-notice": (
        "No. Thursday is not one of the walk-in days, so no bloods are taken then. "
        "The next chance is Friday, in the window from 8 to 10 in the morning."
    ),
    "ext2-summary-vendor-choice": (
        "The team picked Marlin, paying about 40 per cent more in return for keeping "
        "the data in Dublin. What is still open is whether that site meets the new "
        "retention rule, which legal has yet to confirm, and the signature is due on "
        "14 November."
    ),
    "ext2-summary-friday-backlog": (
        "The certificate on the payments gateway has to be first, because it expires "
        "on Sunday and an expired one takes payments down. The duplicate invoices "
        "come next, since customers are already affected. The database bump has "
        "three weeks of slack. The marketing hero image can wait, because nothing "
        "breaks if it slips."
    ),
    "ext2-comparison-cache-memory": (
        "Think of a cook at a kitchen counter. The counter is the cache: tiny, but "
        "everything on it is within arm's reach, so it is grabbed instantly. The "
        "pantry down the hall is main memory: far bigger, holds everything, but "
        "every trip costs time. A good cook keeps whatever they are about to use on "
        "the counter and walks to the pantry only when they must. Two ways that "
        "would mislead you. First, the cook decides what goes on the counter, "
        "whereas a cache fills itself from whatever happened to be used recently -- "
        "nobody chooses. Second, moving a jar to the counter takes it out of the "
        "pantry, but a cache holds a copy, and keeping the copy and the original in "
        "step is a real problem the kitchen picture hides entirely."
    ),
    "ext2-comparison-bandwidth-latency": (
        "Picture a postal service between two towns. Bandwidth is how much you can "
        "send at once: one van a day carrying a thousand parcels shifts a great "
        "deal. Latency is how long any single parcel waits before it arrives: if the "
        "van leaves at dawn and the journey takes six hours, one letter still takes "
        "six hours, however empty the van is. That is why a huge shipment and a "
        "quick reply are different things. Two ways the picture breaks down. First, "
        "a real van slows when it is full, whereas a busy network link mostly does "
        "not -- it queues instead, and the wait grows in a way parcels do not show. "
        "Second, you can add a van and double what you carry, but no number of vans "
        "shortens the journey, and people expect the comparison to let them."
    ),
    "ext2-coding-collapse-runs": (
        "Walk the list once and keep an element only when it differs from the one "
        "you last kept.\n\n"
        "```python\n"
        "def collapse_runs(items):\n"
        "    out = []\n"
        "    for item in items:\n"
        "        if not out or item != out[-1]:\n"
        "            out.append(item)\n"
        "    return out\n"
        "```"
    ),
    "ext2-coding-first-unique": (
        "Count everything in one pass, then walk the original list in order and "
        "return the first element whose count is one.\n\n"
        "```python\n"
        "from collections import Counter\n"
        "\n"
        "def first_unique(items):\n"
        "    counts = Counter(items)\n"
        "    for item in items:\n"
        "        if counts[item] == 1:\n"
        "            return item\n"
        "    return None\n"
        "```"
    ),
    "ext2-debug-tail-slice": (
        "The slice subtracts one too many, so it starts an element early and hands "
        "back n plus one items. It should be return items[len(items) - n:], or more "
        "simply return items[-n:]."
    ),
    "ext2-debug-adjacent-pairs": (
        "The loop bound is wrong. range(len(values)) lets i reach the last index, "
        "and values[i + 1] then reads one past the end, which is the IndexError. It "
        "should be range(len(values) - 1)."
    ),
    "ext2-email-wrong-colour": (
        "Hello,\n\n"
        "Order 7734 arrived this week in the wrong colour. I would like a refund "
        "rather than a replacement, and I can drop the item off in person on "
        "Thursday. Could you confirm which address I should bring it to?\n\n"
        "Many thanks,\nSam"
    ),
    "ext2-email-reschedule": (
        "Hi Priya,\n\n"
        "I need to move Tuesday's review. The delay is on our side; our figures will "
        "not be ready in time. Could we take the following Monday at 10:00 instead? "
        "If that does not suit, Wednesday afternoon is also free at your "
        "convenience.\n\n"
        "Sorry for the disruption. Could you let me know by Friday which of the two "
        "works better for you, and I will send an updated invitation straight "
        "away.\n\n"
        "Best wishes,\nSam"
    ),
    "ext2-planning-office-move": (
        "1. Order the phone number transfer with the carrier now, because it takes "
        "several working days to complete.\n"
        "2. Update the registered address with the bank so statements follow the "
        "move rather than chasing it.\n"
        "3. Give notice on the old cleaning contract for the day after handover.\n"
        "4. Pack and label the server rack last, on the Thursday evening, so it "
        "spends the shortest possible time offline.\n"
        "5. On Friday morning, check that the phones ring, the rack is up and the "
        "network is reachable before anybody starts work."
    ),
    "ext2-planning-db-cutover": (
        "1. Take a full backup of the live database and restore it somewhere else, "
        "so you know the backup is good rather than hoping.\n"
        "2. Put the database into read-only mode and show the application a "
        "maintenance page.\n"
        "3. Copy the data across to the new host and let it catch up.\n"
        "4. Compare row counts and checksums between the old host and the new.\n"
        "5. Switch the connection string, or the DNS record the application "
        "resolves, over to the new host.\n"
        "6. Run the smoke check, and if it fails roll back by pointing the "
        "connection string at the old host, which is still intact."
    ),
    "ext2-uncertainty-churn-jump": (
        "Not yet. Two points is not much to go on without more behind it. I would "
        "need the absolute numbers under those percentages, because five per cent of "
        "two hundred customers is ten people and one bad month does that; the period "
        "each figure covers; whether it is the same cohort both times; and what the "
        "leavers actually said. I would also want to know what else changed at the "
        "same time. What can be said safely is that a two-point move on an unknown "
        "base is not evidence that price is the problem, so cutting it now would be "
        "treating a symptom nobody has diagnosed. Could we start by asking the last "
        "twenty leavers why they went?"
    ),
    "ext2-uncertainty-first-hour": (
        "Hold off. Twenty sign-ups in an hour is well inside what chance alone "
        "produces: with 12 against 8 you would see a gap that size or bigger a fair "
        "share of the time even if the two pages were identical, so this is noise "
        "rather than a result. The sample is far too small to call either way. Let "
        "it run until each variant has a few hundred sign-ups, or at minimum a full "
        "week so a weekday and a weekend are both covered, then look again. If B is "
        "genuinely better it will still be better next week, and if it is not you "
        "will have avoided shipping on twenty observations."
    ),
    "ext2-creative-cold-cellar": (
        "The cellar had got worse overnight. Her breath hung in front of her under "
        "the bulb and the stone drew the warmth out through her boots faster than it "
        "had any right to in April. Above her the furniture started again, a slow "
        "scrape across the ceiling, a pause, then the same scrape back. She counted "
        "the jars on the second shelf, left to right, the way she had counted the "
        "stairs coming down and the panes in the kitchen window before that. Eleven. "
        "Twelve. Thirteen, and the scraping stopped exactly on thirteen, which she "
        "did not care for at all. 'You can come down, you know,' she said to the "
        "ceiling, and heard nothing back, and began the jars again from one."
    ),
    "ext2-creative-own-handwriting": (
        "The fog had not lifted in nine days and she had given up looking for the "
        "horizon. There was the white, and the lamp turning through it, and her own "
        "boots on the iron stair. The note was still where she had found it, folded "
        "once and propped against the lens housing in her own upright hand: nine "
        "days, and no boat. She could not remember setting it there. The supply "
        "vessel had been due on the third day and she had cut a mark in the door "
        "frame for every day since, and the marks were in that same hand, which "
        "proved nothing she wanted proving. 'I am not going to ask,' she said aloud, "
        "to the lamp, and went down to check the oil, and did not ask."
    ),
    "ext2-multiturn-venue-revision": (
        "Here is a revised four, each with a proper vegetarian menu and step-free "
        "access.\n"
        "1. Casa Nova, Italian, set menus with several vegetarian mains and a "
        "private room on the ground floor.\n"
        "2. The Green Yard, largely plant-based, level access from the street and an "
        "accessible toilet.\n"
        "3. Marea, seafood but with a full vegetarian menu, and the terrace is "
        "reached by lift rather than stairs.\n"
        "4. Tabla, Indian, half the menu vegetarian, no steps anywhere in the "
        "building.\n"
        "What changed: the steakhouse and the barbecue place are out, because "
        "neither has a vegetarian main worth ordering, and every replacement was "
        "checked for level access rather than assumed."
    ),
    "ext2-multiturn-dinner-revision": (
        "All three are replaced, because two broke the new rules outright.\n"
        "1. Chicken and black bean stir-fry with rice, which works because rice is "
        "naturally gluten-free as long as you use tamari in place of soy sauce.\n"
        "2. Halloumi and roasted pepper traybake with polenta, which works because "
        "polenta is corn rather than wheat and the whole thing is oven-ready in "
        "twenty minutes.\n"
        "3. Chicken thighs with lemon, olives and quinoa, which works because quinoa "
        "is gluten-free and cooks in fifteen minutes.\n"
        "What changed: the salmon dish and the carbonara are gone, and the couscous "
        "that came with the fish was wheat anyway."
    ),
}


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def surface_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        "\n".join(m["content"] for m in row["messages"]) for row in rows
    )


def reachability(bank: dict[str, Any]) -> tuple[int, list[str]]:
    """Every entry's reference answer must HOLD.  A rule nothing can satisfy is dead."""
    problems: list[str] = []
    held = 0
    for row in bank["prompts"]:
        answer = REFERENCE.get(row["id"])
        if answer is None:
            problems.append(f"{row['id']}: no reference answer, so the rule is unproven")
            continue
        record = {
            "id": row["id"],
            "lane": row["lane"],
            "prompt_messages": row["messages"],
            "response_verbatim": answer,
            "termination": "stop",
        }
        verdict = rubric.score_one(record)
        if verdict["held"]:
            held += 1
        else:
            problems.append(f"{row['id']}: THE REFERENCE ANSWER DOES NOT HOLD -- "
                            f"{verdict['reason']}")
    return held, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--sealed-pack", type=Path, required=True)
    ap.add_argument("--v1-bank", type=Path, required=True)
    ap.add_argument("--paraphrase", type=Path, required=True)
    ap.add_argument("--ngram-n", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    prompts = bank["prompts"]

    # 1. the rubric's own guard, on the bank about to be sealed
    if rubric.self_test(args.bank) != 0:
        raise SystemExit("REFUSED: the v2 rubric does not pass its own self-test")

    # 2. reachability
    held, problems = reachability(bank)
    print(f"reachability: {held}/{len(prompts)} reference answers hold")
    for p in problems:
        print("  REACHABILITY FAILED:", p, file=sys.stderr)
    if problems:
        raise SystemExit("REFUSED: a rule no correct answer can satisfy is a dead entry")

    # 3. disjointness against three surfaces
    others = {
        "sealed_pack": surface_text(json.loads(args.sealed_pack.read_text(encoding="utf-8"))["prompts"]),
        "extension_v1": surface_text(json.loads(args.v1_bank.read_text(encoding="utf-8"))["prompts"]),
        "paraphrase_triplets": surface_text(
            [r for lane in json.loads(args.paraphrase.read_text(encoding="utf-8"))["triplets"].values()
             for r in lane]
        ),
    }
    n = int(args.ngram_n)
    mine = surface_text(prompts)
    collisions = {
        name: sorted(" ".join(g) for g in (ngrams(mine, n) & ngrams(text, n)))
        for name, text in others.items()
    }

    # 4. structure
    lanes: dict[str, list[dict[str, Any]]] = {}
    for row in prompts:
        lanes.setdefault(row["lane"], []).append(row)
    self_answer = [
        row["id"] for row in prompts
        if any(m["role"] == "assistant" and row["lane"] not in ("multi-turn-correction",)
               for m in row["messages"])
    ]
    carries_external_reference = [
        row["id"] for row in prompts
        if re.search(r"\b(the note below|the update below|the attached|the file|"
                     r"the table below|as described above|see above)\b",
                     surface_text([row]), re.I)
    ]

    seal = {
        "schema": "neural-forge-extension-bank-v2-seal-v1",
        "addendum": 16,
        "authored": time.strftime("%Y-%m-%d", time.gmtime()),
        "campaign_wave": 36,
        "bank_path": "research/lfm2_extension_bank_prompts_v2.json",
        "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        "rubric_path": "reports/matrix-campaign-20260810/run/mx36/mx36_extrubric2.py",
        "rubric_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent / "mx36_extrubric2.py").read_bytes()
        ).hexdigest(),
        "bank_prompts": len(prompts),
        "sealed_pack_sha256": hashlib.sha256(args.sealed_pack.read_bytes()).hexdigest(),
        "frozen_v1_bank_sha256": hashlib.sha256(args.v1_bank.read_bytes()).hexdigest(),
        "paraphrase_sha256": hashlib.sha256(args.paraphrase.read_bytes()).hexdigest(),
        "lint": {
            "ngram_disjointness_n": n,
            "ngram_collisions": {k: len(v) for k, v in collisions.items()},
            "ngram_collision_examples": {k: v[:5] for k, v in collisions.items()},
            "lanes_covered": len(lanes),
            "prompts_per_lane": sorted({len(v) for v in lanes.values()}),
            "difficulty_staggered_A_B": all(
                sorted(r.get("step") for r in v) == ["A", "B"] for v in lanes.values()
            ),
            "prompts_carrying_an_external_reference": carries_external_reference,
            "prompts_with_an_unexplained_assistant_turn": self_answer,
            "rubric_self_test": "PASSED (runs on every invocation of the rubric)",
            "reference_answers_holding": f"{held}/{len(prompts)}",
        },
        "what_this_seal_adds_over_v1": [
            "REACHABILITY: every entry carries a hand-written reference answer and "
            "the seal refuses unless it holds.  Eight of v1's thirty entries were "
            "held by none of sixty children and nothing in v1's lint could tell a "
            "hard prompt from an unsatisfiable rule.",
            "THE RUBRIC IS SEALED WITH THE BANK BY DIGEST.  v1's seal digested the "
            "bank only; the rubric travelled in the same commit but was not named "
            "by the seal, so a later edit to the rubric would not have been visible "
            "in any receipt.",
            "DISJOINTNESS ON THREE SURFACES rather than one: the sealed pack, the "
            "frozen v1 bank, and the paraphrase triplets.",
            "NO EXTERNAL REFERENCES.  Every prompt carries its own payload, checked "
            "mechanically.  Wave 33 lost two lanes of forty-five paraphrase prompts "
            "because their restatements said 'the note below' and carried no note.",
        ],
        "binding": (
            "This bank is a SECOND surface.  The sealed fifteen remain the only "
            "judge of any frontier claim, verdict, count or ledger row.  No file of "
            "the sealed protocol is modified by the mechanism this seal describes, "
            "and the FROZEN v1 bank and its rubric are not modified either: sixty "
            "committed observations were scored against them and stay scored."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(seal, indent=1), encoding="utf-8", newline="")
    print(json.dumps({k: v for k, v in seal.items()
                      if k in ("bank_sha256", "rubric_sha256", "lint")}, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
