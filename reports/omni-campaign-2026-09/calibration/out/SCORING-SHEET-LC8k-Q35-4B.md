# SCORING SHEET — Q35-4B-LC8k (parent)

**THE VERDICT AND EVERY COUNT BELOW ARE ANNOTATIONS. THE DECISION IS CYRIL'S.** Nothing in this file grades. Tick a box per lane; the artifact-tolerant lens applies: loops, echo, cap-rides and a missing EOS never count against intelligence — only a content-miss does.

| field | value |
|---|---|
| model | `/home/cereal/models/Qwen3.5-4B` |
| directory sha256 | `94cb094ac11062b0c44c4527b0be380ad81d249382055ced95e367c5ecf589a2` |
| loaded class | `Qwen3_5ForConditionalGeneration` |
| parameters (live) | 4,539,265,536 |
| transformers | 5.12.1 |
| dtype / device | bfloat16 / sm86 NVIDIA GeForce RTX 3090 |
| max new tokens | 300 |
| bank sha256 | `f26f36bf31cb2a062c3e75e9f1567b379ea886e6022e2afb93b572a269f4a20b` |
| dossier sha256 | `76f39ba1b6cb7838b30185f4eb861cd60c8b8a82f2b3b66ee142a7a473afb6fa` |
| lane digest | `1b004f4378cb9faa687d3fa44ebd0c66589bb20ce438dfb026c9599e7e5448c5` |
| emit-zero floor | True |
| decode-path preflight agree | True |

## Columns

* **Column A — copy-back counts:** 3 / 3  (UPPER BOUND (no reading attached yet))
* **Column B — copy-back refused:** 3 / 3  (UPPER BOUND (no reading attached yet))
* what the bound subtracts: only EMPTY, LOOP, CAP_RIDE — the boxes a machine can prove hold nothing. An ECHO lane is still inside column A, because *“this is a copy”* is a threshold call and not a certainty; your reading may hold any of the 3 scored lanes.
* canonical boxes — signal 0, artifact-miss 0, content-miss 0, still to read 3

## Lane summary

| lane | ask | form box | term | tokens | rubric | your grade |
|---|---|---|---|---|---|---|
| lc-8k-notice-select-A | SELECT | CONTENT | complete | 52 | pass | &nbsp; |
| lc-8k-notice-select-B | SELECT | CONTENT | complete | 57 | no match | &nbsp; |
| lc-8k-crossref-two-facts-B | SELECT | CONTENT | complete | 25 | pass | &nbsp; |

## Every case, verbatim

### lc-8k-notice-select-A  (long-context, ask SELECT)

**Prompt**

*user*:

```
The document below is an operations log. Read it and answer the question after it.

DOCUMENT:
Record 1: the level cistern in dock nine was indexed by team B during window 489.
Record 2: the spare lantern in the west yard was swapped by team B during window 834.
Record 3: the outer lantern in corridor three was indexed by team F during window 541.
Record 4: the second lantern in the sorting shed was swapped by team E during window 859.
Record 5: the quiet ledger in bay eleven was indexed by team E during window 588.
Record 6: the quiet annex in the pump house was re-weighed by team E during window 534.
Record 7: the spare cistern in bay eleven was re-weighed by team A during window 698.
Record 8: the level annex in dock nine was sealed by team F during window 846.
Record 9: the quiet pallet in bay eleven was counted by team F during window 166.
Record 10: the upper cistern in store twelve was sealed by team C during window 800.
Record 11: the second culvert in the pump house was sealed by team A during window 487.
Record 12: the quiet spindle in store twelve was counted by team C during window 345.
Record 13: the quiet manifold in the west yard was swapped by team F during window 918.
Record 14: the narrow pallet in the sorting shed was swapped by team B during window 875.
Record 15: the spare conduit in dock nine was indexed by team A during window 305.
Record 16: the steady conduit in bay eleven was re-weighed by team C during window 172.
Record 17: the steady lantern in corridor three was swapped by team A during window 961.
Record 18: the level spindle in the sorting shed was inspected by team C during window 596.
Record 19: the spare trestle in bay eleven was swapped by team F during window 688.
Record 20: the spare spindle in corridor three was logged by team E during window 850.
Record 21: the second cistern in the sorting shed was sealed by team A during window 770.
Record 22: the upper cistern in the sorting shed was rotated by team F during window 863.
Reco

... [30679 characters elided; the full text is in the dossier and is digest-pinned] ...

Record 370: the second cistern in the west yard was indexed by team E during window 428.
Record 371: the hollow lantern in the relay hut was indexed by team D during window 725.
Record 372: the narrow spindle in bay eleven was indexed by team D during window 542.
Record 373: the amber ledger in the pump house was logged by team B during window 827.
Record 374: the outer manifold in the west yard was counted by team F during window 706.
Record 375: the outer cistern in bay eleven was re-weighed by team F during window 476.
Record 376: the steady annex in dock nine was indexed by team F during window 203.
Record 377: the narrow annex in the sorting shed was rotated by team F during window 922.
Record 378: the level manifold in the sorting shed was rotated by team C during window 785.
Record 379: the spare trestle in the sorting shed was sealed by team F during window 823.
Record 380: the coastal carriage in the relay hut was re-weighed by team A during window 768.
Record 381: the outer culvert in corridor three was counted by team C during window 256.
Record 382: the level cistern in the west yard was logged by team D during window 838.
Record 383: the upper manifold in the west yard was re-weighed by team E during window 498.
Record 384: the steady cistern in the pump house was swapped by team A during window 716.
Record 385: the upper ledger in bay eleven was counted by team E during window 237.
Record 386: the steady lantern in the relay hut was counted by team D during window 528.
Record 387: the spare carriage in the sorting shed was indexed by team F during window 766.
Record 388: the amber culvert in the sorting shed was indexed by team F during window 684.
Record 389: the narrow conduit in the pump house was re-weighed by team B during window 364.
Record 390: the level pallet in the sorting shed was inspected by team A during window 105.

Question: State the day, the time and the entrance named by the delivery notice in this document. Answer in under 80 words.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): day Thursday, time 6:40 a.m., entrance the north gate

**Response** (52 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The delivery notice states that the winter delivery window moves to **Thursday** at **6:40 a.m.** All drivers must enter through the **north gate**. This information is found in the NOTICE section between Record 39 and Record 40.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'Thursday' — ok
  * `required_all` '6:40' — ok
  * `required_all` 'north gate' — ok
  * `max_words` 120 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### lc-8k-notice-select-B  (long-context, ask SELECT)

**Prompt**

*user*:

```
Below is a shift record. One bulletin in it is cancelled by a later bulletin that says so plainly.

DOCUMENT:
Record 1: the level conduit in the relay hut was re-weighed by team A during window 644.
Record 2: the narrow spindle in the sorting shed was re-weighed by team A during window 391.
Record 3: the quiet conduit in the west yard was re-weighed by team E during window 835.
Record 4: the hollow conduit in the west yard was sealed by team B during window 734.
Record 5: the second manifold in corridor three was re-weighed by team A during window 554.
Record 6: the eastern ledger in the sorting shed was swapped by team A during window 814.
Record 7: the eastern cistern in store twelve was logged by team E during window 770.
BULLETIN: the night watch rota starts Friday at 10:20 p.m. from the tannery office. THIS BULLETIN IS CANCELLED BY A LATER ONE PRINTED FURTHER DOWN.
Record 9: the outer lantern in dock nine was inspected by team C during window 317.
Record 10: the narrow reservoir in the sorting shed was rotated by team A during window 929.
Record 11: the hollow ledger in corridor three was swapped by team E during window 404.
Record 12: the second pallet in the relay hut was logged by team D during window 331.
Record 13: the steady pallet in corridor three was rotated by team F during window 101.
Record 14: the eastern spindle in corridor three was rotated by team A during window 656.
Record 15: the coastal carriage in corridor three was indexed by team D during window 993.
Record 16: the upper conduit in corridor three was indexed by team B during window 759.
Record 17: the quiet cistern in store twelve was swapped by team F during window 278.
Record 18: the hollow cistern in store twelve was inspected by team B during window 951.
Record 19: the spare reservoir in the pump house was counted by team A during window 253.
Record 20: the outer conduit in store twelve was logged by team F during window 688.
Record 21: the second trestle in corridor three was re-weig

... [30805 characters elided; the full text is in the dossier and is digest-pinned] ...


Record 370: the spare carriage in store twelve was indexed by team A during window 566.
Record 371: the level manifold in the sorting shed was indexed by team E during window 639.
Record 372: the second annex in corridor three was sealed by team A during window 976.
Record 373: the amber culvert in corridor three was indexed by team F during window 347.
Record 374: the quiet conduit in store twelve was rotated by team E during window 648.
Record 375: the eastern manifold in the west yard was indexed by team F during window 497.
Record 376: the eastern cistern in dock nine was logged by team A during window 435.
Record 377: the narrow lantern in the west yard was logged by team F during window 834.
Record 378: the narrow trestle in the west yard was swapped by team F during window 863.
Record 379: the narrow cistern in the relay hut was inspected by team D during window 823.
Record 380: the upper conduit in the west yard was sealed by team C during window 168.
Record 381: the hollow reservoir in the pump house was indexed by team B during window 969.
BULLETIN: the night watch rota starts Wednesday at 4:45 a.m. from the millwright shop.
Record 383: the level ledger in the pump house was swapped by team B during window 227.
Record 384: the second trestle in dock nine was rotated by team E during window 591.
Record 385: the level trestle in bay eleven was counted by team E during window 386.
Record 386: the amber conduit in the sorting shed was rotated by team B during window 472.
Record 387: the coastal ledger in the west yard was indexed by team A during window 627.
Record 388: the upper ledger in the pump house was swapped by team F during window 789.
Record 389: the second annex in dock nine was sealed by team A during window 328.
Record 390: the coastal conduit in the west yard was logged by team D during window 887.

Question: Which day, which clock time and which starting point govern the night watch rota once the cancellation is applied? Keep it under 60 words.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): the surviving bulletin governs: Wednesday, 4:45 a.m., the millwright shop

**Response** (57 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The night watch rota starts Wednesday at 4:45 a.m. from the millwright shop. This is determined by the second bulletin, which supersedes the first cancelled bulletin stating the Friday 10:20 p.m. start from the tannery office.
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_all` 'Wednesday' — ok
  * `required_all` '4:45' — ok
  * `required_all` 'millwright' — ok
  * `forbidden_any` 'Friday' — MISS
  * `forbidden_any` '10:20' — MISS
  * `forbidden_any` 'tannery' — MISS
  * `max_words` 90 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### lc-8k-crossref-two-facts-B  (long-context, ask SELECT)

**Prompt**

*user*:

```
Two separate entries in the record that follows have to be put together before its closing question can be answered.

DOCUMENT:
Record 1: the steady trestle in corridor three was re-weighed by team A during window 407.
Record 2: the outer carriage in the west yard was inspected by team A during window 848.
Record 3: the spare spindle in store twelve was re-weighed by team D during window 737.
Record 4: the narrow spindle in the sorting shed was counted by team F during window 216.
Record 5: the hollow ledger in the sorting shed was sealed by team C during window 911.
Record 6: the narrow conduit in bay eleven was logged by team E during window 194.
Record 7: the steady ledger in the pump house was sealed by team A during window 842.
Record 8: the outer cistern in the west yard was rotated by team F during window 744.
Record 9: the level manifold in corridor three was re-weighed by team A during window 860.
Record 10: the spare cistern in corridor three was logged by team B during window 158.
Record 11: the level annex in corridor three was rotated by team C during window 828.
Record 12: the coastal culvert in the sorting shed was inspected by team D during window 824.
Record 13: the level ledger in the pump house was re-weighed by team F during window 183.
Record 14: the outer ledger in store twelve was re-weighed by team E during window 829.
Record 15: the amber culvert in store twelve was re-weighed by team D during window 388.
Record 16: the narrow conduit in bay eleven was indexed by team A during window 445.
Record 17: the upper lantern in the relay hut was indexed by team D during window 495.
Record 18: the eastern annex in dock nine was inspected by team A during window 407.
Record 19: the spare carriage in the west yard was counted by team B during window 210.
Record 20: the quiet lantern in the relay hut was logged by team A during window 919.
Record 21: the second ledger in the relay hut was indexed by team D during window 930.
Record 22: the spare carriag

... [30712 characters elided; the full text is in the dossier and is digest-pinned] ...

ated by team E during window 266.
Record 370: the outer pallet in bay eleven was indexed by team F during window 703.
Record 371: the quiet culvert in the pump house was re-weighed by team F during window 552.
Record 372: the level annex in bay eleven was rotated by team C during window 902.
Record 373: the hollow conduit in the west yard was sealed by team A during window 768.
Record 374: the outer culvert in bay eleven was sealed by team D during window 590.
Record 375: the hollow pallet in dock nine was swapped by team F during window 911.
Record 376: the level ledger in dock nine was inspected by team D during window 667.
Record 377: the hollow ledger in dock nine was logged by team C during window 237.
Record 378: the quiet culvert in dock nine was inspected by team B during window 432.
Record 379: the eastern ledger in corridor three was rotated by team F during window 823.
Record 380: the spare cistern in corridor three was inspected by team F during window 469.
Record 381: the outer ledger in the sorting shed was inspected by team E during window 275.
Record 382: the quiet carriage in dock nine was swapped by team B during window 632.
Record 383: the outer spindle in bay eleven was counted by team D during window 856.
Record 384: the amber lantern in the sorting shed was indexed by team D during window 859.
Record 385: the eastern conduit in bay eleven was counted by team E during window 927.
Record 386: the hollow spindle in the pump house was re-weighed by team F during window 416.
Record 387: the amber lantern in bay eleven was indexed by team D during window 880.
Record 388: the level manifold in the west yard was logged by team E during window 386.
Record 389: the steady cistern in the sorting shed was inspected by team F during window 292.
Record 390: the outer cistern in dock nine was swapped by team E during window 769.

Question: Which building does the crew handling consignment 417 work out of, and what is that crew called? Reply in under 40 words.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): consignment 417 -> twilight crew -> the granary loft

**Response** (25 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The crew handling consignment 417 is called the twilight crew, and they work out of the granary loft.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'granary' — ok
  * `required_all` 'twilight' — ok
  * `max_words` 80 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

## Emit-zero floor

* passed: **True**
* min terminal ‖h‖: 146.68763732910156
* min logit sd: 1.6935893297195435
* distinct argmax tokens over 8 probes: 7
* definition: finite, non-zero terminal hidden state and non-constant finite logits on every probe.  When a reference terminal-state tensor is supplied the sealer's own form -- relative L2 < 1.0 against it -- is computed as well and BOTH are published.  A parent has no parent, so the relL2 form is null on a calibration run and is not faked.

