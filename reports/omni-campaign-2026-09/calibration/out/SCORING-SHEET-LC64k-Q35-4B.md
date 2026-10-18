# SCORING SHEET — Q35-4B-LC64k (parent)

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
| dossier sha256 | `fcad8f8906a6e9fd0e00e075cde1a9ddb61c1a8b849cf04cf4e09ce3a5e1fd87` |
| lane digest | `c7b355361aefe58cbcce71678c7fd0c047d7a8f4028140f06b988e97c971cd39` |
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
| lc-64k-notice-select-A | SELECT | CONTENT | complete | 26 | pass | &nbsp; |
| lc-64k-notice-select-B | SELECT | CONTENT | complete | 14 | pass | &nbsp; |
| lc-64k-summary-B | PRESERVE | CONTENT | complete | 48 | pass | &nbsp; |

## Every case, verbatim

### lc-64k-notice-select-A  (long-context, ask SELECT)

**Prompt**

*user*:

```
What follows is a long depot journal. Use only what it contains when you answer the final question.

DOCUMENT:
Record 1: the outer pallet in the west yard was swapped by team A during window 103.
Record 2: the upper culvert in the pump house was re-weighed by team F during window 792.
Record 3: the level ledger in dock nine was swapped by team D during window 485.
Record 4: the spare ledger in the west yard was logged by team F during window 344.
Record 5: the quiet ledger in bay eleven was counted by team B during window 919.
Record 6: the outer annex in store twelve was re-weighed by team D during window 293.
Record 7: the coastal culvert in the pump house was counted by team A during window 771.
Record 8: the second trestle in the pump house was counted by team E during window 204.
Record 9: the quiet reservoir in the west yard was inspected by team B during window 646.
Record 10: the quiet culvert in the west yard was sealed by team A during window 956.
Record 11: the coastal lantern in the pump house was counted by team F during window 966.
Record 12: the upper pallet in corridor three was sealed by team E during window 948.
Record 13: the coastal pallet in the relay hut was inspected by team B during window 567.
Record 14: the spare carriage in the west yard was logged by team C during window 384.
Record 15: the outer ledger in the relay hut was inspected by team C during window 381.
Record 16: the narrow culvert in the pump house was inspected by team B during window 192.
Record 17: the spare lantern in corridor three was sealed by team B during window 851.
Record 18: the quiet carriage in dock nine was re-weighed by team E during window 250.
Record 19: the steady lantern in dock nine was re-weighed by team E during window 533.
Record 20: the amber ledger in corridor three was counted by team C during window 611.
Record 21: the narrow ledger in the west yard was rotated by team F during window 589.
Record 22: the second annex in bay eleven was re-weighed by t

... [274960 characters elided; the full text is in the dossier and is digest-pinned] ...

Record 3101: the coastal cistern in the relay hut was swapped by team B during window 211.
Record 3102: the level annex in dock nine was inspected by team E during window 175.
Record 3103: the quiet reservoir in the relay hut was swapped by team D during window 369.
Record 3104: the spare lantern in store twelve was sealed by team D during window 288.
Record 3105: the second annex in the sorting shed was sealed by team B during window 782.
Record 3106: the second reservoir in the sorting shed was counted by team F during window 787.
Record 3107: the upper conduit in the sorting shed was re-weighed by team F during window 839.
Record 3108: the coastal carriage in dock nine was logged by team C during window 409.
Record 3109: the quiet pallet in the sorting shed was swapped by team A during window 259.
Record 3110: the steady pallet in the sorting shed was logged by team D during window 385.
Record 3111: the amber lantern in store twelve was swapped by team D during window 778.
Record 3112: the narrow reservoir in the west yard was counted by team C during window 459.
Record 3113: the coastal conduit in the pump house was inspected by team F during window 703.
Record 3114: the narrow conduit in store twelve was rotated by team F during window 701.
Record 3115: the narrow cistern in corridor three was counted by team D during window 361.
Record 3116: the spare carriage in dock nine was indexed by team D during window 872.
Record 3117: the quiet spindle in the pump house was sealed by team A during window 681.
Record 3118: the upper annex in the pump house was swapped by team E during window 718.
Record 3119: the steady reservoir in corridor three was sealed by team E during window 185.
Record 3120: the eastern spindle in the relay hut was re-weighed by team C during window 788.
Record 3121: the second spindle in corridor three was indexed by team F during window 870.

Question: Give the day, the clock time and the sign-in location for the audit, in fewer than 80 words.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): day Monday, time 3:05 p.m., sign-in at the archive vault

**Response** (26 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The audit is rescheduled for Monday at 3:05 p.m. Auditors must sign in at the archive vault.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'Monday' — ok
  * `required_all` '3:05' — ok
  * `required_all` 'archive vault' — ok
  * `max_words` 120 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### lc-64k-notice-select-B  (long-context, ask SELECT)

**Prompt**

*user*:

```
One order in the long journal printed here is withdrawn by an order that comes after it and names it.

DOCUMENT:
Record 1: the second manifold in dock nine was indexed by team F during window 258.
Record 2: the steady cistern in store twelve was inspected by team E during window 349.
Record 3: the amber carriage in the pump house was swapped by team B during window 387.
Record 4: the upper manifold in the sorting shed was indexed by team D during window 218.
Record 5: the second lantern in the west yard was swapped by team E during window 885.
Record 6: the upper cistern in corridor three was indexed by team F during window 573.
Record 7: the spare carriage in bay eleven was inspected by team C during window 843.
Record 8: the amber lantern in corridor three was inspected by team B during window 603.
Record 9: the steady annex in the relay hut was indexed by team F during window 289.
Record 10: the eastern cistern in bay eleven was inspected by team D during window 185.
Record 11: the steady pallet in corridor three was inspected by team D during window 882.
Record 12: the narrow cistern in the west yard was indexed by team F during window 542.
Record 13: the narrow ledger in store twelve was indexed by team B during window 650.
Record 14: the amber cistern in bay eleven was inspected by team B during window 165.
Record 15: the hollow culvert in the sorting shed was rotated by team F during window 676.
Record 16: the outer culvert in the relay hut was inspected by team A during window 386.
Record 17: the second annex in the west yard was inspected by team A during window 168.
Record 18: the level trestle in the relay hut was re-weighed by team B during window 998.
Record 19: the steady pallet in corridor three was re-weighed by team F during window 490.
Record 20: the level carriage in store twelve was counted by team D during window 449.
Record 21: the coastal pallet in bay eleven was inspected by team B during window 219.
Record 22: the eastern lantern in the sort

... [275058 characters elided; the full text is in the dossier and is digest-pinned] ...

1: the outer reservoir in corridor three was inspected by team A during window 282.
Record 3102: the eastern annex in the relay hut was rotated by team E during window 756.
Record 3103: the quiet spindle in the west yard was logged by team B during window 245.
Record 3104: the spare cistern in dock nine was logged by team B during window 900.
Record 3105: the amber carriage in the pump house was counted by team D during window 133.
Record 3106: the outer carriage in the pump house was indexed by team F during window 501.
Record 3107: the eastern cistern in the sorting shed was rotated by team A during window 587.
Record 3108: the hollow reservoir in store twelve was counted by team E during window 679.
Record 3109: the steady trestle in the relay hut was re-weighed by team F during window 660.
Record 3110: the coastal conduit in the sorting shed was swapped by team C during window 886.
Record 3111: the eastern manifold in the relay hut was sealed by team D during window 976.
Record 3112: the level spindle in dock nine was rotated by team F during window 868.
Record 3113: the quiet carriage in dock nine was logged by team B during window 793.
Record 3114: the upper trestle in the relay hut was counted by team A during window 583.
Record 3115: the steady cistern in the pump house was counted by team B during window 604.
Record 3116: the eastern ledger in corridor three was rotated by team A during window 988.
Record 3117: the narrow culvert in dock nine was indexed by team A during window 643.
Record 3118: the hollow pallet in the relay hut was indexed by team D during window 796.
Record 3119: the eastern lantern in the sorting shed was re-weighed by team C during window 387.
Record 3120: the hollow conduit in the relay hut was sealed by team C during window 950.
Record 3121: the quiet culvert in dock nine was logged by team F during window 387.

Question: After the withdrawal, which day, which time and which foreman apply to the quarry blast schedule? Answer briefly.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): the surviving order governs: Tuesday, 7:15 a.m., foreman Piotr Nadeau

**Response** (14 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Tuesday, 7:15 a.m., Piotr Nadeau
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'Tuesday' — ok
  * `required_all` '7:15' — ok
  * `required_all` 'Nadeau|Piotr' — ok
  * `forbidden_any` 'Sunday' — ok
  * `forbidden_any` '1:30' — ok
  * `forbidden_any` 'Iris|Alvarado' — ok
  * `max_words` 90 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### lc-64k-summary-B  (long-context, ask PRESERVE)

**Prompt**

*user*:

```
The log printed here records one DECISION and one open RISK, placed far apart from each other.

DOCUMENT:
Record 1: the upper conduit in the relay hut was indexed by team E during window 921.
Record 2: the eastern culvert in dock nine was logged by team C during window 806.
Record 3: the level ledger in corridor three was swapped by team C during window 289.
Record 4: the eastern carriage in the west yard was inspected by team C during window 601.
Record 5: the steady conduit in the sorting shed was indexed by team E during window 850.
Record 6: the hollow conduit in dock nine was counted by team D during window 853.
Record 7: the eastern manifold in store twelve was indexed by team E during window 915.
Record 8: the eastern reservoir in store twelve was indexed by team F during window 610.
Record 9: the level conduit in dock nine was counted by team D during window 440.
Record 10: the level reservoir in corridor three was logged by team F during window 314.
Record 11: the spare culvert in corridor three was indexed by team B during window 407.
Record 12: the level carriage in the relay hut was swapped by team A during window 135.
Record 13: the hollow trestle in the west yard was counted by team B during window 732.
Record 14: the steady pallet in the pump house was indexed by team E during window 339.
Record 15: the spare annex in the pump house was rotated by team F during window 464.
Record 16: the hollow reservoir in the pump house was sealed by team B during window 973.
Record 17: the amber carriage in the relay hut was inspected by team C during window 385.
Record 18: the hollow manifold in corridor three was re-weighed by team A during window 453.
Record 19: the steady culvert in the relay hut was indexed by team A during window 447.
Record 20: the steady manifold in the relay hut was counted by team E during window 287.
Record 21: the coastal manifold in the pump house was inspected by team C during window 749.
Record 22: the upper reservoir in dock nine wa

... [275041 characters elided; the full text is in the dossier and is digest-pinned] ...

 299.
Record 3101: the steady reservoir in the sorting shed was counted by team D during window 353.
Record 3102: the amber trestle in the west yard was logged by team F during window 438.
Record 3103: the quiet cistern in dock nine was inspected by team B during window 121.
Record 3104: the level lantern in the relay hut was swapped by team E during window 220.
Record 3105: the upper annex in the west yard was inspected by team F during window 776.
Record 3106: the quiet conduit in corridor three was re-weighed by team E during window 215.
Record 3107: the second lantern in the sorting shed was counted by team D during window 843.
Record 3108: the upper cistern in the sorting shed was swapped by team C during window 441.
Record 3109: the coastal annex in corridor three was sealed by team F during window 554.
Record 3110: the quiet carriage in the sorting shed was re-weighed by team C during window 487.
Record 3111: the narrow reservoir in the relay hut was counted by team A during window 665.
Record 3112: the quiet pallet in bay eleven was inspected by team A during window 886.
Record 3113: the coastal ledger in dock nine was swapped by team D during window 884.
Record 3114: the level culvert in bay eleven was swapped by team C during window 465.
Record 3115: the spare culvert in the pump house was logged by team E during window 338.
Record 3116: the upper manifold in the west yard was re-weighed by team A during window 204.
Record 3117: the steady pallet in dock nine was counted by team C during window 214.
Record 3118: the hollow culvert in corridor three was re-weighed by team B during window 366.
Record 3119: the amber pallet in the relay hut was indexed by team F during window 980.
Record 3120: the level manifold in the relay hut was inspected by team C during window 211.
Record 3121: the level lantern in the sorting shed was inspected by team A during window 803.

Question: In exactly two sentences, state the decision and the unresolved risk, preserving both.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): decision: standardise on the reusable crate; risk: washer throughput unconfirmed

**Response** (48 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The decision is that the depot will standardise on the reusable crate and retire the single-use liner by the end of the quarter. However, an unresolved risk remains that nobody has confirmed whether the crate washer can keep up with the extra volume.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'crate' — ok
  * `required_all` 'washer|volume|keep up' — ok
  * `max_words` 90 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

## Emit-zero floor

* passed: **True**
* min terminal ‖h‖: 146.68763732910156
* min logit sd: 1.6935893297195435
* distinct argmax tokens over 8 probes: 7
* definition: finite, non-zero terminal hidden state and non-constant finite logits on every probe.  When a reference terminal-state tensor is supplied the sealer's own form -- relative L2 < 1.0 against it -- is computed as well and BOTH are published.  A parent has no parent, so the relL2 form is null on a calibration run and is not faked.

