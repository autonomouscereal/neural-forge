# SCORING SHEET — Q35-4B-LC200k (parent)

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
| dossier sha256 | `b1412ded70838efb2428b93107c81c959997af4ffc8a65f224a94f67179274c2` |
| lane digest | `14c070394316eda040cd1247d171ccaf21bcde1b6702d5c60ed8467e8123d557` |
| emit-zero floor | True |
| decode-path preflight agree | None |

## Columns

* **Column A — copy-back counts:** 0 / 2  (UPPER BOUND (no reading attached yet))
* **Column B — copy-back refused:** 0 / 2  (UPPER BOUND (no reading attached yet))
* what the bound subtracts: only EMPTY, LOOP, CAP_RIDE — the boxes a machine can prove hold nothing. An ECHO lane is still inside column A, because *“this is a copy”* is a threshold call and not a certainty; your reading may hold any of the 2 scored lanes.
* canonical boxes — signal 0, artifact-miss 0, content-miss 0, still to read 2

## Lane summary

| lane | ask | form box | term | tokens | rubric | your grade |
|---|---|---|---|---|---|---|
| lc-200k-notice-select-A | SELECT | EMPTY | out-of-memory | 0 | no match | &nbsp; |
| lc-200k-notice-select-B | SELECT | EMPTY | out-of-memory | 0 | no match | &nbsp; |

## Every case, verbatim

### lc-200k-notice-select-A  (long-context, ask SELECT)

**Prompt**

*user*:

```
A very long journal appears beneath this line. Answer its closing question from its contents alone.

DOCUMENT:
Record 1: the spare cistern in the pump house was swapped by team B during window 468.
Record 2: the level culvert in bay eleven was rotated by team F during window 747.
Record 3: the upper spindle in the sorting shed was inspected by team B during window 328.
Record 4: the second cistern in bay eleven was swapped by team A during window 652.
Record 5: the eastern ledger in the sorting shed was counted by team C during window 375.
Record 6: the level carriage in store twelve was logged by team A during window 675.
Record 7: the second cistern in the west yard was inspected by team D during window 580.
Record 8: the steady manifold in the sorting shed was sealed by team C during window 807.
Record 9: the second annex in the sorting shed was rotated by team E during window 884.
Record 10: the spare lantern in bay eleven was counted by team E during window 240.
Record 11: the narrow cistern in bay eleven was rotated by team C during window 166.
Record 12: the amber reservoir in corridor three was swapped by team E during window 622.
Record 13: the eastern cistern in store twelve was logged by team D during window 229.
Record 14: the upper manifold in corridor three was inspected by team F during window 864.
Record 15: the second conduit in the relay hut was sealed by team E during window 960.
Record 16: the narrow pallet in store twelve was re-weighed by team E during window 557.
Record 17: the outer spindle in the west yard was logged by team A during window 356.
Record 18: the hollow culvert in corridor three was indexed by team F during window 653.
Record 19: the second conduit in the west yard was re-weighed by team D during window 207.
Record 20: the steady culvert in the pump house was logged by team B during window 399.
Record 21: the hollow reservoir in dock nine was indexed by team F during window 133.
Record 22: the outer spindle in corridor three wa

... [848930 characters elided; the full text is in the dossier and is digest-pinned] ...

g window 654.
Record 9504: the quiet trestle in bay eleven was logged by team C during window 164.
Record 9505: the level annex in dock nine was indexed by team D during window 216.
Record 9506: the steady culvert in corridor three was counted by team A during window 898.
Record 9507: the second ledger in bay eleven was indexed by team C during window 527.
Record 9508: the steady reservoir in the relay hut was re-weighed by team E during window 661.
Record 9509: the amber carriage in store twelve was counted by team D during window 996.
Record 9510: the level lantern in the relay hut was counted by team E during window 883.
Record 9511: the quiet culvert in bay eleven was swapped by team D during window 277.
Record 9512: the quiet trestle in the west yard was inspected by team F during window 426.
Record 9513: the narrow cistern in the pump house was inspected by team D during window 199.
Record 9514: the narrow reservoir in the relay hut was counted by team B during window 949.
Record 9515: the upper manifold in bay eleven was swapped by team F during window 519.
Record 9516: the amber conduit in the west yard was counted by team C during window 118.
Record 9517: the second conduit in the west yard was rotated by team D during window 804.
Record 9518: the narrow cistern in the relay hut was sealed by team A during window 174.
Record 9519: the level cistern in corridor three was rotated by team E during window 662.
Record 9520: the upper trestle in corridor three was indexed by team B during window 809.
Record 9521: the amber ledger in the sorting shed was inspected by team D during window 720.
Record 9522: the steady cistern in corridor three was counted by team D during window 456.
Record 9523: the coastal conduit in the relay hut was sealed by team E during window 670.
Record 9524: the second ledger in store twelve was inspected by team B during window 121.

Question: Name the weekday, the hour and the reporting point for the bunker survey. Under 80 words please.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): day Saturday, time 11:25 a.m., report to the mill office

**Response** (0 tokens, out-of-memory, form box EMPTY — the prompt did not fit in memory on this device, so the model was never asked; this measures the envelope, not the model)

```
(empty)
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_all` 'Saturday' — MISS
  * `required_all` '11:25' — MISS
  * `required_all` 'mill office' — MISS
  * `max_words` 120 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### lc-200k-notice-select-B  (long-context, ask SELECT)

**Prompt**

*user*:

```
Somewhere inside the enormous journal quoted next, an earlier memo is revoked by a memo that identifies it.

DOCUMENT:
Record 1: the outer carriage in bay eleven was swapped by team A during window 623.
Record 2: the upper pallet in corridor three was counted by team E during window 304.
Record 3: the second ledger in dock nine was inspected by team C during window 230.
Record 4: the eastern reservoir in corridor three was inspected by team F during window 134.
Record 5: the coastal manifold in the west yard was swapped by team E during window 340.
Record 6: the outer pallet in corridor three was inspected by team C during window 447.
Record 7: the amber cistern in dock nine was indexed by team B during window 652.
Record 8: the upper pallet in dock nine was swapped by team A during window 306.
Record 9: the eastern conduit in the pump house was re-weighed by team C during window 135.
Record 10: the upper spindle in corridor three was indexed by team F during window 878.
Record 11: the steady ledger in the sorting shed was re-weighed by team A during window 590.
Record 12: the outer carriage in store twelve was logged by team E during window 608.
Record 13: the upper annex in the west yard was sealed by team D during window 311.
Record 14: the outer cistern in store twelve was indexed by team C during window 645.
Record 15: the steady carriage in the sorting shed was sealed by team A during window 356.
Record 16: the outer pallet in the pump house was indexed by team F during window 243.
Record 17: the second culvert in the relay hut was counted by team D during window 573.
Record 18: the second conduit in bay eleven was swapped by team F during window 108.
Record 19: the eastern trestle in the relay hut was indexed by team E during window 672.
Record 20: the amber reservoir in the west yard was logged by team C during window 237.
Record 21: the steady conduit in the west yard was swapped by team F during window 155.
Record 22: the spare cistern in the west yard was

... [849100 characters elided; the full text is in the dossier and is digest-pinned] ...

41.
Record 9504: the outer culvert in the pump house was logged by team D during window 467.
Record 9505: the hollow cistern in the pump house was sealed by team A during window 690.
Record 9506: the spare reservoir in the west yard was rotated by team C during window 735.
Record 9507: the narrow lantern in store twelve was swapped by team E during window 542.
Record 9508: the steady pallet in the sorting shed was swapped by team D during window 346.
Record 9509: the amber cistern in dock nine was logged by team F during window 825.
Record 9510: the coastal culvert in store twelve was logged by team D during window 106.
Record 9511: the steady conduit in bay eleven was swapped by team F during window 110.
Record 9512: the coastal trestle in the sorting shed was counted by team D during window 442.
Record 9513: the eastern conduit in the sorting shed was inspected by team C during window 577.
Record 9514: the narrow cistern in the relay hut was logged by team B during window 475.
Record 9515: the amber cistern in the relay hut was inspected by team E during window 125.
Record 9516: the eastern conduit in store twelve was swapped by team F during window 821.
Record 9517: the quiet reservoir in corridor three was counted by team E during window 526.
Record 9518: the spare reservoir in the pump house was indexed by team B during window 187.
Record 9519: the eastern ledger in bay eleven was indexed by team E during window 859.
Record 9520: the amber manifold in the pump house was counted by team A during window 726.
Record 9521: the level spindle in the relay hut was logged by team A during window 740.
Record 9522: the outer trestle in the west yard was sealed by team F during window 651.
Record 9523: the amber culvert in bay eleven was re-weighed by team E during window 416.
Record 9524: the upper trestle in store twelve was logged by team F during window 517.

Question: Give the surviving weighbridge booking: its weekday, its hour and its technician, in under 60 words.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): the surviving memo governs: Friday, 2:35 p.m., technician Ondine Marsh

**Response** (0 tokens, out-of-memory, form box EMPTY — the prompt did not fit in memory on this device, so the model was never asked; this measures the envelope, not the model)

```
(empty)
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_all` 'Friday' — MISS
  * `required_all` '2:35' — MISS
  * `required_all` 'Ondine|Marsh' — MISS
  * `forbidden_any` 'Halvard|Bruun' — ok
  * `forbidden_any` '5:50' — ok
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

