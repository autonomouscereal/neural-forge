# SCORING SHEET — Q35-4B-LC32k (parent)

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
| dossier sha256 | `38bf7be69c044eb9bc3cb533c956606d96679c8de7237f5d54e7606a16aa589c` |
| lane digest | `3cd9dfdcf7bce1b9757e66c3289a4813ac6fb45ac56f6804d3fed34067fd53bf` |
| emit-zero floor | True |
| decode-path preflight agree | True |

## Columns

* **Column A — copy-back counts:** 1 / 1  (UPPER BOUND (no reading attached yet))
* **Column B — copy-back refused:** 1 / 1  (UPPER BOUND (no reading attached yet))
* what the bound subtracts: only EMPTY, LOOP, CAP_RIDE — the boxes a machine can prove hold nothing. An ECHO lane is still inside column A, because *“this is a copy”* is a threshold call and not a certainty; your reading may hold any of the 1 scored lanes.
* canonical boxes — signal 0, artifact-miss 0, content-miss 0, still to read 1

## Lane summary

| lane | ask | form box | term | tokens | rubric | your grade |
|---|---|---|---|---|---|---|
| lc-32k-conflicting-updates-B | SELECT | CONTENT | complete | 13 | pass | &nbsp; |

## Every case, verbatim

### lc-32k-conflicting-updates-B  (long-context, ask SELECT)

**Prompt**

*user*:

```
Where two updates disagree in the material printed here, the later update replaces the earlier one.

DOCUMENT:
Record 1: the spare conduit in the sorting shed was re-weighed by team B during window 726.
Record 2: the level carriage in the relay hut was counted by team D during window 872.
Record 3: the quiet pallet in bay eleven was swapped by team C during window 886.
Record 4: the quiet pallet in corridor three was swapped by team F during window 790.
Record 5: the level carriage in bay eleven was inspected by team F during window 674.
Record 6: the coastal trestle in store twelve was sealed by team B during window 889.
Record 7: the amber conduit in the west yard was sealed by team B during window 360.
Record 8: the second spindle in store twelve was logged by team C during window 426.
Record 9: the amber trestle in the sorting shed was logged by team A during window 852.
Record 10: the narrow manifold in dock nine was counted by team E during window 280.
Record 11: the level trestle in the sorting shed was counted by team C during window 531.
Record 12: the second manifold in dock nine was rotated by team E during window 498.
Record 13: the second spindle in the relay hut was re-weighed by team C during window 382.
Record 14: the coastal manifold in store twelve was counted by team D during window 536.
Record 15: the second spindle in bay eleven was swapped by team E during window 226.
Record 16: the level lantern in the west yard was sealed by team A during window 382.
Record 17: the steady ledger in the relay hut was rotated by team E during window 802.
Record 18: the steady ledger in the relay hut was counted by team E during window 906.
Record 19: the spare culvert in bay eleven was re-weighed by team F during window 397.
Record 20: the level culvert in the pump house was re-weighed by team E during window 217.
Record 21: the second culvert in the west yard was re-weighed by team F during window 284.
Record 22: the level cistern in the sorting shed was seale

... [134926 characters elided; the full text is in the dossier and is digest-pinned] ...

dow 369.
Record 1540: the coastal reservoir in the pump house was inspected by team E during window 228.
Record 1541: the eastern manifold in the pump house was inspected by team D during window 728.
Record 1542: the outer reservoir in the west yard was counted by team C during window 952.
Record 1543: the spare lantern in the relay hut was re-weighed by team A during window 659.
Record 1544: the amber spindle in corridor three was indexed by team F during window 460.
Record 1545: the spare culvert in the pump house was rotated by team D during window 167.
Record 1546: the steady ledger in the sorting shed was sealed by team F during window 618.
Record 1547: the eastern pallet in the pump house was counted by team F during window 829.
Record 1548: the hollow conduit in the sorting shed was indexed by team B during window 752.
Record 1549: the steady pallet in dock nine was indexed by team B during window 336.
Record 1550: the outer lantern in the west yard was sealed by team F during window 559.
Record 1551: the level annex in the pump house was swapped by team A during window 228.
Record 1552: the outer conduit in corridor three was logged by team A during window 628.
Record 1553: the second carriage in dock nine was inspected by team D during window 656.
Record 1554: the spare carriage in store twelve was re-weighed by team C during window 537.
Record 1555: the narrow culvert in dock nine was indexed by team E during window 480.
Record 1556: the spare culvert in dock nine was rotated by team B during window 523.
Record 1557: the coastal culvert in bay eleven was re-weighed by team B during window 547.
Record 1558: the second culvert in the relay hut was logged by team F during window 688.
Record 1559: the narrow cistern in the west yard was swapped by team C during window 308.
Record 1560: the amber annex in store twelve was sealed by team A during window 866.

Question: Who holds the master key for the granary at the end of this material? Use fewer than 30 words.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): UPDATE 2 governs: Desmond Uhl

**Response** (13 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Desmond Uhl holds the master key for the granary.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'Desmond|Uhl' — ok
  * `forbidden_any` 'Marta|Okonjo' — ok
  * `max_words` 60 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

## Emit-zero floor

* passed: **True**
* min terminal ‖h‖: 146.68763732910156
* min logit sd: 1.6935893297195435
* distinct argmax tokens over 8 probes: 7
* definition: finite, non-zero terminal hidden state and non-constant finite logits on every probe.  When a reference terminal-state tensor is supplied the sealer's own form -- relative L2 < 1.0 against it -- is computed as well and BOTH are published.  A parent has no parent, so the relL2 form is null on a calibration run and is not faked.

