# NOTES-MX-J — the assembly wave: what survives being put together

**Wave 6 (MX-J), sealed 2026-08-11. Behaviour-first grading under ADDENDUM 6.**
Wave 5 took every component the campaign could measure to a true storage-defined
`S4` and read the whole model back. Five components cleared it inside the
parent's own quality band. This wave does the thing the campaign existed to do:
installs those five **one at a time**, running the full frozen 15-prompt review
after **every** addition, and escalates when one of them breaks the child.

Thirteen composed children, thirteen sealed ADDENDUM 6 verdicts, twenty-six new
ledger rows. **Two children hold the parent's band. One more holds the
alive-degraded band. Ten do not.**

**The headline is not the compression number. It is which component broke it.**

---

## 1. The frontier

Every child on every path, with the two numbers that matter. `net stored ratio`
is the composed child's own **whole-model** unique-parameter count over the
parent's 229,693,184 — not a per-component ratio.

| child | components folded | review | verdict | net stored ratio | Δ params |
|---|---|---:|---|---:|---:|
| `COMP.MA.MXJ1` | C13 @L05 | **12 / 15** | **alive-parent-band** | 0.991439× | −1,966,465 |
| `COMP.MA.MXJ2` | + C07 @L12 | **11 / 15** | **alive-parent-band** | **0.989725×** | **−2,360,201** |
| `COMP.MA.MXJ3` | + C05 @L05 | 6 / 15 | impaired | 0.986297× | −3,147,402 |
| `COMP.MA.MXJ4` | + C11 @L00 | 4 / 15 | impaired | 0.977724× | −5,116,554 |
| `COMP.MA.MXJ5` | + C06 @L12 | 4 / 15 | **dead** (degeneracy) | 0.974301× | −5,902,986 |
| `COMP.MA.MXJ3A` | C05 retried at T3 | 5 / 15 | impaired | 0.986301× | −3,146,633 |
| `COMP.MA.MXJ3B` | C05 retried at T4 | 7 / 15 | impaired | 0.986301× | −3,146,633 |
| `COMP.MB.MXJ3` | step 3 under M-B | 6 / 15 | impaired | 0.986297× | −3,147,402 |
| `COMP.MA.MXJ3C` | **C05 dropped**, + C11 | **9 / 15** | **alive-degraded** | **0.981152×** | **−4,329,353** |
| `COMP.MA.MXJ4C` | + C06 @L12 | 8 / 15 | **dead** (degeneracy) | 0.977728× | −5,115,785 |
| `COMP.MA.MXJP2A` | C13 also @L00 | 5 / 15 | **dead** (degeneracy) | 0.981163× | −4,326,666 |
| `COMP.MA.MXJP2B` | C13 also @L13 | 7 / 15 | impaired | 0.981163× | −4,326,666 |
| `COMP.MA.MXJP2C` | C13 @L00+L05+L13 | 4 / 15 | **dead** (degeneracy) | 0.972602× | −6,293,131 |

*Parent control, same process, every dossier: **12 / 15**.*

**The best child is `COMP.MA.MXJ2`** — two components folded at a real quarter,
**11 of 15 against the parent's 12**, band delta −1, net **0.989725×**.

**The best child at or above the A9 floor is `COMP.MA.MXJ3C`** — three
components, **9 of 15**, net **0.981152×**. It is the deepest fold in the wave
that is still a usable model, and it is the one that drops `conv_out_proj`.

---

## 2. The headline: the component with the *lowest* local drift is the one that destroys the composition

`C05 conv_out_proj` is, on the wave-5 evidence, the campaign's **best**
component. `NOTES-MX-I` §3.1 ranks it first — `conv_out_proj > attn_k >
mlp_down > attn_q > mlp_gate` — and it is the only component that is
`alive-parent-band` under **all six** techniques that can express it. Three
separate C05 cells are sealed at 12/15 in isolation.

Adding it to a working child costs **five review cases**, 11 → 6, and takes the
child out of the alive band entirely.

And it does that while producing the **smallest local drift of any single
install in the wave**:

| component installed alone | local terminal drift vs control | isolation review |
|---|---:|---:|
| **C05 conv_out_proj @L05** | **0.181179** | 12 / 15 |
| C11 mlp_gate @L00 | 0.246692 | 11 / 15 |
| C13 mlp_down @L05 | 0.272342 | 12 / 15 |

The diagnostic that the campaign has used since wave 1 to localize damage ranks
C05 as the *least* damaging thing it can do to the model, and the frozen review
says C05 is the only thing in this wave that cannot be composed at all. This is
the campaign's oldest finding — drift and behaviour are not measuring the same
thing — arriving at its sharpest form: **they do not merely disagree in
magnitude, they disagree in rank order, and the metric's best-ranked component
is the behavioural killer.**

### 2.1 The escalation chain was walked to the end at C05, and only dropping it worked

ADDENDUM 6's escalation is (a) next-best technique, (b) M-B conditioned
re-solve, (c) drop. All three were run.

| branch | what changed | review | net stored |
|---|---|---:|---:|
| baseline | `C05.T8.S4.L05`, the wave-5 winner | 6 / 15 | 0.986297× |
| **(a)** | `C05.T3.S4.L05` — balanced basis truncation, also sealed 12/15 | **5 / 15** | 0.986301× |
| **(a)** | `C05.T4.S4.L05` — low-rank factor writers, also sealed 12/15 | **7 / 15** | 0.986301× |
| **(b)** | M-B, conditioned re-solve of everything downstream | **6 / 15** | 0.986297× |
| **(c)** | **drop C05 entirely, install C11 instead** | **9 / 15** | **0.981152×** |

**All three of `conv_out_proj`'s alive-parent-band techniques fail in
composition** — 6, 5 and 7 of 15 — so this is a property of the component, not
of the solver, which is exactly the shape `NOTES-MX-I` §3.1 predicted for
compressibility and which turns out to hold for *composability* too.

And dropping it is **strictly better on both axes at once**: 9/15 against 7/15,
and 0.981152× against 0.986301×. The child that folds *fewer* components stores
*less*, because the component it gave up is small (1,048,576 parent parameters)
and the one it bought with the slot is large (2,621,440). There was never a
trade to make here.

### 2.2 What C05 actually does to the text

Between `MXJ2` (11/15) and `MXJ3` (6/15) the only difference is C05, and five
specific cases fall:

* `summarization-priorities` stops naming the decision at all — it summarises
  the supplier comparison and never says the team selected Supplier B.
* `comparison-ram-rom` returns a backpack analogy that contradicts itself in
  adjacent clauses ("the backpack bag is much heavier and more bulky than the
  backpack bag itself") and states its two required imperfections as the same
  sentence twice.
* `coding-deduplicate` emits `seen.add(unique_order)` — a `TypeError` on an
  unhashable list. The function does not run.
* `creative-continuity` collapses to a single sentence with **no dialogue line**,
  which the prompt requires.
* `arithmetic-multi-step` loses the five boxes: `94 / 1 = 94 boxes`.

---

## 3. The two kills, and why the count could not see either

Two children are sealed `dead` on the degeneracy clause **at counts that would
otherwise have been alive**. This is the third wave running in which the clause
earns its keep, and both cases are the `NOTES-MX-I` §3.3 standard verbatim — a
**response that is the loop**, not local phrase doubling inside progressing
text.

| child | count alone would have said | why it is dead |
|---|---|---|
| `COMP.MA.MXJ5` | impaired (4/15) | `writing-concise-email` emits the same two-sentence email **five times verbatim** plus four stray `Subject:` lines and nothing else; `planning-dependencies` emits its whole six-step plan twice verbatim; `coding-deduplicate` writes `seen = set()` four times consecutively then recurses infinitely |
| `COMP.MA.MXJ4C` | **alive-degraded (8/15)** | `creative-continuity` emits `"Did you hear the fog?"` **three times verbatim** and its closing narration sentence twice; eight sentences, four of them repeats, and nothing happens between the first and the last |

`COMP.MA.MXJ4C` is this wave's `C08.T8.S4.L02`. Eight of fifteen is inside the
alive-degraded band; the model cannot write a paragraph. A count cannot see
that, and the clause is what stops an 8/15 loop from being reported as a
survivor.

The test applied, stated so it can be overturned deliberately rather than
drifted: **remove the repetitions and ask whether a response remains.** For
`MXJ4C`'s creative lane nothing remains. For `MXJ3C`'s coding lane — eight
consecutive `seen.add(None)` inside a response that still carries a function, an
explanation and a complexity claim — something does, and that child is sealed
alive at 9/15 with the repetition recorded rather than charged.

---

## 4. M-A vs M-B at S4: conditioning still buys nothing the review can see

ADDENDUM 4's remaining question, asked again one storage level down. Wave 4
answered it at M16: +0.041 free-run on the 7-component child, **zero** review
cases. Wave 6 asks it where the fold is four times deeper.

| | M-A | M-B |
|---|---:|---:|
| **frozen review** | **6 / 15 impaired** | **6 / 15 impaired** |
| global drift | **0.369708** | 0.378189 |
| composed free-run (diagnostic) | **0.132813** | 0.070313 |
| net storage | 0.986297× | 0.986297× |
| conditioned captures | — | 2 |
| wall clock | **63.6 s** | **1,743.5 s (27.4×)** |

**Same band, same count, worse on both composed diagnostics, twenty-seven times
the wall clock.** Per step:

| step | M-A drift / free-run | M-B drift / free-run | conditioned |
|---|---:|---:|---|
| `C05.T8.S4.L05` | 0.181179 / 0.1562 | 0.181179 / 0.1562 | **no** — first in topological order |
| `C13.T8.S4.L05` | 0.317372 / 0.0586 | **0.313035** / **0.0703** | yes |
| `C07.T5.S4.L12` | **0.369708** / **0.1328** | 0.378189 / 0.0703 | yes |

Three things are worth keeping.

**The solve genuinely moves.** Conditioned `C13` reports gram 0.17850 / cos
0.92790 against M-A's 0.20058 / 0.92045 — this is not a re-run that landed on
the same numbers, which is what most of wave 4's conditioned steps did. At S4
the evidence a component sees really is different.

**It moves in both directions and cancels.** `C13` improves under conditioning
and `C07` gets worse, and the composed child ends level. The *text* is
substantially different — M-B writes the wave's best `planning-dependencies`
answer (six steps, all five requested topics, a real rollback) and the only
correct dedup function any C05 branch produced, and pays for both by losing
*Saturday* in `context-retrieval` and the decision in `summarization`. Two
upgrades, two downgrades, same 6/15.

**The asymmetry this cell could not design around, stated rather than hidden.**
The topological install order is layer-ascending, conv before attention before
MLP, so `C05` at L05 installs **first** — and the component that actually broke
the composition is therefore the one M-B is structurally unable to condition.
M-B conditions what is *downstream* of the damage. A cell that could condition
C05 would need C05 to install after something, which the pre-registered order
forbids.

**A9 conclusion unchanged, one level down: conditioning is a second-order
correction to a first-order law, and at S4 it does not even reliably have the
right sign.**

---

## 5. Phase 2: the site ladder, and why it is only three rungs long

The brief asked for `C13 T8` across L00–L13. **That is not expressible in this
campaign**: `matrix_cell_receipt.COMPONENTS` registers `C13`, `C11`, `C05`,
`C12` at sites `(0, 5, 13)` and `C06`–`C09` at `(2, 8, 12)`, and
`_validate_cell` enforces it. A fourteen-layer ladder would mean editing the
receipt schema mid-wave, which is the one thing the standing rules forbid, so
the ladder is three rungs per component and is reported as three.

`C13 mlp_down` expanded on top of `COMP.MA.MXJ2`, the deepest alive-parent-band
child:

| sites folded | review | verdict | net stored | local drift at the new site |
|---|---:|---|---:|---:|
| L05 *(the phase-1 child)* | 11 / 15 | alive-parent-band | 0.989725× | 0.272342 |
| L05 + **L00** | 5 / 15 | **dead** | 0.981163× | **0.422844** |
| L05 + **L13** | 7 / 15 | impaired | 0.981163× | 0.380073 |
| L05 + L00 + L13 | 4 / 15 | **dead** | **0.972602×** | 0.539868 |

**Both expansion sites refuse**, so the back-off rule stops the component after
two consecutive failures and no other component's ladder was climbed. `L00` is
much the worse of the two — the L00 install alone drifts further (0.422844) than
the entire three-component phase-1 step-3 child did (0.369708) — which is the
same S3 layer ordering the campaign has seen since wave 1, now measured inside a
composed child.

`C05` was deliberately **not** expanded. Phase 1 measured it as the one
component that cannot compose at a single site; widening it would have spent the
budget on the known failure.

`COMP.MA.MXJP2C` is the honest bottom of the wave and worth stating plainly: it
is the **best storage number any child reached** — 0.972602×, 6,293,131
parameters removed — and it emits the same two-line email nine times, terminates
`arithmetic` by printing `\boxed{1}` about twenty times into the 1,024-token
cap, and repeats the word `clean` roughly fifty times in the safety lane.

### 5.1 What the global number can be, and why it is small

Even a hypothetical child that folded all five alive-parent-band components at
all three of their registered sites at a true quarter would store

> 23,592,960 parent parameters → 5,884,002 stored → net **0.9229×**

because 67 M of the parent's 229,693,184 is the tied embedding table that no
component in the matrix touches, and because eleven of the fourteen layers have
no registered site at all. **The single-site assembly ceiling is ~0.974× and the
all-registered-sites ceiling is ~0.923×.** A whole-model 4× is not reachable
from this component set at these sites, and no arrangement of these cells gets
close to it. That is a statement about the campaign's site policy, not about the
technique surface.

---

## 6. The harness reproduced the isolation runner to the digit

Step 1 is one component in an otherwise-parent model, which is exactly what a
wave-5 isolation cell is, so it is a free calibration and it was spent as one.
`COMP.MA.MXJ1` against sealed `C13.T8.S4.L05`:

| | isolation cell (wave 5) | composed step 1 (wave 6) |
|---|---|---|
| calibration relL2 | 0.0006382105222036804 | **identical** |
| solve | rank 639, gram 0.20058, cos 0.92045 | **identical** |
| stored | 654,975 / 2,621,440 = 0.249853× | **identical** |
| free-run | 0.06640625 | **identical** |
| **16 frozen-review responses** | — | **byte-identical** |
| **16 parent-control responses** | — | **byte-identical** |

Because the child's text is byte-identical, step 1 carries that cell's sealed
per-case grading rather than a re-grade: re-reading identical text could only
differ by reviewer noise, and pretending otherwise would manufacture an
independent reading that does not exist. Every other child in the wave was
graded from its own dossier.

The parent control was calibrated **before any child was graded**, from the
step-1 dossier, and reproduced wave 5A and 5B exactly: **5 useful / 7 partial /
3 observed miss = 12/15**, with the misses at `arithmetic-multi-step`,
`logic-ordering` and `multi-turn-correction`. Its sixteen responses are
byte-identical in every wave-6 dossier.

---

## 7. What this wave did not settle

* **The other four site ladders were not climbed.** `C11`, `C05`, `C06` and
  `C07` each have two unmeasured registered sites. The C13 ladder failed at both
  of its, which is one component's evidence and not a law.
* **The degraded bench was never reached.** `C12 mlp_up` (9/15), `C09 attn_o`
  (9/15) and `C08 attn_v` (8/15) were queued behind the APB core and the core
  never got deep enough to earn them. Nothing in this wave measures them in
  composition.
* **`C11.T1.S8.L00` was not composed.** The storage-vs-behaviour branch at
  mlp_gate — 8× storage at 10/15 in isolation — is a cell this wave designed and
  did not run, because the T8 branch it was meant to be compared against was
  already inside a child that had left the alive band.
* **M-B was run once.** One M-A/M-B pair at one recipe is one comparison; wave 4
  ran two and got two different answers.
* **The frozen review is one reviewer, one pass.** ADDENDUM 2's ±2 is the
  measured reviewer-sensitivity interval on this surface. `COMP.MA.MXJ2`'s −1 is
  inside it; `COMP.MA.MXJ3`'s −6 is not, and that is the only reason the C05
  finding can be asserted at all.
* **No claim about a site that was not folded**, per ADDENDUM 5 rule 2-S.

---

## 8. Ledger, receipts, disk

| what | where |
|---|---|
| ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.pre-mxj-20260811T161256` (`e212e7f6…`) |
| ledger before the merge | `…/backup/ledger.jsonl.before-mxj-merge-20260811T171810` (`e212e7f6…`) |
| `LEDGER.md` before the render | `…/backup/LEDGER.md.before-mxj-20260811T171935` |
| pruned heavy artifacts | `reports/matrix-campaign-20260810/wave6/pruned-tensors-manifest.json` |

**The ledger.** 322 → **348 rows (+26)**: 13 composed-child instances and 13
review instances. `sha 5fc36c1b41489da7…`, **348 distinct
`(cell_instance_id, receipt_sha)` pairs**, 348 distinct receipt digests, **26 of
26 wave-6 rows on `sm86 NVIDIA GeForce RTX 3090`**. The merge refused-unless-
prefix check passed byte for byte, and the merger was run exactly once, because
`NOTES-MX-H` §9 records that it is not idempotent.

**Seals.** 344 of 348 ledger rows resolve to a receipt on disk and **all 344
verify; 0 fail**. All **26 of 26** wave-6 seals verify. 283 receipts satisfy
today's mandatory field set and **61 wave-1 receipts do not**, for the
documented reason — they predate `code_identity.compute_architecture` — which is
the same 61 wave 4 reported.

**Four ledger rows do not resolve to a receipt, and none of them is wave 6's:**
three wave-1 `CAL.C01.T5.X4` calibration rows that sealed an infra failure
without persisting a receipt, and one wave-5 `C06.T2.S4.L12` review instance
whose file was overwritten by a second seal 21 seconds later. That second one is
also the ledger's **only** duplicated `cell_instance_id`
(`1cf2c017335b4b34…-a6review`, two distinct receipt digests). Wave 6 found both
by checking and neither by causing; they are recorded here rather than fixed,
because rewriting a sealed row to tidy an append-only ledger is worse than the
untidiness.

**Mirror.** `reports/matrix-campaign-20260810/ledger.jsonl` and `LEDGER.md` are
**byte-identical** to the authority:

```
5fc36c1b41489da7d87a3771280d4d2f0dc102f668c484b8167b7c5a320276cd  ledger.jsonl
d1a251845770883ee8be4075492417a98eb4c629c1c87cbc99be0d16717ef5f5  LEDGER.md
```

**Retention.** Five children's composed tensors are kept — `MXJ1` (the
calibration anchor), `MXJ2` (deepest alive-parent-band), `MXJ3C` (deepest
alive-degraded), and the `MXJ3` M-A/M-B pair, which is worthless with one arm
missing. **116 files, 17.76 GB** — eight children's tensors and both 8 GB
conditioned captures — were sha256-digested into
`wave6/pruned-tensors-manifest.json` and then deleted. Every entry carries the
sentence that says how to re-derive it.

**Disk.** Root `/` was **7.2 GB free at the start of the wave and 7.2 GB free at
the end** — every wave-6 artifact, every new evidence cache and both conditioned
captures were written under `/mnt/ai_data/neural-forge-matrix/wave6/`, and the
already-built site evidence caches were reached through symlinks so that reuse
cost nothing and no write landed on the root filesystem.

## 9. Provenance

**No new deployment was built, deliberately.** `git diff eb32153e 217458eb --
src experiments docs research` is **empty** — the three commits since wave 5's
deployment touched only `reports/` — so
`/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-eb32153e` *is*
the technique surface that produced the survivors this wave assembles, and
re-deploying could only have introduced a difference. It also sidesteps the
known `git archive` re-lining wart: the deployment's
`V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md` was verified in place at `e5e37768…`
before a single token was generated, with the prompt pack at `0ee6bf14…`, and
neither file was written to.

Review surface, digest-pinned in every one of the thirteen dossiers: protocol
`e5e37768…`, prompt pack `0ee6bf14…`, greedy decoding at the 1,024-token cap,
the parent decoded on the same sixteen prompts **in the same process** as every
child. GPUs `sm86 NVIDIA GeForce RTX 3090` indices 0 and 4, pinned by PCI bus id
in the shell preflight **and** re-checked by `--require-compute-architecture
sm86` inside every sealing process. Evidence base `2dec87ac…`, opened read-only.
Knob file byte-identical to wave 5's.

Every verdict was computed by `behavior_verdict_for` from counts the reviewer
sealed through `seal_matrix_behavior_review.py`; no verdict in this wave was
stated by hand.

## 10. Paths

* composed children: `reports/matrix-campaign-20260810/cells/COMP.M{A,B}.MXJ*/`
  — `cell-receipt.json`, `cell-receipt-behavior-reviewed.json`,
  `cell-diagnostics.json`, `composed-child-state-digest.json`,
  `frozen-review-dossier.json`
* recipes, gradings and run scripts:
  `reports/matrix-campaign-20260810/run/mxj/`
* surviving tensors:
  `/mnt/ai_data/neural-forge-matrix/wave6/cells/{COMP.MA.MXJ1,COMP.MA.MXJ2,COMP.MA.MXJ3,COMP.MA.MXJ3C,COMP.MB.MXJ3}/composed-child-tensors.pt`
* prune manifest: `reports/matrix-campaign-20260810/wave6/pruned-tensors-manifest.json`
