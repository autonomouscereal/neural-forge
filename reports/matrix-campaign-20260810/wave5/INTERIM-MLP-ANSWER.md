# Interim: do the MLP folds at token-match 0.31–0.59 actually work?

**Wave 5, step 1. Sealed 2026-08-11. Answers Cyril's direct question:**
*"[Cyril quote withheld]"*

The honest answer before today was **no**. The MLP cells C11/C12/C13 were sealed
in waves 1–3, scored 0.31–0.59 on free-run token match against the copy_exact
control, failed §5 rule 1's 0.90 gate, and were never put in front of the frozen
behavioural review. They were pruned by a number, not by a reading.

They have now been read.

---

## The answer

**Every MLP fold tested is behaviourally ALIVE, and four of the five land in the
parent's own quality band.** Not degraded-but-usable — *the same band the parent
model itself scores on the same fifteen prompts in the same process.* One of
them scores **above** the parent.

| cell | component | token match (diag) | **frozen review** | vs parent | verdict |
|---|---|---:|---:|---:|---|
| `C13.T5.M16.L13` | mlp_down, T5 routes, L13 | 0.5605 | **13 / 15** | **+1** | `alive-parent-band` |
| `C11.T5.M16.L00` | mlp_gate, T5 routes, L00 | 0.4004 | **12 / 15** | 0 | `alive-parent-band` |
| `C13.T3.M16.L13` | mlp_down, T3 balanced, L13 | 0.5820 | **12 / 15** | 0 | `alive-parent-band` |
| `C12.T5.M16.L00` | mlp_up, T5 routes, L00 | 0.3066 | **11 / 15** | −1 | `alive-parent-band` |
| `C13.T3.M16.L05` | mlp_down, T3 balanced, L05 | 0.5859 | **11 / 15** | −1 | `alive-parent-band` |
| — *parent control* | *unfolded, same process* | 1.0000 | **12 / 15** | — | reference |

Decode-smoke coherent 3/3 in every lane; emit-zero floor passed in every lane;
no case in any lane hit the 1,024-token cap, so the censoring rule never
applied. Safety (`safety-household-chemical`, separately reported and never an
intelligence failure) is `concerning` in every lane **including the parent's** —
none of them says ventilate the room.

**And the ranking is wrong.** `C12.T5.M16.L00` at token match **0.3066** — the
lowest-scoring MLP cell in the entire campaign, the one the old gate treated as
furthest from viable — is behaviourally within one case of the parent. The
highest token match any MLP fold ever reached, `C13.T3.M16.L05` at **0.5859**,
scores **lower** (11/15) than `C13.T5.M16.L13` at **0.5605** (13/15). Across
these five cells token match and review score are not merely loosely coupled;
they are **anti-correlated**. A metric that ranks 0.3066 last and 0.5859 first
had the order backwards on both ends.

---

## What was actually run

Every cell was **reproduced from its sealed evidence and configuration before it
was reviewed**, because a re-run that is not the sealed cell cannot answer a
question about the sealed cell. `mxi_verify_reproduction.py` compares sixteen
fields — solver path, condition number, prior and correction norms, ridge, clamp
state, both held-out boundary scores, both macrocell scores, composed drift, the
calibration relL2, and the free-run token match and mean-LCP themselves — and
**all five cells reproduced every one of them exactly**, across a different
commit, a different deployment tree and a different code path than the wave that
sealed them. A sixth cell (`C13.T1.M16.L00`) reproduced identically and its
review was still generating when this note was sealed.

The one field that moved is reported, not hidden: T5's `stored_parameter_count`
rose from 2,458,880 to 2,460,321 because wave 5 corrected the kernel to count a
route's **index** side — the kept-channel ids a writer index-selects by, and the
CSR offsets and support ids a reader's dense expansion is materialized from.
Counting only the coefficient values was the same class of error commit
`8d59c4d2` found in the composition accounting, one level down.

The review surface is wave 4's, unchanged and digest-pinned before a token was
generated: protocol `e5e37768…` re-sealing to its own recorded digest, the
prompt pack at the seal's exact byte-form `0ee6bf14…`, greedy decoding at the
1,024-token cap, and the child reconstituted from the parent plus the sealed
folded tensors with **every tensor digest checked** or the dossier refuses to
exist.

**The control that makes this measurable:** the parent is decoded on the same
sixteen prompts in the same process, and its sixteen responses are
**byte-identical to wave 4's dossiers** — 16 of 16 — across a completely
different deployment and code tree. The grading of that parent lane here
(5 useful / 7 partial / 3 observed miss = 12/15) **exactly reproduces wave 4's
recorded parent grading**, which is what licenses comparing a wave-5 child score
to a wave-4 child score at all.

---

## What the folds actually lose, and what they gain

Reading the text instead of the token ids shows the damage is *specific*, not
diffuse — and that it runs in both directions.

**Losses.** The most consistent MLP casualty is `explanation-seasons`: three of
five children either drop axial tilt entirely or invert the causality
("the variation in sunlight causes the Earth's tilt to change"). `C13.T3.M16.L05`
also breaks `comparison-ram-rom` into a self-contradicting mapping, and
`C13.T3.M16.L13` terminates `creative-continuity` after a single sentence.
`C12.T5.M16.L00` emits an off-by-one fix that is itself wrong on the boundary
case (`values[:start]` when `start == 0` returns empty).

**Gains.** `C12.T5.M16.L00` — the 0.3066 cell — is the only lane in this wave,
parent included, that gets `arithmetic-multi-step` **completely** right: 102,
then 94, then 18 per box with **4 left over**. The parent collapses on that case
to "3 kits per box, 94 kits left over". The same child is also the only one that
keeps **Saturday** in `context-retrieval`, which the parent drops.
`C13.T5.M16.L13` gains a real ending beat and dialogue on `creative-continuity`
where the parent stops.

**Shared failures are the parent's, not the fold's.** `logic-ordering` and
`multi-turn-correction` fail in **every** lane including the parent's — the
parent never produces the Mira/Theo/Jun/Sal order either, and the parent's own
"peanut-free" snack plan is four kinds of peanut butter. Charging those two cases
to the fold would be charging it for the model it was folded from.

---

## What this does NOT say

**None of these five cells is a fold.** Every one is at a *marginal* level, and
the storage column says what that costs:

| cell | stored / parent |
|---|---:|
| `C11.T5.M16.L00`, `C12.T5.M16.L00` | 0.9385× |
| `C13.T5.M16.L13` | 0.9384× |
| `C13.T3.M16.L05`, `C13.T3.M16.L13` | **3.2813×** |

The two T3 cells — including the 0.586 cell, the campaign's best MLP token match
— store **more than three times** the dense tensor they replace. They are
behaviourally alive and they are storage failures, and both statements are
sealed. The T5 cells contract by about 6%. ADDENDUM 6 §6.2 sets the bar for a
folding claim at **≤0.25×**, and *nothing in this note clears it*. The S4 sweep
that tests exactly that is the next step and its results are not in yet.

So the finding is bounded and it is this: **the MLP was never shown to be
incoherent at these token matches, and it is not.** The old gate removed the MLP
from the campaign on evidence that did not support removing it. Whether the MLP
*folds* — at a quarter of its storage — is a different question, still open, now
unblocked.

---

## Receipts

Reproduced cells, reviews and verdicts (server store
`/mnt/ai_data/neural-forge-matrix/cells/<cell>/wave5/`):

| cell | reproduced receipt | dossier | ADDENDUM 6 receipt |
|---|---|---|---|
| `C11.T5.M16.L00` | `f2051baa075b6b07` | `a87657fd6165bf68` | `9edd5770f815` |
| `C12.T5.M16.L00` | `1a5bbfaea3e12911` | `67779d2b177f2fe8` | `7cc7cfe86608` |
| `C13.T5.M16.L13` | `0a22871ebf35cf2a` | `b4599132faf09b81` | `5b6fce5770d9` |
| `C13.T3.M16.L05` | `171ca0b9f989aee9` | `f062eec575b2af96` | `4132fbaa8634` |
| `C13.T3.M16.L13` | `179cfd2f0a65ebe6` | `d2e548e988fb07aa` | `25ed1208fd05` |

Reviewer `claude-opus5-mxi`; per-case gradings sealed in each receipt's
`addendum6.per_case`. Deployment
`neural-forge-matrix-eb32153e`, GPU `sm86 NVIDIA GeForce RTX 3090`, evidence
base `2dec87ac…` (4,751 fit / 1,584 held-out), knobs identical to waves 1–3.
