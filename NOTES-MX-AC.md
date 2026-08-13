# NOTES-MX-AC — wave 27: the number nobody built, and two claims withdrawn

Wave 27, branch `codex/lrc-blank-compression`, source commits `49770e9f` (the
wave-27 tree, the sealer's allowlist line and `bodyq.sh`) plus the sealing commit
for this note. `NOTES-MX-AB` is wave 26; the series is contiguous again.

## 1. What the wave was for, and what it actually found

Wave 26 sealed a 4,096-row "shelf" of three 13/15 rungs and asked wave 27 to
measure two things about it: whether an excision costs the same on every rung
(the identity matrix), and whether the shelf extends below 18,432 (the edge
probes). Both were answered. The edge probe also found something nobody was
looking for.

**`ARCH.V19` — vocab_size 19,456, zero solve, 0.794570x — reads 14/15 with zero
cap hits.** That is the highest review in twenty-seven waves and **two cases above
the parent**. It sits between two rungs that read 13, so the shelf is not a
plateau: it is a **peak at 19,456 with 1,024-row shoulders**, 11 / 13 / 14 / 13 /
13. Both interior rungs of the band had simply never been built.

I want to be exact about what this is and is not. It is a 24% smaller model that a
frozen sixteen-prompt review reads two cases better than its parent, on one
reading, by one reviewer. It is **not** evidence that folding made the model
smarter. The parent flubs several of these lanes itself, and a child that fixes
one by accident collects a case. What the row does retire is the idea that the
13/15 band is flat and therefore a measurement artifact of one lucky table: a
1,024-row step in either direction costs a case, which is the behaviour of a real
optimum, not of a plateau.

## 2. The question the wave was sent to answer

**The excision-composition cost belongs to the BLOCK, not to K.** Three blocks ×
two shelf K, all at identical parameter counts and identical ratios:

* block 13 costs **3** cases at K = 18,432 and **3** at 20,480
* block 7 costs **4** at both
* block 9 costs **3** and **4**

So the block sets the cost to within one case and K moves it by at most one. It
holds off the shelf too: `V9X13` reads 6 against V9's 9, block 13 costing three
again. And wave 26's open puzzle — why the same excision cost 1 case on V32 and 3
on V20 — dissolves: the *base* changed, not the block. V32 reads 11 and lands at
10; V18 and V20 read 13 and land at 10. On any shelf-or-better base, block 13
lands the child at ten.

`ARCH.V18X13` and `ARCH.V18X09` are the **deepest ten in the campaign** at
0.737484x, from two different blocks.

## 3. Two claims from earlier waves, withdrawn

**Prompt-echo is not a width signature.** Wave 26 saw it three times, all on width
children, and wrote it down as a property of the width axis. Wave 27 produced four
more sightings on children with **no width fold at all** — `V18X13X01` and
`V20X13X01` are vocabulary coresets on a double block excision at hidden size 1024
and each returns the prompt verbatim on four of sixteen lanes. The width reading
was an artifact of which children had been built deep enough. Prompt-echo is a
symptom of severe damage from any axis.

**λ = 0.3 is not "the rung that never hurts".** Wave 26's four ladders made it the
peak or joint peak everywhere. Wave 27 built six more arms and 0.3 gained on
**none** of them and lost on four. Two rows are worth naming: `V20X13X01G03`
improved its own held-out terminal-logit objective by 23% and lost **two** review
cases; `V20X13KV6JQG03` improved it by 26% and took the cap count from three lanes
to five. The objective and the behaviour move in opposite directions, measured,
twice, in one wave. The wave-26 LIST→SET regression on `coding-deduplicate` also
reproduced on a third base, which makes it a property of the terminal-gain solve
rather than an accident of one child.

## 4. The rest of the matrix

The head-merge axis had never been paired with a vocabulary base. It is fatal:
all three pairings read **2/15**. That gives three second folds priced on one base
— block excision **−3**, head merge **−11**, width **−11** — and once the merge has
taken the child, K stops mattering: seven of sixteen lanes are *byte-identical*
between `V20X13KV6JQ` and `V22X13KV6JQ`. The same inertness appears on the healthy
side (six identical lanes between `V18X07` and `V20X07`), which is the honest
frame for the whole vocabulary axis at these depths: inside the shelf, a
2,048-row change to the token table is very nearly a no-op on the text.

Two depth records at the bottom. `ARCH.V20CL896` reads 2/15 at **0.487566x** —
under half the parent, fifteen points deeper than the previous child to hold two.
`ARCH.V18X13X01KV6JQ` is the **first 0/15 in twenty-seven waves**, and it is
informative rather than merely dead: every other collapsed child in this wave,
down to 0.4876x, still answers **summarisation**, usually in full. This one does
not. Summarisation is the last lane to go.

## 5. Three things that went wrong

* **A sealed body did not reproduce, and the guard caught it.** `bodyq.sh` was
  written to rebuild wave 17's `ARCH.F1301KV6Q` and *check* the rebuild against the
  sealed `architecture.child_tensors_sha256` rather than trust it. It refused:
  identical parameter count, different digest. A second rebuild was **byte-identical
  to the first**, so this is code drift between wave 17's builder and wave 27's on
  the joint-solve path, not nondeterminism. Wave 26 proved the *pure excision* path
  rebuilds bit-exactly across nine waves and I had quietly generalised that to
  "excision bodies reproduce"; they do, and **solved bodies do not**. The composed
  child cites its own rebuilt body and not the sealed cell.
* **Two λ arms were refused by an identity gate and left refused.** The
  terminal-gain objective requires the child's token table to *be* the parent's rows
  at the injected ids; a width fold projects the embedding and destroys that. So
  `V18W896G03` and `V20X13W896G03` cannot exist as posed. The gate is protecting a
  real precondition and I did not route around it — two numbers produced by
  disabling it would have meant nothing.
* **Wave 26's scorecard mis-described the unresolved ledger rows** as four
  `CAL.C01.T5.X4` rows. Three are; the fourth is `COMP.MC.MXO0`. Confirmed by
  re-running the verifier against this wave's pre-merge backup — wave 27 introduced
  nothing, but the description was wrong and is corrected here.

## 6. Estate

Two GPUs throughout: independent seal+review queues on sm86 indices 0 and 4, three
concurrent CPU vocabulary batches feeding them, benchmarks split the same way.
Twenty-three children and twenty-four benchmark rows in **≈75 minutes** of wall
clock against ≈150 serial — about **1 h 15 m** saved, close to ideal because
nothing in the review path contends. Sealing and the ledger merge were kept
single-writer and serial on purpose.

Ledger 724 → **770 rows**, backed up with sha + `cmp` first and byte-prefix
verified after, **and committed** — the thing wave 25 forgot and wave 26 had to
repair. 23 cell roots audited, `wave26/cells` still declared. Package built
**before** retention, so all 25 build receipts survive in it; wave 26 lost three
that way. Nine of twenty-five checkpoints retained (2.9 GiB), sixteen dropped with
reasons. Deployment verified before *and* after the wave: 135 undeclared files,
all `__pycache__`, exactly as wave 26 recorded.
