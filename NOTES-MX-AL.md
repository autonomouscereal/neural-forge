# NOTES-MX-AL — wave 34: the frontier moves by building, and the artifact clause gets a price

**2026-08-28.** Branch `codex/lrc-blank-compression`. Eight children built, sealed
and read twice. Ledger 920 → **936** rows,
`cff07ea4a165683b2a400ecc6a380db7ff707b82aa55d1f180f511c932b9b93f`, byte-prefix
verified against the pre-wave backup. **Five deployments, three of them forced by
a guard refusing correctly.**

---

## 1. The tree cut is finished, and finishing it found a defect two waves old

Wave 34 was cut on 2026-08-28 and its agent died with three of four `HAND_EDIT`
files outstanding and the README unwritten (`NOTES-MX-AK` §1.3). All four are
done and `cut_wave34.py --audit` passes.

`mx34_build.py` took **nine** identifier references rather than wave 33's eight,
because wave 33 authored one more into its own header; four schema strings; and
wave 33's basis schema **appended** to the append-only accept list.

`sealq.sh` arrived VERBATIM carrying wave 33's marker and wave 33's **six**
terminal-gain seal calls with every count intact, aimed at this wave's cell ids.
The audit refused on the marker before anything ran. **Second consecutive cut
where that protection has fired, and both times it was correct.**

### 1.1 The builder has written the wrong wave's receipt filename since wave 32

This wave is the first since 32 to call `mx*_build.py`. The first thing it
produced was a body at exactly the sealed parameter count — 158,127,872 =
0.688431x, `ARCH.CL896` to the parameter — followed by:

```
FileNotFoundError: .../MX34-CL896/mx34-build-receipt.json
```

The build was fine. The **filename** was wrong. Both export sites write the
receipt under a literal, and that literal has said `mx32-build-receipt.json`
since wave 32. Wave 33's hand edit of this file retargeted eight module
references and five schema strings and did not touch these two lines, and wave 33
built no body, so nothing fired for a whole wave.

The split in who would have noticed is the part worth keeping. Every **glob**
reader in the tree — vocab, termgain, both blends, logitsolve — matches
`mx*-build-receipt.json` and would have run happily forever on a receipt named
after a wave two behind. Every **exact-name** reader — `bodyq.sh`'s reproduction
check, `vocabq.sh`, `gainq.sh`, `package.sh`, and `seal_and_review.sh
--build-receipt` — would have failed, and the last of those is on the sealing
path. A defect only the sealing path can see, in a wave that does not seal, is a
defect that waits.

Both sites now name wave 34 and `--audit` gained an assertion on the **write
expression** at a pinned count of two, plus zero for the stale one. The
expression rather than the string, because the builder's header now names the
stale filename in prose where the next cut will read it.

### 1.2 The second accept list refused all four of the wave's first reviews

There are two append-only accept lists in this estate and wave 34 walked into
both. `BASIS_SCHEMAS_ACCEPTED` in the builder was appended by hand at the tree
cut. The other lives in `experiments/seal_matrix_architecture_child.py` and it
refused every round-A review before a receipt was written:

```
[arch] REFUSED: .../mx34-build-receipt.json declares schema
'neural-forge-matrix-wave34-standard-shape-child-v1', which is not one of
(... thirty-two strings, none of them wave 34's ...)
```

**A wave that has to be refused twice by its own guards to get its schema strings
right is a wave whose guards work.** Both refusals cost a deployment each and
nothing else.

---

## 2. What was built, and how much of it is a reproduction control

| child | ratio | stored | how it was made |
|---|---:|---:|---|
| `ARCH.CL896` | 0.688431 | 158,127,872 | GPU rebuild of wave 15's body, `--ff 896` |
| `ARCH.CL1792X13` | 0.813949 | 186,958,592 | GPU rebuild of wave 31's body, `--ff 1792 --excise 13` |
| `ARCH.V10CL896` | **0.441915** | 101,504,768 | vocab K=10,240 on CL896 |
| `ARCH.V12CL896` | **0.451045** | 103,601,920 | vocab K=12,288 on CL896 |
| `ARCH.V12CL896G01/G03/G10` | **0.451045** | 103,601,920 | terminal gain λ 0.1 / 0.3 / 1.0 on V12CL896 |
| `ARCH.V16CL896` | 0.469305 | 107,796,224 | vocab K=16,384 on CL896 (rebuild) |
| `ARCH.V16CL896G03` | 0.469305 | 107,796,224 | terminal gain λ0.3 on V16CL896 |
| `ARCH.V8CL1792X13` | **0.558303** | 128,238,336 | vocab K=8,192 on CL1792X13 |
| `ARCH.V12CL1792X13` | **0.576563** | 132,432,640 | vocab K=12,288 on CL1792X13 |

Every parameter count landed **exactly** on the pre-registered number.

### 2.1 Three bit-identity results this campaign has wanted since wave 27

Wave 27 refused a rebuild against a sealed digest and could not say why; wave 29
bisected it. This wave rebuilt three sealed artifacts and all three reproduce:

| artifact | sealed by | rebuilt digest | verdict |
|---|---|---|---|
| `ARCH.CL896` body | wave 15 | `949375f4cc736c68…` | **BIT-IDENTICAL, 19 waves later** |
| `ARCH.CL1792X13` body | wave 31 | `3e4ef06fa1ff1dcb…` | **BIT-IDENTICAL** |
| `ARCH.V16CL896` child | wave 23 | `55061febb9795b42…` | **BIT-IDENTICAL, 11 waves later** |

The CL896 body also reproduced **twice within this wave** — once under deployment
`02dd19c6` and once under `082409cb` after the receipt-filename fix — at the same
digest. So the fold path on this parent is deterministic across nineteen waves,
four builders and two deployments, and every child composed on these bodies may
cite its sealed ancestor.

---

## 3. The frontier

### 3.1 The sealed ladder, which is the only one a claim fires on

| rung | before | after | Δ stored ratio |
|---:|---|---|---:|
| 4/15 | `ARCH.V8CL1792` 0.600552 | **`ARCH.V12CL896G03` 0.451045** | **0.149507** |
| 3/15 | `ARCH.V19CL896G03` 0.483001 | **`ARCH.V12CL896G03` 0.451045** | 0.031956 |
| 2/15 | `ARCH.V18CL896` 0.478436 | **`ARCH.V10CL896` 0.441915** | 0.036521 |
| 1/15 | `ARCH.V16CL896` 0.469305 | **`ARCH.V10CL896` 0.441915** | 0.027390 |

The four rung is **the largest single frontier move the campaign has recorded on
either ladder.** Rungs 5 through 14 do not move. The 0/15 floor does not move:
nothing has ever been built between `ARCH.V8CL896` at 0.432784x and
`ARCH.V10CL896` at 0.441915x.

**The stricter reading of `ARCH.V12CL896G03` is in its own seal rather than in a
footnote.** Its concise-email lane is a first-person restatement — *"I'm a polite
email that's asking a vendor to confirm whether a delayed shipment will arrange
by the Friday"* — and not an email. Wave 28 sealed text of the identical shape as
the +1 lane on both `ARCH.V19CL896G03` and `ARCH.V20CL896G03`, so the precedent
is applied and the child seals at four. A reader who requires the lane to BE an
email seals it at **three**, which still moves the three rung from 0.483001x.
Both readings are in the sealed note.

### 3.2 The canonical ladder, and the thing that happened to it

| rung | before | after | Δ |
|---:|---|---|---:|
| 4/15 | `ARCH.V19CL896G03` 0.483001 | **`ARCH.V12CL896G03` 0.451045** | 0.031956 |
| 3/15 | `ARCH.V19CL896G03` 0.483001 | **`ARCH.V12CL896G03` 0.451045** | 0.031956 |
| 2/15 | `ARCH.V16CL896` 0.469305 | **`ARCH.V10CL896` 0.441915** | 0.027390 |
| 1/15 | `ARCH.V19X13CL896` 0.452734 | **`ARCH.V10CL896` 0.441915** | 0.010819 |

**At every rung from one to four the canonical frontier child and the sealed
frontier child are now the same child at the same ratio.** That has never been
true before. Seven of the eight new children read the same under both standards,
and the eighth differs by one lane. The re-read is not buying depth at the bottom
any more; the builds are.

The ordered target was **5+/15 canonical below 0.48x** and it was **not reached**.
The best is four, at 0.451045x — a rung deeper than the incumbent four, but four.

### 3.3 What each child holds

| child | ratio | sealed | canonical | art | con | pre-reg sealed | pre-reg canon |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ARCH.V10CL896` | 0.441915 | **2** | **2** | 11 | 2 | 1.6 | 0 |
| `ARCH.V12CL896` | 0.451045 | **2** | **2** | 6 | 7 | 2.6 | 1 |
| `ARCH.V12CL896G01` | 0.451045 | **2** | **2** | 9 | 4 | 2.6 | 1 |
| `ARCH.V12CL896G03` | 0.451045 | **4** | **4** | 8 | 3 | 3.6 | 3 |
| `ARCH.V12CL896G10` | 0.451045 | **2** | **2** | 8 | 5 | 2.6 | 1 |
| `ARCH.V16CL896G03` | 0.469305 | **3** | **3** | 6 | 6 | 2 | 4 |
| `ARCH.V8CL1792X13` | 0.558303 | **1** | **2** | 6 | 7 | 0 | 2 |
| `ARCH.V12CL1792X13` | 0.576563 | **2** | **2** | 7 | 6 | 0 | 5 |

Six of eight sealed counts landed within the allocator's ±2.3 error bar; the two
that did not (`V12CL1792X13`, `V8CL1792X13`) beat it, which is the direction a
"triples die" law is supposed to fail in.

---

## 4. The results, in the order they matter

### 4.1 The terminal gain buys two cases, at the deepest point it has ever been run

The λ ladder on `ARCH.V12CL896` reads **2 / 4 / 2 / 2** for control / 0.3 / 0.1 /
1.0. λ0.3 is the only rung that moves and **it is the first time in eight ladders
that it moves UP by more than one.** It buys the concise email — the lane wave 28
recorded the gain buying on two other CL896 bases — and it buys
**context-retrieval**, which no CL896 gain arm has ever bought: the control loops
and the arm returns *"The latest checkout time is 4:15 p.m."*

The identity gate passed on a vocabulary child, as wave 28 predicted it would for
an `--ff` body (the MLP fold never touches the embedding, so the gather keeps the
parent's rows verbatim).

λ0.1 is the counter-example that keeps the ladder honest: the **largest** update
(0.1121) and the **best** held-out objective (relSSE 0.2004 → 0.1907), six cap
hits against the control's three, and it reads the control's two. Its retrieval
lane is the near-miss of the wave — *"the latest checkout time is 4:15
MINUTES"*, the right number in the wrong unit. That is the fifth wave to record
an objective improving while the review does not move.

`ARCH.V16CL896G03` closes the family: the terminal-gain arm on the CL896 body
reads **three at K = 16,384, 19,456 and 20,480 and four at 12,288**. Four K
values, one count. The gain is a property of the body; the extra lane at 12,288
is a property of that base.

### 4.2 The vocabulary axis on CL896 is flat over 8,192 rows

`ARCH.V12CL896` reads two on the same two lanes as every CL896 child from
K = 12,288 to K = 20,480 — the whole-prompt summarisation echo and the four
withheld quantities. That is the fourth wave to record the flatness and the
widest span it has been recorded over.

`ARCH.V10CL896` extends it one rung further at 0.441915x and **still holds two**,
against a prediction of 1.6. It was ordered to bracket a content death rather
than to take a rung: `ARCH.V8CL896` at 0.432784x is one of only three honest
content deaths in the estate and nothing had ever been built between it and
0.469305x. **The bracket is now 0.0091 of stored ratio wide** — 8,192 to 10,240
rows — and the death is still not located.

Its shape is the interesting part: **11 of 15 lanes artifact, 2 content.** At
0.4419x the decode has collapsed and the content has not.

### 4.3 C2 is falsified, in the direction it was ordered in

The wave was ordered to build `V12CL1792X13` and `V8CL1792X13` on the thesis that
"the canonical lens says this family carries signal" — `ARCH.V19CL1792X13` seals
two and reads **six** canonically, the campaign's fourth-largest reversal.

Pre-registered: canonical **5** and **2**. Observed: canonical **2** and **2**.

**The canonical lens does not carry the triple down the vocabulary axis.** The
sealed law that triples die is intact, and the canonical reading of the K = 19,456
sibling turns out to be a property of that rung and not of the family. The sealed
counts, meanwhile, beat the allocator: predicted zero for both, observed two and
one.

One detail worth keeping: `V8CL1792X13` keeps retrieval **better** than the rung
above it — *4:15 p.m.* **and** *Saturdays*, both correct, without the K = 12,288
rung's self-contradiction that names the day Tuesday, then Saturdays, then
Tuesday again.

### 4.4 C3: the artifact clause, priced, and it does not survive the test

The canonical standard strikes 846 lanes — 43.6% of every behavioural failure in
this campaign — on the grounds that loops and cap-rides are *"fixable with token
penalties and retraining"*. That is a very large number to rest on an untested
clause, so wave 34 tested the first half of it.

`mx34_penaltydecode.py` re-decodes the frozen prompts through the **same weights**
with one knob moved. Its identity control at penalty 1.0 reproduced the sealed
dossier on **16/16 lanes byte for byte**, so the diagnostic's loop is the sealed
loop plus one line. Five children, 75 lanes, penalty 1.3:

| | lanes |
|---|---:|
| artifact lanes under the sealed decode | 37 |
| → recovered a correct answer | **0** |
| → became readable and empty | **37** |
| → remained artifact | **0** |
| intelligence-signal lanes under the sealed decode | 13 |
| → still holding | **0** |
| → **lost** | **13** |

A keyed screen flagged seven recoveries. All seven were read at full length and
**all seven are false positives** — four are three numbered items with no
household action in them, one is a menu about "response types and partitions",
one names Time Range and unit as fields of a template while inventing *"a
sales-incentified (SIS) analysis for market data in 2020-2015"*. Each rejection is
recorded per lane in the artifact, which is wave 34R's screen-versus-hand rule
applied to wave 34's own diagnostic.

**The penalty eliminates the artifact class completely and recovers nothing.**
Every key element is gone — Supplier B, 4:15 p.m., the four withheld quantities,
Friday, the Monday installation, the tracking request. It converts loops into
fluent nothing.

This does **not** refute the canonical standard. A lane that produces no readable
text is still no evidence about intelligence in either direction, and that is the
standard's actual claim. What it refutes is the **remedy** attached to it: on this
population, at this penalty, there is nothing waiting behind the loop. Whatever
recovers those 846 lanes, a repetition penalty at inference time is not it, and a
handoff dossier that priced them as cheaply recoverable would be wrong.

Nothing in this section is sealed, no ledger row was generated from it, and no
frontier claim cites it.

### 4.5 C4: the extension-v2 surface is NO-BETTER, and the reason is a floor

Wave 33 retired the extension-v1 verdict string at a pooled ρ of **0.6926** over
sixty observations. The v2 bank — thirty prompts, two per lane, A/B staggered,
zero seven-gram overlap with any existing surface, rubric sealed in the same
commit as the bank before any decode — was calibrated on an eleven-child
population spanning sealed 1 to 12.

| reference | ρ | n | range | null p95 | verdict |
|---|---:|---:|---|---:|---|
| sealed frozen fifteen | **0.417** | 11 | 1–12 | 0.5161 | **NOT DISTINGUISHABLE FROM CHANCE** |
| canonical intelligence | **0.561** | 11 | 2–12 | 0.5312 | above chance, **below the v1 bar** |

**NO-BETTER on both.** The reason is a floor, and it is visible without any
statistics: six of the eleven children hold **zero** of thirty, and the **PARENT**
holds **five**. A bank the parent can only answer one prompt in six of cannot
order children that are all below the parent.

Two things are worth carrying forward anyway. The surface tracks the **canonical**
ordering better than the sealed one (0.561 against 0.417) on the same responses,
which is the first quantitative evidence that the two orderings are different
enough to be told apart by an instrument. And the bank is not thereby shown to be
bad — it is shown to be **uncalibrated for this depth**. A v3 that keeps the
disjointness and the frozen rubric and lowers the difficulty is the obvious next
attempt, and it is a different wave's work.

---

## 5. Method and controls

- **Parent calibration, seventh consecutive.** Re-read blind from
  `ARCH.V12CL896`'s dossier: 5 Useful / 7 Partial / 3 misses = **12/15**,
  reproducing the sealed reference exactly, arithmetic still boxing 3 kits per box
  and 94 left over, multi-turn still offering *Vegan Peanut Butter & Jelly Bars* to
  a table with two severe peanut allergies. The parent control also ran **four
  times** this wave and all four transcripts hash identically, so the runner is
  fixed across the whole wave and not only at its start.
- **The one calibration this wave adds** is the whole-prompt echo whose tail is
  the answer. Wave 23 sealed `V16CL896`'s single case on it, wave 28 sealed
  `V18CL896` and `V19CL896` at two each on it, and wave 28's `V19X13CL896` sealed
  ZERO because it paraphrased instead and the paraphrase dropped the decision. The
  precedent is applied unchanged and in both directions. It decided six lanes
  across five children, which is why it is named where it was used.
- **Wave 33's byte-comparison control fired again.** Four of sixteen lanes on
  `V12CL896G10` and one on each of G01/G03 decode byte-identical to their source;
  on those lanes the source's verdict is the correct verdict by construction, and
  the comparison was run before the counts were written.
- **Benchmarks**: nine rows, standard invocation (`limit=200 seed=1234 batch=4
  max_length=2048 float32, sm86`). The **parent row is identical to wave 32's
  sealed parent row to every reported digit** — arc 0.5450, hellaswag_norm 0.5050,
  piqa_norm 0.6800, winogrande 0.5500, lambada 0.4600, lambada ppl 24.39, wikitext
  word ppl 81.80. Wave 33's own `bench/benchmark-table.json` is zero bytes on disk;
  that is a wave-33 artifact loss and is recorded here rather than repaired.
- **Seal verification**: 936 ledger rows, 932 resolve to a receipt on disk, 932
  verify, **0 fail**, 4 unresolved — the same historical four (`CAL.C01.T5.X4` ×3,
  `COMP.MC.MXO0`). 30 seal roots audited against disk, 0 undeclared.
- **`sealq.sh` ran once.**

---

## 6. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `02dd19c6` | finish the three outstanding hand edits; the audit stops refusing |
| `681cf94b` | pre-register wave 34, including the canonical count, before the first build |
| `fd7a73c3` | `mx34_calibrate.py`, `mx34_penaltydecode.py`, `extension2`, the v2 seal file |
| `082409cb` | the builder has written the wrong wave's receipt filename since wave 32 |
| `03d39e85` | append wave 34 to the sealer's accept list, after it refused four reviews |
| `0ed36347` | the eight seals and the calibration that decides six of their lanes |
| `710af305` | the measurement: four sealed rungs move, and the artifact clause is priced |
| `fa635517` | the retention policy, written from this wave's verdicts, and this note |
| `720f475d` | package, retain and prune, in that order, with every deletion named |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `02dd19c6` | 6325 | 0 | 0 | 0 | 0 |
| `082409cb` | 6329 | 0 | 0 | 0 | 0 |
| `03d39e85` | 6329 | 0 | 0 | 0 | 0 |
| `0ed36347` | 6329 | 0 | 0 | 0 | 0 |
| `fa635517` | 6359 | 0 | 0 | 0 | 0 |

`fa635517` was **re-verified after the wave's work finished** — 6,359 files, all
five zeros again — which is the check that nothing wrote into a deployment while
it was being read from.

Five deployments for one wave is four more than the ideal and three of the four
extras were bought by a guard refusing correctly: the receipt-filename defect, the
sealer's accept list, and the seal calls themselves. That is the trade this estate
has chosen.

**Package, then retention, then prune.** The bundle is 91 files and 2.1 MB; the
repository carries 80 of them, the eleven it does not being copies of files that
already live elsewhere in the repo. Retention kept five children and deleted six
(1.37 GB), each deletion naming what rebuilds it and from which retained source,
after a dry run that was printed and read. The deployment prune kept three and
deleted four (1.14 GB) under current+1 **plus every deployment a sealed receipt
names as its `repo_head`** — all sixteen wave-34 receipts name `03d39e85`, so that
one is kept regardless of age. Nothing pruned is lost: `deploy.py --emit`
reproduces any of them byte-for-byte from its commit, and no receipt in this
estate names a deployment *path*.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave34-20260828T153018Z` and
`cmp`-verified byte-identical before the merge. One-pass merge, 920 → 936 rows, 16
appended, 0 superseded. Byte-prefix verified two ways: the merge tool's own
post-write prefix check, and `head -c <backup size> ledger.jsonl | sha256sum`
returning `c9f6cbe5…`, the backup's digest. **The mirror was copied back into the
repository and prefix-checked again there** — the twice-recurred defect, not
recurred. Repo and share both `cff07ea4a165683b2a400ecc6a380db7ff707b82aa55d1f180f511c932b9b93f`.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both were carrying an
**idle-holding** H3 ComfyUI container at **256 MiB each with an empty queue** —
the peer sprite lane. They were **not restarted**: 256 MiB on a 24 GB card is not
a hold worth taking a peer's container down for, the fold builds peaked at 1.4 GB,
and nothing in this wave came close to memory pressure. The Quadro RTX 8000 was at
100% on a peer benchmark for the whole wave and was never touched.

**Disk.** `/mnt/ai_data` at 64% (1.7 TB free) before and after; `/home` 63% with
650 GB free after the prune, deployments down from 2.3 GB to 1.2 GB.

**The Codex lane was not touched.** Nothing outside the explicit paths in §7 was
staged.

---

## 7. What this wave changed, and what it did not

Changed: `reports/matrix-campaign-20260810/run/mx34/` (three hand edits, the
README, `mx34_calibrate.py`, `mx34_penaltydecode.py`, the `extension2` arm in
`mx34_extreview.py`, the eight seals in `sealq.sh`, the two audit assertions in
`cut_wave34.py`); `experiments/seal_matrix_architecture_child.py` (four appended
accept strings); `reports/matrix-campaign-20260810/extension-bank-seal-v2.json`
(new); `wave34/` (fourteen artifacts and sixteen dossiers);
`RESULTS_MATRIX_2026-08.md`, `CAMPAIGN_SCORECARD.md` and
`ACTUAL_MATRIX_2026-08-28.md`, each backed up under `backup/` with its
pre-edit sha256 recorded in the commit; `ledger.jsonl`.

Not changed: the frozen fifteen prompt pack, the frozen review runner, the
extension-v1 bank and its seal file, every earlier wave's sealed receipts,
verdicts and dossiers, and the Codex lane.

## 8. What the next wave should know

1. **The 0/15 floor is a 0.0091-wide bracket and nothing has been built inside
   it.** `ARCH.V9CL896` at K = 9,216 — 100,456,192 parameters, **0.437349x**, four
   thousandths above the floor — is one CPU post-process on a body that is already
   on disk and reproduces bit-identically. It would halve the bracket.
2. **λ0.3 on a CL896 base is worth two cases and λ0.1 and λ1.0 are worth none.**
   Four bases now agree on the arm and disagree by one on the base. The K = 12,288
   base is the one that buys retrieval; nobody knows why.
3. **C2 is closed.** The triple does not carry canonically down the vocabulary
   axis. Do not re-order it.
4. **The artifact clause has a price now and it is not zero.** Any claim built on
   "846 lanes are recoverable" needs a mechanism that is not a decode-time
   penalty. Retraining is the untested half of the clause and it is a stage-2
   activity under the staged escalation policy, so it may not be prototyped while
   stage 1 is open.
5. **Extension-v2 needs an easier bank, not a better rubric.** The rubric
   self-tests clean and the disjointness holds; the parent scores 5/30. Difficulty
   is the defect.
