# NOTES-MX-AH — wave 32: the flatness law is a fact about one body, and the wave that went looking for a better one moved nothing

Wave 32 was sent to exploit wave 31's composed-flatness law — improve the
`CL1792` body and every composed vocabulary rung inherits it. It built fifteen
children to do that and **moved no frontier row at any of the fifteen
thresholds.** It also retired one that wave 31 had claimed.

That is the headline and it is a negative one. What the wave bought instead is
four mechanisms, each closed with a measurement rather than a null:

1. **A convex blend of two sealed BODIES is a ridge, not a segment**, and the
   reason was measurable in 6.6 seconds of CPU before any GPU ran.
2. **The flatness law holds on exactly one body in the campaign** — the one it
   was discovered on. It fails on the reader-metric body at four rungs and on a
   blend body at one.
3. **The vocabulary floor is 7,213 rows**, it is a property of the census corpus
   alone, and it is **monotone increasing** in that corpus — so the instruction
   to move it by capturing a broader one had a sign error in it. Released
   properly, the axis goes two rungs deeper and finds a cliff, not headroom.
4. **B4 is answered three waves late: sequentially conditioned capture is worth
   three cases**, measured on a controlled pair at byte-identical storage.

---

## 0. Two failures of this wave's own discipline, stated before the results

**The W1 slate was graded before a pre-registration existed.** Seven blends were
built, reviewed and read against the standard before
`prereads/preregistration-wave32.json` was written. The excuse that offers
itself — the allocator prices axis *levels* and a convex blend of two bodies
moves no level, so it cannot price one — is not good enough, because wave 31
faced exactly that with `ARCH.CL1792ME` and still wrote a `predicted_review:
null` entry with `UNPRICEABLE` and its honest prior stated. That entry was
available here and was not written. The file says so in its own second key, and
the seven W1 counts below carry **no prediction column** because they are not
entitled to one. W2, W3 and W4 were pre-registered with every dossier unopened
and four still generating.

**The tree cut carried nineteen of wave 31's seal calls forward.** `sealq.sh` is
not in `cut_wave32.py`'s `HAND_EDIT` set, so the copy renamed `mx31`→`mx32` and
brought wave 31's calls across with **every count intact** — `CL1664` at 1/4/10,
`V16CL1792` at 2/6/7 — onto cell ids in this wave's tree. Wave 31's own comment
*in that exact position* records that wave 30's calls were DELETED rather than
renamed, and says why. Wave 32 inherited the sentence and not the protection.

What saved it is thin and the file now says so: the tags do not exist in
`wave32/cells`, so `seal_verdict.sh` would have failed on a missing receipt
rather than sealing last wave's numbers onto this wave's rows. To that script a
count is just a number. Three fixes shipped: the calls deleted by hand,
`sealq.sh` added to `HAND_EDIT` for the next cut, and `cut_wave32.py --audit`
now asserts the seal block's marker names **this** wave.

---

## 1. The alignment census: why a body blend is not a diagonal blend

Wave 29's `ARCH.B75` took the ≥11 frontier with a **zero-solve convex blend of
two siblings at alpha 0.75**, and the build list ordered the same move on the
`CL1792` family. Wave 29's blender refuses it, correctly, and the refusal is the
first result of this wave.

`mx32_build.py`'s `intermediate_size` fold is a **channel selection**, not a
projection: `pivoted_columns` picks `keep_ff` real channels of the parent's
2,560, `w1` and `w3` keep those parent rows **verbatim in the pivot's own
order**, and only `w2` is re-solved. So two siblings that disagree about which
channels to keep store *the same parent rows in a different order*.

`mx32_blend_body.py --census` recovers each sibling's selection by **exact
`(w1,w3)` row match against the parent** — no tolerance, a row that matches no
parent row is a refusal — and measures the overlap:

| | shared channels | same channel at the SAME row index |
|---|---:|---:|
| `CL1792` vs `CL1792M` | 23,015 of 25,088 (91.7%) | **2,053 (8.2%)** |
| `CL1792` vs `CL1792MV` | 23,014 of 25,088 (91.7%) | **1,795 (7.2%)** |

**91.8 per cent of rows carry two unrelated parent channels.** An entrywise
blend of `w1`/`w3` mixes them. Wave 29's diagonal blend mixed 1,024 numbers with
one shared coordinate meaning; this is a different operation wearing the same
word, and the census cost 6.6 seconds of CPU and no GPU.

So the builder has two arms. `--align channel` keeps the source's selection and
its `w1`/`w3` byte-for-byte and blends **`w2` column-wise on the recovered
channel intersection**, with the convex weight renormalised over the siblings
that actually kept each channel — a column multiplied by `1−a` is not a convex
combination, it is a shrunken one. `--align none` is the naive arm, built once
so that "meaningless" could be a number.

## 2. W1: the blend simplex, and the ridge

Seven children, all at **byte-identical storage 0.856199x, 196,663,040
parameters**, all one CPU pass, no solve, no capture.

| cell | arm | weights | **sealed** |
|---|---|---|---:|
| `ARCH.CL1792` (w30) | endpoint | — | 7/15 |
| `ARCH.CL1792M` (w30) | endpoint | — | **8/15** |
| `ARCH.CL1792MV` (w31) | endpoint | — | 7/15 |
| `ARCH.BM25` | aligned | 0.75 CL1792 + 0.25 M | **6/15** |
| `ARCH.BM50` | aligned | 0.50 / 0.50 M | **4/15** |
| `ARCH.BM75` | aligned | 0.25 / 0.75 M | **2/15** |
| `ARCH.BMS75` | aligned, **source-swapped** | 0.75 CL1792 + 0.25 M on M's selection | **5/15** |
| `ARCH.BV50` | aligned | 0.50 / 0.50 MV | **4/15** |
| `ARCH.BMV3` | aligned, 3-way barycentre | ⅓ each | **7/15** |
| `ARCH.BM50N` | **naive** | 0.50 / 0.50 M, no alignment | **0/15** |

**No interior point beats any endpoint.** Read the `CL1792`→`CL1792M` segment in
order: **7, 6, 4, 2, 8**. It falls monotonically away from `CL1792` and then
jumps five cases at the far endpoint. That is not an interpolation of anything —
it is the signature of two solves that live in *different bases*, where
alignment recovers the identity of the channels but not the basis the
coefficients were solved in. The target the build list set — a body ≥9/15 at
0.856x — is not reachable this way, and the best body at that ratio remains
wave 30's `ARCH.CL1792M` at 8/15.

**Three controls make that a mechanism rather than a shape.**

* **The source swap.** `BMS75` is the *identical convex point* to `BM25` built
  on the other sibling's selection and `w1`/`w3`. 5/15 against 6/15 — one case
  on a fifteen-point instrument. **Which sibling's selection survives is worth
  at most one case**, so the collapse is not an artifact of the source choice.
* **The second midpoint.** `BV50` (against `CL1792MV`) reads 4/15, the same as
  `BM50` (against `CL1792M`). Two midpoints, two different siblings, one count.
  The collapse is a property of *mixing two solves*, not of which solve.
* **The naive arm.** `BM50N` — identical weights, identical storage, identical
  everything except the alignment — reads **0/15 with 16 of 16 lanes charged,
  11 cap hits and four lanes returning the empty string.** This is the
  campaign's second 0/15 and the first **predicted in a committed file before
  the build**: `mx32_blend_body.py`'s docstring says the naive arm is meaningless
  and says exactly why, and the census measured the reason.

The one result a distance-from-endpoint story cannot produce is `BMV3`. The
three-way barycentre reads **7/15** — three cases *above* the pairwise midpoint
at a point no closer to any endpoint. It ties `ARCH.CL1792` and sits one below
`ARCH.CL1792M`. A tie, not a gain, and worth naming because it is the only
evidence that the simplex interior has structure the segment does not.

## 3. W2: composing the better body, and the flatness law's real domain

Wave 31's headline is that on `ARCH.CL1792` the composed vocabulary axis is flat
from K = 26,624 to 8,192 — the composed child reads the **body's** number. Wave
32 composed the same four rungs onto `ARCH.CL1792M`, which reads **one better**
alone, at byte-identical storage.

| K | ratio | on `CL1792` (w31) | on `CL1792M` (w32) | Δ | predicted |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 0.637073 | **8** | **3** | **−5** | 6 |
| 12,288 | 0.618813 | **7** | **4** | **−3** | 6 |
| 9,216 | 0.605117 | **6** | **2** | **−4** | 5 |
| 8,192 | 0.600552 | **5** | **3** | **−2** | 4 |
| body | 0.856199 | 7 | **8** | **+1** | — |

**Every rung is negative, mean 3.5 cases, and the body is the only place the M
arm wins.** Wave 30 measured this inversion once at K = 19,456 and read it as
one case; it is one case *only at the rung where it was first seen*. The M
body's composed ladder is 3/4/2/3 against a body of 8 — not flat, and nowhere
near the body's number.

The one shape inside it worth keeping: **the penalty shrinks as K falls** — five
cases at 16,384 down to two at 8,192 — so the two bodies converge as the
vocabulary gets small enough to dominate.

**And the law fails on a blend body too.** `ARCH.V16BMV3` composes K = 16,384
onto `ARCH.BMV3` (7/15) and reads **3/15**, four below the body and five below
`ARCH.V16CL1792` at the same K and the same storage. Predicted 7 on the law.

> The flatness law now has **three bodies where it fails and one where it
> holds**, and the one where it holds is the body it was discovered on. It should
> be restated as a fact about `ARCH.CL1792` and not as a law about composition.

The pre-registration was systematically optimistic here: 6/6/5/4 against 3/4/2/3,
mean error −2.25, every one in the same direction. That is the same sign the
allocator has carried on every axis it can price itself.

## 4. W3: the corpus floor is 7,213 rows and a broader corpus RAISES it

Wave 31 refused K = 6,144, 5,120 and 4,096 and recorded "a hard floor between
6,144 and 8,192 rows on this corpus". Wave 32 was told to move that floor by
capturing a larger census corpus. **That instruction has a sign error in it and
the sign is measurable in seconds.**

The kept set the gate accepts is exactly `mandatory ∪ closure(observed)`:

| | |
|---|---:|
| mandatory (509 specials + 256 ByteLevel + closure) | 765 |
| distinct tokens the sealed corpus observes | 5,371 |
| merge-tree ancestors those pull in | 1,187 |
| **the floor** | **7,213** |

Since `observed(C₁) ⊆ observed(C₁ ∪ C₂)`, the floor is **monotone increasing**
in the census corpus. Measured rather than argued: widening the census with
2,372 lines of the deployment's own committed research prose adds 2,926 distinct
tokens and moves the floor **7,213 → 10,390, up by 3,177 rows.** A broader
corpus cannot lower it.

What actually held the floor up is that wave 25's gate refused on *"the closure
is wrong **or** K is below the observed vocabulary"* — one refusal for two
conditions, only one of which is a defect in the builder. Wave 32 splits them:

* **structural**, always fatal, no flag — `merge_closure_intact()` asserts every
  kept token is producible from the restricted merge list, and it names no corpus;
* **coverage**, a decision — `--below-floor`, which prints the exact loss and
  records `below_census_floor` in the receipt.

This axis has published sub-unity coverage on sealed children since wave 24:
`ARCH.V8` reads 4/15 solo at 0.80049 token coverage and nothing refused it.

| cell | K | ratio | dropped (donor / canonical) | tok/byte | **sealed** | predicted |
|---|---:|---:|---:|---:|---:|---:|
| `ARCH.V8CL1792` (w31) | 8,192 | 0.600552 | 0 / 0 | 0.2773 | 5/15 | — |
| `ARCH.V6CL1792` | 6,144 | 0.591422 | 1 / 782 | 0.2917 | **3/15** | 3 ✓ |
| `ARCH.V4CL1792` | 4,096 | 0.582292 | 324 / 2,860 | 0.3164 | **1/15** | 1 ✓ |

Both exact hits. The composed ladder below the floor is **5, 3, 1 — two cases
per 2,048 rows** — against the flat 7‑to‑8 the same ladder reads from K = 26,624
down to 8,192. **The flatness law ends at the floor, and the floor was hiding a
cliff rather than headroom.** Both children still pass the merge-closure
assertion and are lossless on text on all 256 corpus rows and all 16 review
prompts; what a dropped token costs is sequence length and a different
segmentation, and `V4CL1792` pays 48% more tokens per byte than the parent.

**Neither rung moves a frontier.** `ARCH.V19CL896G03` holds 3/15 at 0.483001x
and `ARCH.V16CL896` holds 1/15 at 0.469305x — both far deeper. The floor was
never what was stopping the campaign from going deep; it was stopping it from
going deep *on this body*.

## 5. W4: B4, answered, and three reproduction checks

**B4's controlled pair, three waves late.** `ARCH.CL1792S` is `ARCH.CL1792`
rebuilt with `--arm stale`: byte-identical storage, the identical solver, the
identical evidence budget, the identical K, and the only difference is whether
each site's evidence is captured through the partially built child or read off
the unfolded parent. The build log is the control's own receipt — **input drift
versus parent 0.0000 at all fourteen sites**, against 0.17–0.27 on the
conditioned arms.

> `ARCH.CL1792S` reads **4/15** against `ARCH.CL1792`'s **7/15**. Predicted 4.
> **Sequentially conditioned capture is worth three cases.** Eleven waves have
> paid for it without pricing it; the price is three cases and it is worth
> paying.

**The reproduction check wave 31 owed is now a measurement.** Wave 31 argued from
a diff that its four `dbf2294e` bodies were unaffected by the builder change.
Rebuilt under this wave's deployment:

| body | sealed digest | rebuilt digest | |
|---|---|---|---|
| `CL1408` | `ca0b64c88073cf01…` | `ca0b64c88073cf01…` | **BIT-IDENTICAL** |
| `V9CL1792` | `902d686fa77006e5…` | `902d686fa77006e5…` | **BIT-IDENTICAL** |
| `V8CL1792` | `29433f6ce9b7fe58…` | `29433f6ce9b7fe58…` | **BIT-IDENTICAL** |

Three, not one: the two vocabulary rebuilds were needed anyway so the paraphrase
surface could be run on rows wave 31 dropped, and they came with a free
cross-wave reproduction test of the vocabulary path.

**The paraphrase triplets ran, and they retire a frontier row.** Authored and
unrun for three waves, now decoded on the frontier five. `ARCH.V16CL1792`'s
eighth case — the ≥8/15 rung at 0.637073x — rides entirely on
`creative-continuity`, and wave 31 flagged it as a one-case margin on a split
precedent. Under the three restatements of that lane's ask:

* *"unfroken, with a lighthouse keeper who is replying to the foghorn… not
  prepared to engage in a conversation"* (186 chars)
* *"The keeper is noticing the foghorn's unsettling, leaving it unexplained."* (72)
* *"The foghorn has been a constant source of concern for me… I am worried about
  the weather"* (219, and the keeper is gone)

**Zero of three would be graded Partial.** `ARCH.V12CL1792` fails the same lane
0 of 3 as well. So the sealed difference between them is carried by a case that
does not survive rewording, and **the robust claim at 0.637073x is 7/15, not
8/15** — which ties `ARCH.V12CL1792`'s 7/15 at 0.618813x, and that one is
deeper.

A census over all 225 paraphrase responses says the same thing quantitatively:
the **median within-triplet length spread is 53–79%** for every one of the five.
Three restatements of one ask produce answers of wildly different length. These
readings are prompt-specific in a way the sealed fifteen cannot see.

## 6. Both surfaces, and the second one refuses to help

Bank `c9d37004c901d757…`, unchanged since wave 29; parent run once per bank.
**Spearman 0.3537 over seventeen children — `SCRAMBLED: the sealed ordering does
NOT reproduce`**, against wave 31's 0.7411 `VALIDATED` and wave 30's 0.6544
`PARTIAL`.

| child | sealed 15 | ext 30 | charged (30) |
|---|---:|---:|---:|
| `V16CL1792` | 8 | 9 | 5 |
| `BMV3` | 7 | 8 | 4 |
| `V12CL1792` | 7 | 9 | 3 |
| `BM25` | 6 | 8 | 3 |
| `BMS75` | 5 | 7 | 5 |
| `BM50` | 4 | 8 | 4 |
| `BV50` | 4 | 8 | 1 |
| `CL1792S` | 4 | 8 | 6 |
| `V12CL1792M` | 4 | 4 | 6 |
| `V16BMV3` | 3 | 7 | 5 |
| `V16CL1792M` | 3 | **9** | 6 |
| `V6CL1792` | 3 | 8 | 7 |
| `V8CL1792M` | 3 | 6 | 3 |
| `BM75` | 2 | 7 | 5 |
| `V9CL1792M` | 2 | 7 | 2 |
| `V4CL1792` | **1** | **10** | 2 |
| `BM50N` | 0 | 3 | 25 |

**The bank's known failure mode fires harder than it ever has.** `V4CL1792` —
the worst reviewed child in the wave at 1/15 — takes the **highest extension
score in the wave at 10/30**. `V16CL1792M` scores 9/30 against a sealed 3. The
rubric is fooled by fluent strings exactly as its docstring says it can be, and
this wave's children are clustered so tightly at the bottom that the ordering it
produces is noise.

The score's own verdict string says what follows: *"under the plan's
pre-registered go/no-go every one-case play pauses and this goes to Cyril."*
**Nothing in this note fires on the second surface**, and the one-case margins
in §2 are labelled as such where they appear.

`mx32_extscore.py` **cannot score the paraphrase surface** — its `RUBRIC` is
keyed on the extension bank's own thirty prompt ids and returns 0 for all 45
paraphrase ids. That is a refusal to guess, not a result, and the paraphrase
finding in §5 is a hand reading of the dossiers with a mechanical degeneracy
census beside it.

## 7. Benchmarks

`limit=200 seed=1234 batch=4 max_length=2048 float32`, sm86. The parent row was
re-measured and reproduces the sealed reference exactly (0.5450 / 0.5050 /
0.6800 / 0.5500 / 0.4600, ppl 24.39, wiki 81.80) — the fifth calibration.
`ARCH.BM50N` is **excluded** and the exclusion is the point: eleven of its
sixteen lanes hit the cap and four return the empty string, so it is a looping
child and the standing rule is benchmarks per non-looping child.

| model | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 | 12/15 |
| `BMV3` | 0.4050 | 0.4550 | 0.5650 | 0.5550 | 0.2350 | 252.34 | 185.07 | **7/15** |
| `BM25` | 0.4050 | 0.4500 | 0.5900 | 0.5400 | 0.2650 | 192.13 | 181.76 | 6/15 |
| `BMS75` | 0.4100 | 0.4300 | 0.5400 | 0.5750 | 0.2500 | 171.76 | 193.74 | 5/15 |
| `BM50` | 0.4050 | 0.4450 | 0.5800 | 0.5500 | 0.2350 | 217.71 | 181.00 | 4/15 |
| `BV50` | 0.4000 | 0.4600 | 0.5900 | 0.5500 | 0.2450 | 216.53 | 181.60 | 4/15 |
| `CL1792S` | 0.3750 | 0.4300 | 0.5900 | 0.5450 | 0.2550 | 196.36 | 196.82 | 4/15 |
| `BM75` | 0.4050 | 0.4600 | 0.5700 | 0.5900 | 0.2050 | 294.16 | 190.88 | **2/15** |
| `V16CL1792M` | 0.3600 | 0.3900 | 0.5150 | 0.5650 | 0.2000 | 639.71 | 521.59 | 3/15 |
| `V12CL1792M` | 0.3650 | 0.3600 | 0.5050 | 0.5750 | 0.1900 | 740.27 | 759.46 | 4/15 |
| `V9CL1792M` | 0.2900 | 0.3650 | 0.5050 | 0.5450 | 0.1800 | 1036.55 | 1301.09 | 2/15 |
| `V8CL1792M` | 0.3100 | 0.3550 | 0.5150 | 0.5050 | 0.1750 | 1406.93 | 1782.48 | 3/15 |
| `V16BMV3` | 0.3550 | 0.4350 | 0.5050 | 0.5650 | 0.1500 | 663.34 | 507.62 | 3/15 |
| `V6CL1792` | 0.3050 | 0.3150 | 0.4950 | 0.5300 | 0.1500 | 2029.04 | 2836.62 | 3/15 |
| `V4CL1792` | 0.3250 | 0.3400 | 0.5200 | 0.4850 | 0.1150 | 4480.44 | 5163.13 | 1/15 |

**The blend row is the cleanest instance of the perplexity inversion this
campaign has produced.** All seven blends sit in a 13-point wiki band — 181.00 to
196.82 — and read **2 to 7 of fifteen inside it.** `BM75` at wiki 190.88 reads
2/15; `BM50` at wiki 181.00 reads 4/15; `BMV3` at 185.07 reads 7/15. Word
perplexity cannot separate children that differ by five cases at byte-identical
storage, and this is the eleventh wave to say so.

## 8. What was ordered and not run

* **LACE shrinkage on the CL solve** — the field slate's #4, ordered as a fourth
  sibling for the blend simplex. Not run. §2's three controls closed the blend
  lane on their own and a fourth endpoint would have been a fourth point on a
  ridge that three points had already characterised.
* **Composing V16/V12/V9/V8 onto the best W1 body.** The build list said
  "compose the best body". The best body at 0.856199x is `ARCH.CL1792M` at 8/15,
  which is what §3 composed onto — **no wave-32 blend beat it**, so the blend
  arm's composed test is the single point `V16BMV3` rather than a four-rung
  ladder. That is a substitution, not an omission, and it is stated here because
  the plan's sentence and the wave's action are not word-for-word the same.
* **The paraphrase surface is decoded but not machine-scored** (§6), because the
  scorer's rubric is bank-specific. Five children × 45 prompts of verbatim text
  are packaged; a rubric for them is a next-wave job.

## 9. Estate

**Four commits, four deployments**, every one verified file by file against the
commit's own blob digests: **5,951 / 5,952 / 5,953 / 5,955 files, 0 content
mismatches, 0 missing, 0 undeclared, 0 hardlinks**, every time, plus a post-wave
re-verification of the survivor at 5,955 and the same four zeros. The sealer's
four new schema strings — three renamed builders and the genuinely new
`neural-forge-matrix-wave32-body-blend-build-receipt-v1` — shipped **in the
first commit, before the first build**, because wave 31 learned that this list
refuses six reviews at the top of the sealer.

**The tree cut.** `run/mx32/cut_wave32.py` is wave 31's, with two changes: wave
31's tree-cut manifest is declared `NOT_CARRIED` rather than renamed forward and
overwritten, and `ALLOWED_HISTORICAL` starts with only the build header. Five
substitutions, all identifiers, paths and schema strings, no prose form of the
wave number. **27 declared seal roots became 28 with `wave30/cells` and
`wave31/cells` still declared**, and `BASIS_SCHEMAS_ACCEPTED` **gained** wave
31's schema. The audit fired twice and was right twice: once on `sealq.sh` (§0)
and once on the retention policy, where two DROP reasons cite the retained
wave-31 checkpoints their children are measured against. That count is **pinned
at two**, not waived.

**Ledger: 878 → 908 rows, 30 appended, 0 superseded.** Backup
`backup/ledger-pre-wave32.jsonl` at the pre-merge digest `85d5f12674169787…`,
`cmp` identical. The merge ran in **two passes and that is a defect worth
naming**: the first pass was given only the `arch-*` shards, appended 15 rows,
and the campaign table then showed **zero** wave-32 children — the verdict
shards carry the review rows and they had been left out. The second pass
appended the missing 15 behind its own `.MERGED_VERDICTS_ONCE` guard, and the
authority's pre-wave bytes are a **byte prefix** of the result at both steps
(1,500,006 of 1,553,089). New digest `48fb010284bdec15…`.

**Seal verification: 28 roots audited against disk, 0 undeclared; 908 rows, 889
resolve and all 889 verify, 0 failures**, with the same four unresolved rows
waves 25–32 carry (`CAL.C01.T5.X4` ×3, `COMP.MC.MXO0`).

`sealq.sh` ran **once**, behind `.SEALQ_RAN`; the eight-argument guard fired on
none of the fifteen. Parent lane calibration re-read blind from `ARCH.BM25`'s own
dossier and reproduces the sealed 5 Useful / 7 Partial / 3 miss = **12/15**
exactly, including the sealed parent arithmetic defect. The wave adds one
calibration and states it where it cost the cases: **`creative-continuity` keys
on THREE motifs held, not two**, which decided six lanes in one direction —
including `ARCH.CL1792S`'s, which is a real scene with a marked line of dialogue
and loses the lane on the fog alone.

Both sm86 3090s ran the wave. Both H3 containers were queue-empty and holding
288 and 279 MiB; wave 31 established by restarting them that **288/279 is the
free-standing floor on these cards rather than a leak**, so this wave checked the
queues, read that result and did not re-run a settled experiment. `/mnt/ai_data`
at **1.7 TB free, 64% used, unchanged across the wave**; both cards free-standing
at 288 and 279 MiB with nothing running when it closed.

Retention ran **after** the package, `--plan` read before `--apply`: **three
children kept (1.07 GiB), fifteen deleted (5.00 GiB)**. Two of the three keeps
deviate from the drop rule and both deviations are stated in the manifest —
`MX32-BMV3` because `mx32_blend_body.py` is a builder this wave *wrote* and no
wave has reproduced its output across a wave boundary, and `MX32-V6CL1792`
because somebody who wants to argue with the `--below-floor` release should be
able to do it against bytes. Deployments pruned from six to two (current+1),
survivor re-verified after the wave.

## 10. What the next wave should do with this

1. **Stop blending bodies and stop composing onto anything but `ARCH.CL1792`.**
   §2 and §3 each closed a lane with a measurement. The one composition arm that
   has ever paid is a vocabulary gather onto the stream-selected 1792 body, and
   §3 says that is a fact about *that body*.
2. **Price conditioning on a second axis.** §5 says conditioned capture is worth
   three cases on the mlp axis at `intermediate_size` 1792. That is one point.
   The same `--arm stale` control on a vocabulary composition or a depth excision
   costs one build each and would say whether three cases is the axis's number or
   the campaign's.
3. **Fix the second surface or stop running it.** Three waves of Spearman —
   0.6544, 0.7411, 0.3537 — is not an instrument, and this wave's run put the
   campaign's worst child at the top of it. Either the rubric gets a
   degeneracy gate that a 1/15 child cannot pass, or the GPU it costs goes back
   into sealed reviews.
4. **The ≥8 rung at 0.637073x is retired** until something holds
   `creative-continuity` under rewording. The campaign's honest deep frontier is
   `ARCH.V12CL1792` at **7/15, 0.618813x**.
