# NOTES-MX-U — wave 17: the census closes, the pair cost is a distance law, and the merge finally has a solve

Wave 17, branch `codex/lrc-blank-compression`, source commits `0a7d8f62`
(ADDENDUM 14 + the wave-17 builder with T7's joint solve), `df735362` (the
preconditioner the self-test demanded) and `049326e0` (the gauge the first joint
build lost), on top of wave-16 HEAD `2d92b8d7`. Deployment
`neural-forge-matrix-049326e0`, cut with `git archive -c core.autocrlf=false`
from this wave's own commit; protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` and prompt
pack `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` verified
on extraction and on every review.

**Three calibrations passed before anything was graded.** The fresh conditioned
capture reproduces the sealed collector's rows to wave 15's numbers exactly (L00
relL2 3.367e-3 cos 0.9999944, L05 1.647e-2 / 0.9997207, L13 2.034e-2 /
0.9997842). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23`** on **all twenty-one** wave-17 dossiers, byte-
identical to every wave since 7. And the parent's benchmark row reproduces the
sealed reference in all seven columns (0.5450 / 0.5050 / 0.6800 / 0.5500 /
0.4600, lambada ppl 24.39, wikitext 81.80).

*§0's sentences are at the bottom.*

---

## 1. The census is closed: all fourteen blocks, behaviourally

Six blocks were unbuilt. They are built, and the map is complete. Every child is
one block excised with every surviving FFN writer re-solved at unchanged width,
sealed and read on the frozen surface.

| block | kind | net | score | per-lane | per-child | charged |
|---:|---|---:|---:|---|---|---:|
| **13** | conv | 0.947479x | **13/15** | alive-parent-band | dead | 2 |
| **7** | conv | 0.947479x | **11/15** | alive-parent-band | **alive-parent-band** | 0 |
| **9** | conv | 0.947479x | **10/15** | alive-parent-band | **alive-parent-band** | 0 |
| **1** | conv | 0.947479x | 9/15 | alive-degraded | dead | 1 |
| **11** | conv | 0.947479x | 9/15 | alive-degraded | **alive-degraded** | 0 |
| **6** | attn | 0.952057x | **8/15** | alive-degraded | dead | 1 |
| **8** | attn | 0.952057x | 7/15 | impaired | dead | 1 |
| **10** | attn | 0.952057x | 7/15 | impaired | **impaired** | 0 |
| **5** | conv | 0.947479x | 6/15 | — | dead | — |
| **3** | conv | 0.947479x | 6/15 | impaired | **impaired** | 0 |
| **4** | attn | 0.952057x | 2/15 | impaired | dead | 9 |
| **0** | conv | 0.947479x | **0/15** | **dead (general)** | dead | **16** |
| **2** | attn | 0.952057x | 0/15 | impaired | dead | 6 |
| **12** | attn | 0.952057x | 0/15 | **dead (general)** | dead | **16** |

(Blocks 5, 3, 1, 7, 9, 11, 13 and 12 are waves 15–16; the per-lane column for
X03/X07/X09/X11/X12/X13A is this wave's ADDENDUM 14 re-seal.)

**Wave 16's headline does not survive contact with the other six blocks.** It
read *every conv block is cheap and the attention block is not*. The completed
map says something narrower and stranger:

1. **The first conv block is the most expensive block in the model.** `ARCH.X00`
   removes the same 12.1M parameters as `ARCH.X13A`'s 13/15 and scores **0/15**
   with **all sixteen lanes charged** — two lanes return the empty string, the
   rest loop `is`, `er`, `:` or `,` to the 1,024-token cap. It is the first child
   in the campaign to seal a `general` degeneracy reading, which is the reading
   ADDENDUM 14 was written to make expressible.
2. **Attention blocks are not a class.** 2 and 12 are dead, 4 is 2/15, and 6, 8,
   10 read **8, 7 and 7** — a live band that overlaps the conv blocks. The
   attention axis does not have a price; the *site* does.
3. **Depth orders it, and the order is not monotone.** Reading the conv blocks by
   index: 0 → 0, 1 → 9, 3 → 6, 5 → 6, 7 → 11, 9 → 10, 11 → 9, 13 → 13. Reading
   the attention blocks: 2 → 0, 4 → 2, 6 → 8, 8 → 7, 10 → 7, 12 → 0. **Both kinds
   are cheapest in the middle-to-late interior and expensive at their own first
   block**; attention is additionally expensive at its last.

## 2. The pair map, and the one clean law this campaign has produced

Eight pairs, all at **0.894958x**, all conv, no other axis moved. Two are wave
16's; six are this wave's, chosen to hold one block fixed and move the distance.

| pair | distance | score | charged | per-lane |
|---|---:|---:|---:|---|
| 13+11 | 2 | 2/15 | 10 | impaired *(wave 16)* |
| 9+11 | 2 | **2/15** | 8 | impaired |
| 7+9 | 2 | **4/15** | 7 | impaired |
| 13+9 | 4 | 6/15 | 5 | impaired *(wave 16)* |
| 7+11 | 4 | **6/15** | 2 | impaired |
| 13+7 | 6 | **6/15** | 1 | impaired |
| 13+3 | 10 | **7/15** | 3 | impaired |
| 13+1 | 12 | **7/15** | 2 | impaired |

**The cost of a pair is a function of the distance between the two blocks and
essentially nothing else.** Sorted by distance the scores are
2, 2, 4 | 6, 6 | 6 | 7 | 7 — monotone, with no inversion, across four distances
and two different anchors.

What it is *not* a function of is the solo scores. `ARCH.P0709` removes the
campaign's **two best-behaved blocks** — 7 at 11/15 and 9 at 10/15, both sealed
per-child `alive-parent-band` — and reads **4/15 with seven charged lanes**,
while 13+1, whose blocks read 13/15 and 9/15, reads **7/15**. Wave 14's
absorbed-fraction census was inverted at the top; the natural repair —
*rank by the behavioural solo score and take the top two* — is also wrong.
Adjacent conv blocks are doing related work and the survivors cannot absorb both.

**No triple was built.** The rule was one only if a pair held ≥9/15, and the best
pair is 7/15. Wave 16's 13+11+9 triple (0/15) stands as the only three-block
datum and nothing this wave suggests a different one would do better.

## 3. The head merge, solved properly — and a 2×2 that says which half worked

ADDENDUM 14 §14.3's joint solve is built: per RoPE plane, one rank-one matrix
that has to serve both parents at once, closed form through a 2×2 Hermitian
generalised eigenproblem, with the merged key shared across the GQA group; the
weights recovered by a direction-anchored law-abiding ridge onto wave 16's
average-in-gauge merge; and `v` with `out_proj` alternated as the linear pair
they are with the attention probabilities held fixed. All four cells of the
2×2 were built at the identical **0.979457x**, 16 → 12 query heads:

| | value path OFF | value path ON |
|---|---:|---:|
| **average in gauge** | **5/15** *(wave 16 `ARCH.KV6`)* | **5/15** (`ARCH.KV6A`) |
| **joint relation solve** | **8/15** (`ARCH.KV6JQ`) | **3/15** (`ARCH.KV6J`) |

* **The relation solve is worth three cases.** `ARCH.KV6JQ` at 8/15 is the best
  head merge in the campaign and seals per-lane **alive-degraded**. It is the
  first evidence in seventeen waves that a *better objective* on an attention
  component buys behaviour.
* **The value/`out_proj` alternation overfits, and it is measurable in both
  directions.** It improves the fit sum-of-squares by **16–29%** at every site
  and moves the held-out per-site cosine **down** (median 0.9332 → 0.9253), and
  behaviourally it costs five cases (8/15 → 3/15). Its λ is chosen by THE LAW's
  trust region, which does not see generalisation; the fix — an internal
  fit-side split for λ — is named and unrun.
* **The identity gates hold exactly.** At trivial grouping (`--kv-groups 8`) the
  q/k/v tensors and the `out_proj` prior come back with **max absolute delta
  0.0** at all six sites, and the re-solve floor is relL2 0.0021–0.0024 at cos
  0.9999977. Fed two copies of one head, the rank-one solver returns that head's
  relation to **2.5e-16**.
* **The pair guard fires often.** The solved target always beats the average on
  the objective, but the *installable* head need not — the target rows are a
  rank-one combination of two post-normed row blocks and no linear map followed
  by an RMS norm has to be able to produce them. The builder keeps the joint head
  only when it wins on the same rows: **5 of 12 pairs** at 16 → 12, **8 of 24** at
  16 → 8. Both numbers are sealed either way.
* **16 → 8 is dead.** `ARCH.KV4JQ`, 0.958914x, **0/15**: nine lanes loop
  `I'm sorry` to the cap, safety included. The merge axis dies at the second step
  like every other axis in this campaign.

**The first joint build was worse than the baseline, and the receipt said why.**
`ARCH.KV6J`'s first build read post-norm cosine **−0.9998** at every site: a
rank-one factorisation has a per-plane unit-modulus gauge, left free it landed on
the *reflection* of the average-in-gauge merge, and the prior-anchored ridge then
spent its whole trust region walking to the far side of its own prior. Fixing the
phase by alignment to that merge, and clamping the per-row radius non-negative
(a per-row sign flip is not a gauge — it flips one query row's scores against
every key), is the difference between 0.8322 and 0.9332 median site cosine. The
receipt of the unfixed build is kept at
`wave17/probes/KV6J-v1-unfixed-gauge/`.

**And the self-test caught the solver defect before a child did.** Unpreconditioned
CG made no progress in 400 iterations on a synthetic problem two orders of
magnitude better conditioned than the real one, and needed 20,000 to reach the
closed form; a 120-iteration cap would have silently returned the prior while the
receipt said a solve had run. The preconditioner is T7's own two-`eigh` Stein
solve on the group-summed factors, so at one Kronecker term the preconditioner
*is* the operator and CG converges in one step — which the self-test now asserts.

## 4. Composition, again, and the same wall

| child | recipe | net | score |
|---|---|---:|---:|
| `ARCH.X13KV6JQ` | block 13 + joint merge | 0.926936x | 4/15 |
| `ARCH.X13KV6J` | block 13 + joint merge + value path | 0.926936x | 3/15 |
| `ARCH.F1301KV6Q` | blocks 13+1 + joint merge | 0.874415x | **0/15** |
| `ARCH.F1301W1536` | blocks 13+1 + `intermediate_size` 1536 | 0.730614x | **0/15** |

13 alone is 13/15 and the merge alone is 8/15; together they are 4/15. The best
pair (7/15) plus the merge is 0/15. **Composition is not monotone in the
components' solo scores at any level this campaign has measured** — not for two
excisions, not for an excision plus a merge, not for an excision plus a width.

`ARCH.F1301W1536` at 0.730614x is the deepest child of the wave and it echoes the
prompt back verbatim in fourteen of sixteen lanes. **The frontier does not move:
it is still wave 16's `ARCH.W1536X13`, 0.769439x at 4/15.** The deepest wave-17
child with a non-zero score is `ARCH.P1301` at **0.894958x, 7/15**, and the
deepest per-child **alive** child in the campaign is still `ARCH.X07` at
0.947479x, 11/15.

**Distance to 0.5x.** 0.5x is 114,846,592 parameters; the tied embedding is
67,108,864 = 0.2922x and no axis in the R3-standard set touches it, so the
162.58M body must fall 3.4x. From 0.769439x that is another 61.9M parameters.
The measured reason it stops is unchanged and now better supported: **every axis
that removes real body mass dies alone or dies at the second composition step**,
and this wave adds that the second step is fatal even when both pieces are
individually strong and even when the second piece is the best-solved component
in the campaign.

## 5. Benchmarks — and the review and the benchmarks disagree the other way now

lm-eval, `limit=200 seed=1234 batch_size=4 max_length=2048 dtype=float32`, SM86
RTX 3090. The parent row reproduces the sealed reference in all seven columns.

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | 81.80 |
| `ARCH.X06` | 0.952057 | 8/15 | **0.5300** | 0.4700 | 0.6050 | 0.5200 | 0.3250 | **39.52** | **127.10** |
| `ARCH.X10` | 0.952057 | 7/15 | 0.5100 | 0.4450 | 0.6200 | 0.5200 | 0.3000 | 153.37 | 239.04 |
| `ARCH.KV6JQ` | 0.979457 | 8/15 | 0.4650 | 0.4400 | 0.6100 | 0.5200 | 0.2300 | 104.96 | 241.89 |
| `ARCH.P1301` | 0.894958 | 7/15 | 0.4100 | 0.3750 | 0.5600 | 0.5400 | 0.1750 | **2870.17** | 990.35 |
| `ARCH.P1307` | 0.894958 | 6/15 | 0.4550 | 0.4050 | 0.6050 | 0.5450 | 0.1300 | 1991.53 | 400.16 |
| `ARCH.X13A` | 0.947479 | 13/15 | 0.4600 | 0.4050 | 0.6350 | 0.5200 | 0.3350 | 106.78 | 297.09 *(wave 16)* |

Wave 16 found the review ranking a child *above* what the benchmarks would
support. This wave finds the opposite, on the same surface:

* **`ARCH.X06` is the cheapest real cut the campaign has found.** One attention
  block, 4.6M parameters, **0.015 of arc_easy** and 1.6x lambada perplexity —
  against `ARCH.X13A`, which the review ranks five cases higher and which costs
  **4.4x** perplexity and 3.6x wikitext. The review says 13 > 8; the distribution
  says X06 is a far less damaged language model.
* **The pair children are not models.** `ARCH.P1301` reads 7/15 on fifteen
  prompts and has a lambada perplexity of **2,870** — 118x the parent's. A pair
  excision is a behavioural curiosity and a language-modelling wreck, and the
  frontier claim that quotes only its 7/15 would be the error ADDENDUM 8 exists
  to stop.

## 5a. One free probe the closed census made possible: the lanes fail in an order

`run/mx17/mx17_lane_fragility.py`, no GPU, reading only the sealed per-case
verdicts of the thirteen single-block children that have them (block 5's reading
predates the per-case file). It counts, for each of the fifteen lanes, how many
single-block excisions leave that lane useful or partial.

```
9/13  instruction-three-actions, writing-concise-email,
      planning-dependencies, uncertainty-insufficient-data
8/13  creative-continuity
6/13  context-retrieval, summarization-priorities,
      comparison-ram-rom, conversation-natural-greeting
4/13  arithmetic-multi-step
3/13  coding-deduplicate
2/13  logic-ordering, debugging-off-by-one, multi-turn-correction
1/13  explanation-seasons
```

**A block excision does not damage lanes uniformly; it walks down a fixed
order.** Generation-shaped lanes with no single correct answer (three actions, an
email, a plan, an admission of insufficient data) survive nine of thirteen
excisions. Lanes with one verifiable answer and a chain to get there — ordering,
off-by-one, arithmetic, dedup — survive two to four. The two hardest lanes are
the two the *parent itself* fails (seasons, the multi-turn correction), so a
child that keeps them is not the point; what the order says is that **depth
buys multi-step correctness and not fluency**, which is why the 15-prompt review
and the perplexity columns can disagree in either direction and both be right.

It is a count over thirteen children, it orders attempts, and it gates nothing.

## 6. Estate

* **Commits.** `0a7d8f62` (ADDENDUM 14, the wave-17 builder, the two grading
  repairs and their tests), `df735362` (the CG preconditioner the self-test
  demanded), `049326e0` (the per-plane phase gauge and the per-pair guard), plus
  the sealing commit for this note. Nothing pushed, nothing rebased, staged by
  explicit path only; the Codex lane's ~331 uncommitted paths, including the
  frozen `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.
* **Tests.** `tests/test_matrix_addendum9_per_lane.py` (extended) and the new
  `tests/test_matrix_addendum14_lane_coverage.py` run the repairs through the CLI
  where they broke: **90 passed, 0 failed** on the deployment, with
  `test_matrix_addendum6.py`, `test_matrix_addendum12.py` and
  `test_matrix_cell_receipt.py`. The joint solver's own gate,
  `run/mx17/mx17_joint_selftest.py`, **passes all five** checks.
* **Ledger.** **525 → 578**, sha
  `be452cffb4d235e6ba85fff30132b38843c03154d742b7f4b174372d205f3a8e`. Backed up
  first (`backup/ledger.jsonl.pre-wave17-20260813T040740`, verified by sha **and**
  `cmp`), merged exactly once behind `wave17/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards and again before the repository copy was
  overwritten. Fifty-three rows: 21 ARCH run rows, 21 ADDENDUM 6 verdict rows and
  **11 ADDENDUM 14 re-seals of wave-16 verdicts**, no superseded rows dropped.
* **Seals.** `run/mx17/mx17_verify_seals.py` with `wave17/cells` added: **13
  roots present, 0 undeclared; 578 rows, 574 resolve, 574 verify, 0 fail, 4 do
  not resolve** — the same four pre-existing non-resolvers waves 14–16 named.
* **The re-seals.** Eleven wave-16 verdicts were re-sealed as **new instances**
  (`-a6review-w17`), never edited. `X12`, `D3`, `R2`, `R3`, `R4`, `R5` move from
  `lane-scoped (15 of 16)` to **`general (16 of 16)`** and their per-lane verdict
  from `impaired` to `dead`; `X03`, `X07`, `X09`, `X11` move from `unstated` to
  **`none`** and finally carry the per-lane verdicts their text earned
  (alive-parent-band, alive-parent-band, alive-degraded, impaired); `X13A` is
  re-sealed unchanged at 2 of 16 to record that the sixteenth lane was chargeable
  and was not charged. Every count came off the wave-16 receipt itself.
* **Disk.** Server `/` began and ended at 13 GB free. Deployments end at
  current+1 — `35dfe3a9` (wave 16) and `049326e0` (wave 17) — with `1c197ed4`
  and `df735362` pruned under a manifest
  (`wave17/deployments-prune-manifest-wave17.json`; 3,986 files, 188,669,936
  bytes total). Twenty-two children were built; **six are kept** (`X06`,
  `KV6JQ`, `P1301`, `P1303`, `X10`, `X13KV6JQ`) and the other sixteen are deleted
  and named with their sealed tensor digests in
  `wave17/children-retention-manifest-wave17.json`, freeing 6.83 GB. The wave's
  `/mnt/ai_data` footprint is 2.4 GB.
* **Human-review package, in the repository.**
  `reports/matrix-campaign-20260810/wave17/human-review/` — 148 files: every
  dossier, every child and parent response rendered verbatim, every run receipt,
  every verdict receipt, every per-case reading and every build receipt.

## 7. What the wave did not spend, and one thing it could not attribute

* **Unrun deck probes.** XGRAM retrospective calibration, the gap-4 per-case
  attribution artifact, the CBT identity-permutation cell and the F-LANDMARK
  census were budgeted after the census, the pair map and the merge, and those
  three consumed the wave. None was killed; none was started.
* **The attention-recompute gate refused in the two-excision composition, and
  the reason is not established.** With blocks 13 **and** 1 excised the gate
  measured **0.32–0.77** against a 5e-2 tolerance at all six sites and refused
  the value path, so `ARCH.F1301KV6` and `ARCH.F1301KV6Q` are **bit-identical**
  (tensor digest `90a9e5187feef678`, verified) and their sixteen responses match
  character for character. With block 13 alone the same gate reads 9.3e-3–1.4e-2
  and passes. The gate did its job — it refused to solve against probabilities
  the running model does not share — but *why* a second excision breaks the
  agreement is unattributed. The leading hypothesis is that the composed child's
  attention is near-degenerate and bf16 rounding in the module decides the
  argmax where the fp64 recomputation does not; the falsifying probe (measure the
  attention peakedness per site, and repeat the recomputation in bf16) is named
  and unrun.
* **The value-path overfit has a named fix that was not built:** choose λ for
  the two-sided solve on an internal split of the *fit* rows, never on the
  campaign's held-out block.
* **One cosmetic defect shipped:** `mx17_build.py`'s log prefix still reads
  `[mx16]`. Every receipt, schema string and filename is wave 17's; only the log
  line is wrong.

## 0. The sentences

1. **The first conv block is the most expensive block in the model, and the
   census that says so is now complete.** `ARCH.X00` removes the same 12.1M
   parameters as the 13/15 child and answers **nothing** — 0/15 with **all
   sixteen lanes charged**, the first `general` degeneracy reading in the
   campaign.
2. **Attention is not a class.** Blocks 2 and 12 are 0/15, block 4 is 2/15, and
   blocks 6, 8 and 10 read 8, 7 and 7 — inside the conv band. The axis has no
   price; the site does.
3. **The cost of removing two blocks is a distance law.** Across four distances
   and two anchors: distance 2 gives 2, 2, 4 of 15; distance 4 gives 6, 6;
   distance 6 gives 6; distances 10 and 12 give 7 and 7. **Monotone, no
   inversions.**
4. **Pair cost does not track solo scores.** The campaign's two best-behaved
   blocks — 11/15 and 10/15, both per-child *alive* — compose to **4/15** with
   seven charged lanes, while 13/15 + 9/15 twelve blocks apart compose to 7/15.
5. **T7's joint solve is worth three cases and the half of it that fits the value
   path is worth minus five.** 16 → 12 query heads: average in gauge 5/15, joint
   relation solve **8/15**, joint plus the value alternation 3/15 — at one
   parameter count, one surface, one wave. The value alternation improves its own
   fit objective by 16–29% and loses held-out cosine while doing it.
6. **The solve's own gates fired and were worth having.** Two identical heads
   return that head to 2.5e-16; trivial grouping is bit-exact; the first build
   landed on the reflection of its own prior at −0.9998 post-norm cosine and was
   rebuilt; unpreconditioned CG would have silently returned the prior.
7. **Nothing composes.** 13 (13/15) + the merge (8/15) is 4/15; the best pair
   (7/15) + the merge is 0/15; the best pair + width at 0.730614x echoes the
   prompt back. The frontier is still wave 16's 0.769439x at 4/15.
8. **The review and the benchmarks disagree again, and this time the review is
   the optimistic one.** `ARCH.X06` — one attention block, 4.6M parameters —
   costs **0.015 arc_easy and 1.6x lambada perplexity**, where the 13/15 child
   costs 4.4x; and `ARCH.P1301` reads 7/15 with a lambada perplexity of
   **2,870**.
