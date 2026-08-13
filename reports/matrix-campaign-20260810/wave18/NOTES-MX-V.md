# NOTES-MX-V — wave 18: the distance law dies at three bodies, the overfit is fixed, and the gate was right about something nobody guessed

Wave 18, branch `codex/lrc-blank-compression`, source commits `a1500995` (the
wave-18 builder: the internal fit-side split for λ and the gate probe),
`bf946174` (two repairs to those additions before either built a child),
`c66c38ed` (the logit regression the first probe run demanded) and `b372d47c`
(**the causal-mask repair**), on top of wave-17 HEAD `92e90aa9`. Three
deployments were cut because two of the wave's findings changed the builder
mid-wave; every child receipt names the one that built it, and each is a commit.
Protocol `e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` and
prompt pack `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`
verified on every extraction and every review.

**Three calibrations passed before anything was graded.** The fresh conditioned
capture reproduces the sealed collector's rows to wave 15's numbers exactly (L00
relL2 3.367e-3 cos 0.9999944, L05 1.647e-2 / 0.9997207, L13 2.034e-2 /
0.9997842). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23`** on **all fourteen** wave-18 dossiers. And the
parent's benchmark row reproduces the sealed reference in all seven columns
(0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl 24.39, wikitext 81.80).

*§0's sentences are at the bottom.*

---

## 1. The distance law does not survive to three bodies

Wave 17's one clean law: across eight pairs, four distances and two anchors, the
cost of removing two blocks tracked the **distance between them** and nothing
else — 2 apart read 2, 2, 4 of fifteen; twelve apart read 7. Its direct
prediction is that a widely-spread *triple* should be cheap. Four were built, in
descending order of spread, all conv, all at the identical **0.842437x**:

| triple | pairwise distances | score | charged | per-lane |
|---|---|---:|---:|---|
| **1+7+13** | 6, 6, 12 | **3/15** | 5 | impaired |
| 3+9+13 | 6, 4, 10 | 2/15 | 4 | impaired |
| 1+9+13 | 8, 4, 12 | 1/15 | 8 | impaired |
| 3+7+13 | 4, 6, 10 | 0/15 | 8 | impaired |
| 9+11+13 *(wave 16)* | 2, 2, 4 | 0/15 | — | — |

**The law's sign survives and its magnitude does not.** The tightest triple in
the model is 0/15 and the widest is 3/15, so spread still helps — but the best
pair reads **7/15** and the best triple reads **3/15**, and the ordering inside
the middle is not monotone in either the minimum pairwise distance (3+9+13 and
1+9+13 both have a 4 in them and read 2 and 1) or the sum of them (both 24-spread
triples read 3 and 1, both 20-spread triples read 0 and 2). Two blocks is where
distance is a law; three is where it is a tendency with a large residual.

**No quad was built**, and that was decided before the triples were: the rule was
one only if a triple held ≥6/15. The best held 3/15. A quad at 0.789916x would
have been the deepest meaningful child in the campaign and there is no reading of
these four that supports building it.

What *does* hold at three bodies is the count. Best single 13/15, best pair
7/15, best triple 3/15 — roughly **three to five review cases per block
removed**, which is the additive-interference rule the scaling document already
states, and which distance modulates rather than replaces.

**And the triples are not language models.** `ARCH.T010713`, the 3/15 child, has
a lambada perplexity of **14,861** against the parent's 24.39 — 609x — and 0.39
arc_easy. Wave 17 wrote that a pair excision is "a behavioural curiosity and a
language-modelling wreck"; a triple is the same sentence with another order of
magnitude in it.

## 2. The λ the value path overfits, fixed — and it is worth four of the five cases

`NOTES-MX-U` §7 named the fix and did not build it: choose the value/`out_proj`
alternation's λ on an **internal split of the FIT rows**, never on THE LAW's norm
alone. It is built. Eighteen of the campaign's twenty-four captured batches are
fit; every third one of those is withheld from each rung's solve and used only to
score it, the admitted rung with the lowest check objective is re-solved on all
eighteen, and THE LAW keeps its veto and loses its vote. The campaign's own
held-out block is never read by the selection.

The 2×2 wave 17 built, with the new cell in it, all at **0.979457x**, 16 → 12:

| | value path OFF | value path ON |
|---|---:|---:|
| **average in gauge** | 5/15 *(wave 16 `KV6`)* | 5/15 *(wave 17 `KV6A`)* |
| **joint relation solve** | **8/15** *(wave 17 `KV6JQ`)* | 3/15 *(wave 17 `KV6J`)* |
| **joint + λ on an internal fit split** | — | **7/15** (`ARCH.KV6JS`) |

**3/15 → 7/15 for a change to how one scalar is chosen.** The mechanism is
visible in the receipt and it is not subtle: the split picks λ_rel = **100.0**,
the top of the ladder, at eleven of the twelve merged groups, where THE LAW's own
rule picked **0.001 to 0.01**. Three to five orders of magnitude. The fit
sum-of-squares improvement collapses from wave 17's **16–29%** to **0.004–0.09%**,
and the median held-out per-site cosine goes from wave 17's degraded **0.9253**
back to **0.9333** — which is `KV6JQ`'s own number to four places.

So the honest reading is: *given this evidence, the correct amount to move the
value path is almost none, and an internal split is what discovers that.* The
one case still missing against the value-path-off child is the residual of
moving it at all.

The benchmarks say the same thing from the other side and are worth quoting
because they are the case where the review and the distribution finally agree:

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `KV6JQ` *(w17)* | 0.979457 | 8/15 | 0.4650 | 0.4400 | 0.6100 | 0.5200 | 0.2300 | 104.96 | 241.89 |
| **`KV6JS`** | 0.979457 | **7/15** | 0.4650 | 0.4350 | 0.6100 | 0.5350 | 0.2350 | **103.45** | **240.46** |
| `X07KV6JQC` | 0.926936 | 7/15 | 0.4200 | 0.4200 | 0.5350 | 0.5450 | 0.1200 | 740.48 | 377.65 |
| `T010713` | 0.842437 | 3/15 | 0.3900 | 0.4150 | 0.5350 | 0.4600 | 0.1000 | **14860.55** | 1089.79 |

`KV6JS` and `KV6JQ` are the same model to within noise on all seven columns. The
fixed value path is free; the unfixed one cost five review cases.

**16 → 8 is still dead.** `ARCH.KV4JS`, the second merge step rebuilt with the
full fixed solve, is **0/15** with eight of sixteen lanes — the safety lane
included — looping `I'm sorry` to the 1,024-token cap, which is exactly what
`ARCH.KV4JQ` did without the fix. The merge axis dies at its second step and a
better-regularised solve does not save it.

## 3. The merge composes with an excision, and NOT by distance

`KV6JQ` touches all six attention blocks, so "distance" from the merge to an
excised block is a distance to the nearest attention site. Four compositions,
all joint q/k with the value path off, all causal-repaired except where noted:

| child | excision | distance to nearest attention site | net | score |
|---|---|---:|---:|---:|
| `ARCH.X07KV6JQC` | block 7 | 1 | 0.926936x | **7/15** |
| `ARCH.X09KV6JQC` | block 9 | 1 | 0.926936x | 5/15 |
| `ARCH.X13KV6JQ` *(w17)* | block 13 | 1 past the last | 0.926936x | 4/15 |
| `ARCH.F1301KV6QC` | blocks 13+1 | — | 0.874415x | 1/15 |
| `ARCH.T010713KV6JQC` | blocks 1+7+13 | — | 0.821894x | 0/15 |

**The excision nearest the attention mass composes best**, which is the opposite
of what a distance law would predict, and `X07KV6JQC` at 7/15 is the best
excision-plus-merge composition the campaign has produced — three cases above
wave 17's `X13KV6JQ` at the identical parameter count. What predicts the
composition is not the distance but the *solo* score: 7 (11/15) > 9 (10/15) >
13 (13/15 but at the end of the stack) and then the pair and the triple.
Composition is still not monotone — 11/15 and 8/15 compose to 7/15 — but for the
first time in the campaign a two-axis child sits **at** the better component's
level rather than far below both.

## 4. The anomaly is attributed, and it is a defect in the harness, not in bf16

Wave 17 left one thing unexplained: with blocks 13 **and** 1 excised the
attention-recompute gate read **0.32–0.77** against a 5e-2 tolerance at all six
sites and refused the value path, where block 13 alone reads 9.3e-3. The leading
hypothesis was bf16-versus-fp64 argmax divergence in a near-degenerate composed
attention. **It is wrong, and the gate was right about something better.**

The probe ran in three legs and then a fourth the first three demanded.

1. **The module's own probabilities.** This architecture's eager attention
   returns its `attn_weights`, so the fp64 recomputation can be checked against
   the tensor it models. Fed the module's own probabilities, the reconstructed
   context matches the installed one at **1.9e-3 to 2.5e-3 at every site of
   every child, the 13+1 composition included** — the bf16 floor. The
   disagreement is *entirely* in the probabilities: not `v`, not the capture
   alignment, not the head-to-group map.
2. **Peakedness.** The hypothesis needs near-ties and there are none. The 13+1
   child's *fp64* top-two margins are the healthy child's (median 0.0786 against
   0.0766 at L08). What moved is the **module's** side, and it moved one way:
   median max-probability **0.2139 → 0.1055**, entropy **3.123 → 4.361**. A
   one-sided flattening is not two arithmetics coin-flipping an argmax.
3. **The logit regression.** Per-row centred `log p_module` on `log p_fp64`:
   slope **1.0000**, residual RMS **0.015–0.021** against a logit spread of
   **1.9–2.1**. The logits were never what differed. Only the set they were
   normalised over.
4. **`mx18_mask_probe.py`, three loads and no solve.** `Lfm2Model.forward`
   chooses each block's mask by position in a config list —
   `causal_mask if config.layer_types[i] == "full_attention" else linear_attention`
   — and the excision path deleted modules from the layer list **without
   deleting their types**. Index `i` therefore addressed the parent's fourteen,
   so every attention block an excision shifted down into a conv's slot was
   handed the padding mask, which is `None` on this corpus, instead of the
   causal one:

| case | config type at each attention position | probabilities relL2 | module's attention mass on the future |
|---|---|---:|---:|
| excise 13 (nothing shifts) | `full_attention` ×6 | 1.1e-2 – 1.6e-2 | **0.0000** |
| excise 13+1 (all six shift) | `conv` ×6 | 5.1e-1 – 1.08 | **0.286 – 0.480** |
| excise 13+1, `layer_types` repaired | `full_attention` ×6 | 1.4e-2 – 1.6e-2 | **0.0000** |

**Between 28.6% and 48.0% of the live model's attention mass was on tokens that
had not happened yet.** The gate refused to solve against probabilities the
running model does not share, and the reason it did not share them is that the
running model was reading the future.

### 4.1 The blast radius, measured rather than feared

The first estimate of the damage was wrong and the receipts said so. **A pure
excision build runs no solve at all** — one axis, `num_hidden_layers`, no
per-site write, the conditioned capture taken and never used — so the live
stream cannot reach the child. Five rebuilds against the repaired harness prove
it by digest:

| rebuilt | reproduces | digest |
|---|---|---|
| `X13C` | wave 16 `X13A` (13/15) | `da70dc276bdeca386b5fd25ad3a6081a` |
| `X00C` | wave 17 `X00` (0/15) | `90e6e81712a9095609a2310afc4502b6` |
| `X07C` | wave 16 `X07` (11/15) | `0ebf615078a7b3987e0ba46b27d5eec1` |
| `P1301C` | wave 17 `P1301` (7/15) | `b90c9609e6c88627d56b63be7d90cb31` |
| `T010713C` | this wave's triple (3/15) | `c3417c522c30de773d815f4a4c78b654` |

**Bit-identical, every one.** The single-block census, the pair map and the
triple map are untouched, and they did not need re-reading — a rebuild of the
same bytes cannot produce a different dossier.

What the defect could reach is a child that excises **and** solves against the
live stream, which is every composed child whose excision moves an attention
block. Wave 16's frontier `W1536X13` and wave 17's `X13KV6J*` excise only block
13 and are clean. These were not, and the price is small and real:

| recipe | with the defect | repaired | Δ |
|---|---:|---:|---:|
| block 7 + joint merge | 6/15, 4 lanes charged | **7/15**, 3 charged | +1 |
| blocks 13+1 + joint merge | 0/15 *(wave 17)* | 1/15 | +1 |
| blocks 13+1 + `intermediate_size` 1536 | 0/15 *(wave 17)* | **0/15** | 0 |

**About one review case.** The composition wall is not this defect. It is worth
saying plainly because the temptation on finding a bug this size is to hope it
explains the campaign's central negative result, and it does not.

## 5. Width on the best, and the frontier that does not move

| child | recipe | net | score |
|---|---|---:|---:|
| `ARCH.F1301W1536C` | blocks 13+1, `intermediate_size` 1536 | 0.730614x | 0/15 |
| `ARCH.T010713W1536C` | blocks 1+7+13, 1536 | 0.691788x | 0/15 |
| `ARCH.T010713W1280C` | blocks 1+7+13, 1280 | 0.654126x | 0/15 |

Wave 16's lesson — width and excision compose terribly — was tested with the
blocks the distance law prefers and with a harness that no longer reads the
future, and it survives both. The 1536 rung on the widest triple returns the
empty string on the greeting lane and fills the cap with asterisks on three
more; the 1280 rung below it echoes the prompt verbatim in five lanes. The
bisection has nothing to bisect: both rungs are zero.

**The frontier does not move. It is still wave 16's `ARCH.W1536X13`, 0.769439x
at 4/15** — and it is worth noting that this child excises only block 13, so it
never touched the defect and needs no asterisk. The deepest wave-18 child with a
non-zero score is `ARCH.T010713` at **0.842437x, 3/15**.

**Distance to 0.5x.** 0.5x is 114,846,592 parameters; the tied embedding is
67,108,864 = 0.2922x and no axis in the R3-standard set touches it, so the
162.58M body must fall 3.4x. From 0.769439x that is another **61.9M parameters**.
The measured reason it stops is now stated one notch more precisely than wave 17
could state it: **the second composition step is fatal, the first is not**, and
wave 18 is the first wave to find a two-axis child (`X07KV6JQC`, 7/15) sitting at
its better component's level instead of far below both. Every three-block and
every excision-plus-width child in this wave is zero.

## 6. Estate

* **Commits.** `a1500995` (the wave-18 builder: the internal fit-side split, the
  gate probe, the `[mx18]` log prefix, the wave-18 build-receipt schema and its
  registration in the ADDENDUM 12 sealer, plus `tests/test_matrix_lambda_split.py`),
  `bf946174` (the ladder's unpriced rungs and a probe with nothing to compare),
  `c66c38ed` (the logit regression and the honest relabelling of the
  input-rounding leg as the null it is), `b372d47c` (the causal-mask repair and
  `mx18_mask_probe.py`), plus the sealing commit for this note. Nothing pushed,
  nothing rebased, staged by explicit path only; the Codex lane's uncommitted
  paths, including the frozen `src/neural_forge/hf_lfm2_circuit_trace_cache.py`,
  were not touched.
* **Tests.** **94 passed, 0 failed** on the build deployment
  (`test_matrix_lambda_split.py` — new, four cases on the λ fix — plus
  `test_matrix_addendum9_per_lane.py`, `test_matrix_addendum14_lane_coverage.py`,
  `test_matrix_addendum6.py`, `test_matrix_addendum12.py`,
  `test_matrix_cell_receipt.py`). `mx18_joint_selftest.py` **passes all nine**
  checks, four of them wave 18's: the check objective is the sum of squares it
  stands for to 1e-10, the hoisted `eigh` reproduces the in-solve one to zero
  absolute delta, and the split generalises better than THE LAW's own pick on
  both sides of the alternation.
* **Ledger.** **578 → 606**, sha
  `9dab0d5c760715a27037cd525bf3769c5047537eb15b3d3b84ddc4fa0083f573`. Backed up
  first (`backup/ledger.jsonl.pre-wave18-20260813T060017`, verified by sha **and**
  `cmp`), merged exactly once behind `wave18/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards and again in the repository copy. Twenty-eight
  rows: 14 ARCH run rows and 14 ADDENDUM 6 verdict rows, no superseded rows
  dropped.
* **One merge refusal, and it was the guard working.** `sealq.sh` was run twice
  by accident. The verdict receipt digest carries `sealed_at`, so the second run
  produced a second distinct instance of every reading, and the merge **refused**
  rather than guess which of two receipts for one measurement was real. Nothing
  was appended — the ledger was byte-identical to the backup at that point, which
  is how the refusal was safe to undo — the duplicate shards were moved to
  `wave18/superseded/`, every verdict was sealed exactly once, and the merge was
  re-run.
* **Seals.** `mx18_verify_seals.py --deployment <build deployment>`: **14 roots
  present, 0 undeclared; 606 rows, 602 resolve, 602 verify, 0 fail, 4 do not
  resolve** — the same four pre-existing non-resolvers waves 14–17 named. The
  root audit earned its keep this wave: forking the verifier by rewriting every
  `wave17` to `wave18` silently deleted last wave's root, and the audit refused
  to run until it was put back.
* **Disk.** Server `/` began and ended at 13 GB free. Deployments end at
  current+1 — `b372d47c` (wave 18) and `049326e0` (wave 17) — with `35dfe3a9`,
  `bf946174` and `c66c38ed` pruned under a manifest
  (`wave18/deployments-prune-manifest-wave18.json`). Twenty-one children were
  built; **five are kept** (`KV6JS`, `X07KV6JQC`, `X07KV6JQ`, `X09KV6JQC`,
  `T010713`) and the other sixteen are deleted and named with their sealed tensor
  digests in `wave18/children-retention-manifest-wave18.json`, freeing 6.32 GB.
  Six probe exports were deleted after their measurements were sealed, freeing a
  further 2.4 GB. The wave's `/mnt/ai_data` footprint is 2.0 GB.
* **Human-review package, in the repository.**
  `reports/matrix-campaign-20260810/wave18/human-review/` — 111 files: every
  dossier, every child and parent response rendered verbatim, every run receipt,
  every verdict receipt, every per-case reading, every build receipt, and the
  gate-probe and mask-probe measurements.

## 7. What the wave did not spend, and what it left standing

* **The quad was not built**, by the rule that was fixed before the triples were
  read. It remains the one obvious unbuilt child on the excision axis, and
  nothing in the four triples argues for it.
* **The pre-repair composed children of earlier waves were not rebuilt.** Wave
  17's `F1301KV6` and `F1301KV6Q` (bit-identical to each other, 0/15) and
  `F1301W1536` (0/15) were built by the defective harness; two of the three were
  rebuilt here and moved by at most one case, so the third was not spent on. Any
  earlier-wave child that both excises a block before an attention block and
  solves against the live stream carries the same asterisk and the same measured
  size of error.
* **The unrun deck probes are still unrun.** XGRAM retrospective calibration and
  the gap-4 per-case attribution artifact were budgeted after the triples, the λ
  fix and the anomaly, and the anomaly — which turned into a harness repair, five
  identity rebuilds and seven rebuilt composed children — consumed the wave.
  Neither was killed; neither was started.
* **The input-rounding leg of the gate probe is a null and is labelled one.** The
  captured input and the stored weights are already bf16, so rounding them
  reproduces the fp64 leg to every digit. It cannot reach the module's
  arithmetic. It is kept because it is the leg `NOTES-MX-U` §7 asked for and a
  reader is owed the reason it answers nothing.

## 0. The sentences

1. **The distance law does not survive to three bodies.** The widest triple in
   the model — blocks 1, 7 and 13, pairwise distances 6, 6 and 12 — reads
   **3/15** where the best *pair* reads 7/15, and the ordering among the four
   triples is not monotone in either the minimum or the total pairwise distance.
   Spread still helps; it stopped being a law.
2. **No quad was built, by a rule fixed before the triples were read.** The gate
   was a triple at ≥6/15 and the best was 3/15.
3. **Choosing one scalar out of sample is worth four review cases.** The value
   path's λ, selected on an internal split of the fit rows instead of on THE
   LAW's norm, takes the merge from **3/15 to 7/15** at one parameter count and
   an unchanged benchmark row. The split picks λ_rel **100** where THE LAW picked
   **0.001–0.01**, and the fit-objective improvement it gives up — 16–29% down to
   0.004–0.09% — was the overfit.
4. **The merge composes best with the excision NEAREST it, which is the opposite
   of a distance law.** Block 7 plus the joint merge is **7/15**, the best
   excision-plus-merge child in the campaign and the first two-axis child that
   sits at its better component's level instead of far below both.
5. **The attention gate was right, and about a defect nobody guessed.** Excising
   a block shifted every later attention block into a conv's slot in
   `config.layer_types`, and `Lfm2Model` hands a block its mask by that index —
   so **28.6% to 48.0% of the live model's attention mass was on tokens that had
   not happened yet.** The bf16 hypothesis is dead: the module's logits regress
   on the recomputation's with slope **1.0000**.
6. **The blast radius is one review case, and the census is untouched.** A pure
   excision runs no solve, so five rebuilds against the repaired harness came
   back **bit-identical** to the children they reproduce. Only composed children
   moved, and by at most one case.
7. **Nothing composes at three blocks or with width.** Every three-block child,
   every excision-plus-width child and the triple-plus-merge child is **0/15**;
   `T010713` at 0.842437x has a lambada perplexity of **14,861**.
8. **The frontier does not move.** It is still wave 16's `ARCH.W1536X13`,
   0.769439x at 4/15 — a child that excises only block 13, never touched the
   defect, and needs no asterisk. 0.5x is another 61.9M parameters away.
