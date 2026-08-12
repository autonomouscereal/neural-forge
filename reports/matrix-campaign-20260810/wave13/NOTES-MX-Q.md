# NOTES-MX-Q — wave 13: the last mechanism, and the rule that stopped being a gate

Wave 13, branch `codex/lrc-blank-compression`, source commit **`63e95cd8`**
("Give the planner a width the kernel will actually accept") on top of
**`ebc16da4`** ("Give C10 a kernel that can state a quarter, and correct the
arithmetic that said it could"), both on top of wave-12 HEAD `d14c9850`.
Deployment `neural-forge-matrix-63e95cd8`, cut with
`git -c core.autocrlf=false archive`, review surface verified on extraction
**before any child was generated**: protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, seal
`7b859172fa48c008eb68d5ea070458ac82e2149c19c5141230d45c061aaf00e5`, prompt pack
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`. Knob file
`e355e4e7bb8a3058…`, byte-identical to waves 7-12.

Wave 13 had one mechanism left on the matrix — **M5, the Procrustes head
merge** — and a closing report to write. Mid-wave, Cyril changed the rules the
report was going to be written under, and the change is the more important half
of this note.

*§0's sentences are at the bottom, because they could not be written until the
last receipt was sealed.*

---

## 1. The directive, and what it revoked

Three instructions arrived after the kernel was committed and before the first
stack was built.

1. **Never gate on drift.** The drift law (`review = 12.9554 − 13.1626 × drift`,
   R² = 0.6243) is a **prioritization heuristic** — which lane to try first —
   and never a verdict. Wave 13's own step-2(c) instruction ("if the solo prices
   poorly, seal it and stop") was revoked in flight: whatever M5 priced at, the
   stacks get built and read. The standing proof is `COMP.MA.MXO3`, coherent at
   drift 0.4618 against a nominal budget of 0.3765.
2. **Never gate on efficiency.** Wave 12 retired corrective routes on a 355×
   parameter-efficiency comparison. Wrong frame: 2.1M parameters is **0.91% of
   this model** and bought two review cases with no gradient step. Efficiency
   comparisons stay as recorded diagnostics; they never kill a mechanism. Two
   children were added to this wave because of that: **routes on top of the
   shopped child** (wave 12's central bet, never built) and **a deep shopped set
   with routes on top**.
3. **The frontier report must not declare fold-only closed**, must present drift
   arithmetic as the heuristic that guided search, and must queue the unplayed
   **structural** classes. A fourth instruction followed: **quantization is
   forbidden in this program** — "it is not folding" — and storage stays counted
   in **parameters**, never bytes.

Everything below obeys those. The sentence waves 11 and 12 wrote — *0.5x is
arithmetically out of reach* — is withdrawn in
`docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md` §5, and this wave built the
depth it wrote off.

## 2. M5, the kernel

`NOTES-MX-P` §8 is the specification and it was followed, including where it was
wrong.

**What the mechanism is.** `C10` owns all four attention projections. Head
*drop* — C10's only previous kernel — stores `k_proj`/`v_proj` whole and bottoms
out near 0.5x, which ADDENDUM 6 calls `cannot-express-4x`; no behavioural claim
follows from it. A *merge* stores `g` shared head bases plus one orthogonal
`D×D` alignment per head and reconstructs every head parent-shaped:

```
q_h = R_h Q_g      o_h = O_g R_h^T      k_j = S_j K_g'      v_j = S_j V_g'
stored = g_q*(D*hidden + hidden*D) + g_kv*2*(D*hidden) + (H + H_kv)*D*D
```

**The correction of record.** §8 priced `g = 3` at 688,128 = 0.21875x by writing
the kv term as `(g/2) · 2 · (64·1024)` and evaluating `g/2 = 1.5`. There is no
such thing as one and a half kv groups. Eight kv heads split into
`ceil(3 · 8/16) = 2`, so the implementable `g_q = 3` config stores **753,664 =
0.239583x**. Still under the S4 bar — §8's verdict survives — but the published
number was **8.7% low**, and the kernel refuses to inherit it. `g_q = 4` is
unchanged at 884,736 = 0.28125x and remains a **sub-4x diagnostic**, never a
frontier claim. A third row §8 did not have: `g_q = 1, g_kv = 1` stores
**360,448 = 0.114583x**, so the merge can express **S8**, which head drop never
could at any level.

**Four design decisions worth reading before a receipt.**

* **The grouping is not a heuristic.** Agglomerative clustering whose linkage
  *is* the converged merge objective: at each step the pair of clusters whose
  union costs the least additional residual is merged. 16 heads to 3 groups is
  13 merges and 676 cluster evaluations of 64×64 SVDs, which costs seconds.
* **The alignments are closed form and the alternation is monotone.** Each
  half-step is the exact minimizer of the other's argument (orthogonal
  Procrustes; a weighted mean), so the objective cannot rise, and
  `objective_trace` is sealed so a reader can see that it did not.
* **The O group bases are re-solved against the sealed write.** With the
  alignments fixed, the child's O contribution is linear in the stored bases:
  `context @ o_child.T = Z @ W.T` with `Z = [Σ_{h∈a} context_h R_h]_a`. That is
  `matrix_technique_head_drop`'s absorption step generalized from "the surviving
  columns" to "the shared bases", and it is the only place in the kernel where
  the fit is against what the block **wrote** rather than against how two weight
  matrices look. It cuts held-out write error from 0.52–0.91 to **0.35–0.56**.
* **That refit is adopted only if it wins an INNER split of the fit rows.** A
  least-squares solve is guaranteed to beat its prior only on the rows it was
  fit on, and the held-out rows are the campaign's boundary score, not a
  model-selection set. On the real parent the refit won at every site and was
  adopted; on a 96-row test fixture it loses and is refused, which is the case
  the test pins.

**The identity gate.** At one basis per head the whole pipeline — same
clustering, same alternation, same refit with the parent's own write as target —
must return the parent over all four projections:

| site | `identity_gate_max_abs_deviation` |
|---|---:|
| L02 | 8.708e-15 |
| L04 | 8.882e-16 |
| L08 | 6.661e-16 |
| L10 | 1.055e-15 |
| L12 | 1.055e-15 |

**Registered as T7 pinned to C10**, the shape `T1`-at-`C14` and `T1`-at-`C10`
already use. Minting a `T10` would have widened `TECHNIQUES` and therefore
`known_cell_ids`, which all 442 sealed rows verify against; ADDENDUM 10 went out
of its way to keep those objects byte-identical and this kernel is not what
breaks them. Storage model registered at `("T7","C10")`; **21 cases** in
`tests/test_matrix_technique_head_merge.py`; campaign invocation **309 passed,
1 skipped**.

## 3. The price, measured against a calibrated probe

`mxq_shop_probe.py` is wave 12's probe with two additions: a lane may now be an
**attention** lane (C10 alone) as well as a SwiGLU triple, and it can run a
**greedy ladder** — install the winner permanently and re-price what is left.

Two calibrations, both passed before any number below was read:

* the `COMP.MA.MXO3` base re-reads **0.42132547163418477**, which is MXO3's own
  sealed body drift to eight decimals;
* `L13.T5.S4` prices at **+0.03412939526133918**, which is wave 12's sealed
  **+0.034129**.

**M5's lane price, on that base (`wave13/probes/headmerge-vs-swiglu.json`):**

| lane | marginal drift | removes | drift / M-param | × the cheapest SwiGLU lane |
|---|---:|---:|---:|---:|
| `L13.T5.S4` (SwiGLU reference) | +0.034129 | 5,901,819 | 0.005783 | 1.0 |
| **`L02.T7.S4`** | **+0.110941** | 2,392,064 | 0.046379 | **8.0×** |
| `L08.T7.S4` | +0.157162 | 2,392,064 | 0.065701 | 11.4× |
| `L12.T7.S4` | +0.157983 | 2,392,064 | 0.066045 | 11.4× |
| `L10.T7.S4` | +0.176884 | 2,392,064 | 0.073946 | 12.8× |
| `L04.T7.S4` | +0.183834 | 2,392,064 | 0.076852 | 13.3× |

Every candidate restored the base exactly (`base_restored_exactly: true`, six of
six) and every component satisfied its S4 bound.

**A head-merge lane is not a SwiGLU lane and must not be compared to one on raw
marginal drift.** It removes 2,392,064 parameters against 5,901,819 — 40% of the
purchase. Raw, M5's best site is 3.25× the cheapest SwiGLU lane; per parameter
removed it is **8.0×**. Both numbers are in the table because the second is the
one a storage target cares about and the first is the one a drift budget cares
about.

**Why it prices this way, from the kernel's own telemetry.** The partitions the
objective-linkage clustering finds are degenerate: at L12,
`[[0,8,9,15], [1,2,3,4,5,6,7,10,11,12,13], [14]]`; at L02,
`[[0,1,2,3,6,7,10,15], [4,5,11,12,13], [8,9,14]]`; and the kv partition is
**7 heads + 1 outlier at every single site**. **LFM2's attention heads are not
rotations of a few shared bases.** There is no head redundancy at this scale for
a merge to harvest, and the clustering is merging things that are not alike
because the level told it to.

## 4. The solo ladder — three children, read not predicted

One merged attention block, nothing else folded, net **0.989586x** (1.04% of the
parent removed), reviewed on the frozen 15-prompt surface with the parent
decoded in the same process.

| child | site | drift | review | per-child | per-lane (reading) | lanes charged |
|---|---|---:|---:|---|---|---:|
| `COMP.MA.MXQ1` | L12 | 0.429852 | **6/15** | dead | impaired (lane-scoped) | 1 of 16 |
| `COMP.MA.MXQ2` | L02 | 0.399369 | **6/15** | dead | impaired (lane-scoped) | 4 of 16 |
| `COMP.MA.MXQ3` | L08 | 0.468446 | **4/15** | dead | impaired (lane-scoped) | 6 of 16 |

**One merged attention block costs more behaviour than two whole SwiGLU lanes
plus the entire token interface.** `COMP.MA.MXO3` removes 27.1% of the model and
reads 8/15; `COMP.MA.MXQ1` removes 1.04% and reads 6/15.

**The continuous axis agrees, independently.** `COMP.MA.MXQ1` materialized and
benchmarked under the campaign's fixed settings:

| model | net | review | arc_easy | hellaswag (n) | piqa (n) | winogrande | lambada acc | **lambada ppl** | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | 81.80 |
| `MXQ1` | 0.989586 | 6/15 | 0.5000 | 0.4750 | 0.6650 | **0.5600** | 0.1450 | **341.58** | 105.94 |
| `MXP2` | 0.703409 | 6/15 | 0.4250 | 0.4250 | 0.5850 | 0.5400 | 0.1450 | 857.20 | 231.96 |

Merging one attention block multiplies lambada perplexity **14×** while removing
1% of the model; ten folds that remove 30× more storage multiply it 35×.
(`winogrande` goes *up*, which is what a task 5 points above chance on a 230M
parent does under noise, and is why perplexity is the column that ranks these.)

**And the pair that proves the directive's point.** `MXQ2` has **lower drift**
than `MXQ1` — 0.399369 against 0.429852 — and **four** review lanes that loop to
the 1024-token cap against `MXQ1`'s **one**, with the same-process parent
completing all of them (548 tokens on `planning-dependencies` where `MXQ1` runs
to 1024; 77 tokens on `writing-concise-email` where `MXQ2` runs to 1024). The
drift law ranks these two backwards. **A drift number is a search order, not a
grade**, and this wave has the counter-example in its own receipts.

## 5. A grading note that has to be said out loud

`behavior_verdict_for` returns **`impaired` for any non-degenerate score below
8/15** — it is the calculator's floor band. So `COMP.MA.MXQ3`'s per-lane
`impaired` at **4/15** and `COMP.MA.MXP2`'s per-lane `impaired` at **6/15**
print the same word, while ADDENDUM 8 §8.2's prose means the second and not the
first when it calls `impaired` a success band. **Every per-lane verdict in this
wave is quoted with its score**, and the frontier report does the same. This is
a reporting rule, not a code change: changing the band arithmetic mid-campaign
would re-grade 442 sealed rows.

## 6. What was NOT measured, stated as a gap

* **The gauge-faithful alignment pairing.** §8 ties one rotation to the `(q, o)`
  pair of a head and one to the `(k, v)` pair of a kv head, which is what makes
  24 alignments and what makes `g_q = 3` fit under the bar. A single attention
  head's actual gauge pairs `(q, k)` and `(v, o)` — rotating q and k together
  leaves the scores invariant, rotating v and o together leaves the write
  invariant — so §8's pairing asks one rotation to serve two reconstructions the
  architecture does not couple. Pricing the alternative means a different storage
  table (a separate alignment per projection is 48 of them, 196,608 parameters,
  which pushes `g_q = 3, g_kv = 2` to 0.2708x and **out of S4**), so it is a
  different cell and not a knob. The kernel implements what was registered and
  says so in its docstring.
* **`g = 4` as a diagnostic.** Not run. The `g = 3` price is high enough at every
  site that a 0.28125x arm would characterize a curve nobody is going to buy on;
  the wave spent that GPU time on the stacked and deep children the directive
  asked for instead.
* **`COMP.MA.MXQ5` has no benchmark row.** Its corrective routes are forward
  hooks, not `state_dict` entries, so `mxq_materialize_child.py` cannot write an
  HF child that carries them — the same gap `NOTES-MX-P` §6a stated for
  `COMP.MA.MXP1`, and wave 13 did not close it either. **The fix is designed and
  unbuilt**: construct the HF child from the folded tensors in-process, install
  the routes as forward hooks the way `run_lfm2_composition_frozen_review.py`
  already does, and hand the live module to `lm_eval.models.huggingface.HFLM
  (pretrained=<module>, tokenizer=…)` instead of a directory path. No
  `trust_remote_code` and no custom modeling file is needed for that route; it
  was not attempted here because the wave's remaining GPU time was spent on
  children rather than on tooling.
* **`COMP.MA.MXQ4`, `MXQ6` and `MXQ7` were not benchmarked**, per the standing
  rule that a child which loops a token to the cap has no
  intelligence-per-compression to measure — its perplexity would be a number
  about a collapse. `COMP.MA.MXQ1` was benchmarked because it is the mechanism's
  own solo rung and 6/15 coherent.
* **Per-lane readings for the wave-10 and wave-11 children.** Still `unstated`,
  as `NOTES-MX-P` §10.3 asked. Wave 13 did not re-read earlier children while
  moving the frontier, for the same reason `NOTES-MX-O` §7a refused to.

## 7. The stacks, built because the directive said to build them

### 7.1 M5 on the frontier — `COMP.MA.MXQ4`

`COMP.MA.MXP2`'s exact fold set plus head merges at the two cheapest sites, L02
and L12. **Net 0.682581x — deeper than any previously coherent child — and
0/15.** Five lanes run to the 1024-token cap on a single repeated phrase
(`instruction-three-actions` emits "1. Plan a meal plan:" ninety-odd times), nine
of the fifteen graded lanes leave nothing when their repetitions are removed,
and the smoke read is `incoherent`, so it is **dead under both readings**. Its
own install trace says where the budget went: the L02 merge, installed first, is
**0.399369** of the child's 0.679394 terminal drift by itself, and the L12 merge
adds **+0.111632** on top of six SwiGLU lanes.

### 7.2 Repair on the shopped child — `COMP.MA.MXQ5`

Wave 12's central bet, finally built: three k=256 corrective routes at
L05/L06/L13 on top of `COMP.MA.MXP2`.

| | net | drift | review | per-child | per-lane | lanes charged |
|---|---:|---:|---:|---|---|---:|
| `COMP.MA.MXP2` | 0.703409x | 0.489603 | 6/15 | dead | impaired | — |
| **`COMP.MA.MXQ5`** | **0.710270x** | **0.487874** | **7/15** | dead | impaired (lane-scoped) | 2 of 16 |

**Repair pays on the shopped child too — +1 review case for 1,575,939 added
parameters, 0.69% of the model, charged in full.** It is also the healthiest
text in the wave: one lane at the cap, and the only child here with two `Useful`
gradings (`context-retrieval`, `summarization-priorities`, both near-complete
against the source).

The route trace repeats wave 12's locality result with one new fact:
`0.489603 → 0.474727` (L05, **−0.014876**) `→ 0.472629` (L06, **−0.002098**)
`→ 0.487874` (L13, **+0.015245 — worse**). Wave 12 saw a route make drift worse
only at a block that was *never folded*. Here **L13 is folded** and its route
still hurt. Repair's purchase is not "wherever a fold is"; it is where the
fold's own error is still linear in that block's input, and at the end of a
stack it is not.

### 7.3 The depth the allowance arithmetic wrote off — `COMP.MA.MXQ6`

Eleven SwiGLU lanes plus the token interface, **with the order measured instead
of taken by depth**: every lane after MXP2's three was priced on MXP2's own base
by the ladder and ranked by drift per million removed.

**Net 0.497777x — the same depth as `COMP.MA.MXO2`'s 0.497739x — drift 0.821803
against MXO2's 0.814086, and 0/15.** Ten of sixteen lanes at the 1024-token cap;
one emits **zero tokens**; fourteen of the fifteen graded lanes leave nothing
when their repetitions are removed. **Shopping the order did not help at this
depth**, which is consistent with the ladder's own finding that at depth the
lane price stops discriminating.

### 7.4 The ladder — the compounding rate, measured

`wave13/probes/deep-ladder.json`, 3,982 s, seven rounds, ranked by drift per
million parameters removed, winner installed permanently each round:

| lane bought | base lanes | base drift | cheapest marginal | drift / M-param | winner |
|---:|---:|---:|---:|---:|---|
| 3rd (wave 12) | 2 | 0.421325 | +0.034129 | **0.005783** | `L13.T5` |
| 4th | 3 | 0.455455 | +0.050518 | **0.008555** | `L02.T8` |
| 5th | 4 | 0.505972 | +0.043186 | 0.007314 | `L08.T8` |
| 6th | 5 | 0.549159 | +0.038762 | 0.006565 | `L11.T8` |
| 7th | 6 | 0.587921 | +0.030083 | **0.005095** | `L09.T8` |
| 8th | 7 | 0.618004 | +0.046781 | 0.007923 | `L12.T8` |
| 9th | 8 | 0.664785 | +0.041657 | 0.007055 | `L07.T8` |
| 10th | 9 | 0.706443 | +0.037158 | 0.006293 | — |

**This corrects wave 12.** `NOTES-MX-P` §5 said the per-lane marginal "will rise,
not fall, as the stack fills", inferred from `COMP.MA.MXO2`'s trace. Measured, it
rises 48% from lane 3 to lane 4, then **falls** through lane 7 — the cheapest
per-parameter lane the campaign has ever measured is the **seventh**, on a
six-lane base — then oscillates between 0.0051 and 0.0079. Drift accumulates
roughly **linearly at ~0.042 per lane**. The reason 0.5x is hard is not that
lanes get expensive; it is that the coherence budget is small and drift walks
steadily through it.

## 8. Estate

* **Ledger.** 447 rows sha `a1c6b924238cfecf5f48a49baafcb8e77d2dc32ff071f89526c5e2e57845355a`
  before; backed up **twice** (`backup/ledger.jsonl.pre-wave13-20260812T210030`
  and `wave13/ledger.jsonl.pre-merge-backup`), both verified byte-identical by
  sha256 **and** by `cmp` before the merge; merged **exactly once** behind a
  `.MERGED_ONCE` marker; **461 rows** after, sha
  `4ad102d32efbdf59008427db3b1ccd18b54f4cb1bc753f0d286e036bbbb56d57`. The merger
  verified the prior bytes are a prefix and the check was repeated independently
  afterwards (705,287 bytes before, 729,335 after, **24,048 appended**, prefix
  holds). The first merge attempt failed *before writing anything* — the wrapper
  ran the merger under the system `python3`, which has no `numpy` — and the
  ledger's sha was re-checked as unchanged before the retry, which is why the
  `.MERGED_ONCE` marker is the right guard and a `set -e` would have been the
  wrong one.
* **Seals.** **456 of 456** resolvable rows verify, **0 fail**. Five rows do not
  resolve: the same five as waves 11–12 (three `CAL.C01.T5.X4`, one
  `C06.T2.S4.L12` a6review instance, one `COMP.MC.MXO0` infra row).
* **The seal verifier's search roots, again.** The wave12→wave13 script
  derivation dropped `wave12/cells` from `SEARCH` — the identical defect
  `NOTES-MX-O` §8 and `NOTES-MX-P` §7 each caught in their own wave. It was
  caught here by reading the roots before running, not by the count. **Every
  earlier root is kept**, and this is the third wave in a row that has had to
  say so.
* **Wave-13 rows.** 14: seven composed children and seven ADDENDUM 6 review
  instances, all `sm86 NVIDIA GeForce RTX 3090`, all receipt schema v2, all
  under one `src_tree_sha` (deployment `neural-forge-matrix-63e95cd8`).
* **Retention.** Nothing deleted. `wave13/pruned-tensors-manifest.json` records
  the digest and size of all nine retained tensor sets (three 12.6 MB solo sets,
  `MXQ4` 388.0 MB, `MXQ5` 362.8 MB + 37.8 MB of routes, `MXQ6`/`MXQ7` 614.5 MB
  — byte-identical to each other, as they must be, since `MXQ7` is `MXQ6`'s fold
  set plus hooks — and `MXQ7`'s 37.8 MB of routes). The materialized HF child was
  deleted after its benchmark, as every wave does.
* **Disk.** Server `/` at 4.6 GB free at wave start and **3.5 GB at wave end**;
  the wave's own footprint on `/` is 764 KB of workspace plus a 90 MB
  deployment, and every heavy artifact is under
  `/mnt/ai_data/neural-forge-matrix/wave13/` (**2.7 GB**). The corrective-route
  conditioned captures ran in `/dev/shm` (8 GB each, 126 GB tmpfs), were deleted
  as their rows were read, and the tmpfs was **cleared at wave end (4.0K used)**.
* **Deployments.** One: `neural-forge-matrix-63e95cd8`. The review surface was
  verified on extraction before the first child was generated.
* **Human-review package.** `wave13/human-review/` — parent plus `MXQ5`, `MXQ1`,
  `MXQ2`, `MXQ4`, `MXQ6`, full verbatim responses, machine verdict as an
  annotation.

## 0. The sentences

1. **M5 is expressible at 4x, exact where exactness is checkable, and 8–13×
   too expensive per parameter to matter.** 0.239583x per attention block
   (correcting §8's 0.21875x), identity gate ≤ 8.7e-15, and 0.046–0.077 drift
   per million parameters removed against a SwiGLU lane's 0.0051–0.0086.
2. **The reason is a fact about the model, not about the kernel: LFM2's
   attention heads are not rotations of a few shared bases.** Every site's kv
   partition is 7 heads plus one outlier, and the query partitions are one large
   cluster plus singletons. That is the diagnostic to run first on a bigger
   model, and it costs seconds.
3. **The stacks were built anyway, and they are what settled it.**
   `COMP.MA.MXQ4` reaches **0.682581x — deeper than any coherent child the
   campaign had — and reads 0/15**; one merged block on its own multiplies
   lambada perplexity by **14× while removing 1% of the model**.
4. **Repair pays on the shopped child.** `COMP.MA.MXQ5`: 6/15 → **7/15** for
   1,575,939 added parameters, 0.69% of the model, charged in full — and the
   third route, at a block that **is** folded, still made drift worse.
5. **The saturation claim wave 12 reasoned from is wrong.** The per-lane price
   does not run away: it rises 48% at lane 4 and then falls to its cheapest
   measured value at lane 7. Drift accumulates linearly at ~0.042 per lane, and
   0.5x — built and read at `COMP.MA.MXQ6`, shopped order, **0/15** — fails
   because the budget is small, not because the lanes get dear.
