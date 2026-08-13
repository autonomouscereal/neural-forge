# NOTES-MX-X — wave 20: the basis objective is a variable, and it is the wrong one to move

Wave 20, branch `codex/lrc-blank-compression`, source commits `1c0c7e3a` (the
wave-20 builder: the two new basis objectives, the case-implicated row
selectors, the writers' internal lambda split, the free retention mode),
`1d17b482` (the two never-built slate classes, CBT and F-LANDMARK, and the
subspace comparison), `c317179b` (the seal-root repair the blanket rename
caused and the packaging fixes), plus the sealing commit for this note. One
deployment built, sealed and reviewed every child and both probes:
`neural-forge-matrix-1c0c7e3a`. Protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` and prompt
pack `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` verified
digest-true **in the deployment** before anything was generated, which wave 19's
subset archive did not carry.

**Three calibrations passed before anything was graded.** The fresh conditioned
capture reproduces the sealed collector's rows to wave 15's and wave 19's
numbers exactly (L00 relL2 3.367e-3 cos 0.999994, L05 1.647e-2 / 0.999721, L13
2.034e-2 / 0.999784). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23`** on every wave-20 dossier and on both probe
dossiers. And the parent's benchmark row reproduces the sealed reference in all
seven columns (0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl 24.39,
wikitext 81.80).

*§0's sentences are at the bottom.*

---

## 1. The hypothesis, and the free measurement that mostly killed it

Wave 19 left two facts side by side. Behaviour on the `hidden_size` axis tracks
**retained residual read energy** with a floor near 0.92, and per-site install
quality predicts nothing. And the per-case attribution artifact says the tenth of
the review matrix that is not one difficulty ladder is per-axis: the width fold
keeps summarisation and takes **context-retrieval and planning-dependencies to
zero**.

Read together they suggest a mechanism: the basis is chosen to retain the most
energy *pooled over the corpus*, so it discards directions the killed cases
depend on **because nothing in its objective ever asked about them**. If that is
true the objective is a live degree of freedom and wave 19 only ever moved the
width.

Testing it needs a map from cases to rows of the evidence, and there are exactly
two places that map could come from. One is the review prompts — which would
make every score in the wave a measurement of its own training set, and is
refused. The other is a rule on the token stream of the sealed evidence. The
corpus is twenty-four agentic support dialogues whose system messages are
verification protocols and whose turns are full of identifiers to be recalled, so
both rules are structural and neither contains a word list:

* **retrieval**: position `t` where the bigram `(x_t, x_{t+1})` has already
  occurred earlier in the same sequence — the induction signature, and the
  mechanical content of "go and get it from the context". **998 of 6,335 rows
  (15.75%)**.
* **procedural**: any position within 32 tokens of a list-item opener, matched on
  a two-token decoded window by `\n[ \t]*(?:\*\*)?(?:\d{1,2}[.)]|[-*•])`. **311
  rows (4.91%)**.

Weighted by the width axis's own per-case deficit against the campaign hold rate
(`retrieval` 0.3175, `procedural` = planning-dependencies 0.2698 +
instruction-three-actions 0.0992 = 0.3690, normalised to 0.4625 / 0.5375), with
**60.9% of the total deficit left unassigned and reported** because no
closed-form row rule exists for the rest.

**Then the measurement, which costs one capture pass and no build.** Under wave
19's own sealed `E-reader-pca-768`, on held-out rows:

| rows | retained read energy |
|---|---:|
| pooled stream | **0.9193** |
| retrieval rows | **0.9265** |
| procedural rows | **0.9120** |
| case-weighted | **0.9187** |

The case rows are retained at the pooled rate. **Retrieval rows are retained
BETTER than average**, at every width (0.8254 against a pooled 0.8124 even at
512). The plain basis is not discarding retrieval directions, so the mechanism
the wave was built on is false as stated, and it took no GPU to say so.

The reweighting can barely move the number it exists to move. Across a beta grid
at 768, the best held-out case retention is at beta 0.25 — **0.9204 against
0.9187**, a gain of 0.0017 — bought with a pooled loss of 0.0015. At beta 1.0
the case retention is *worse* than at beta 0 (0.9106), because the mixture
overfits the selector rows: it is monotone on the fit split, which the self-test
proves, and not on the held-out rows, which is the number that counts.

Per the anti-proxy rule the ladder ran anyway, and §2 is why that rule exists —
this time in the opposite direction from wave 19.

## 2. The controlled experiment, and what it says about the law

Five children at 768 are **165,978,624 parameters each, 0.722610x, the same
corpus, the same solver, the same code**. Only the basis or the solve moves.

| child | what moved | retained read (hld) | subspace exchanged vs plain | review |
|---|---|---:|---:|---:|
| `ARCH.W768` (wave 19) | — the baseline | 0.9193 | 0 | **3/15** |
| `ARCH.W768LS` | writers' λ on an internal fit split | 0.9193 | 0 | **3/15** |
| `ARCH.W768ST` | `--arm stale`, nothing re-solved | 0.9193 | 0 | **2/15** |
| `ARCH.W768AW25` | attribution-weighted, beta 0.25 | 0.9178 | **17.2 of 768** | **0/15** |
| `ARCH.W768EV` | wave 19's sealed evidence-PCA basis | 0.9122 | 51.4 | **1/15** |
| `ARCH.W768AW100` | attribution-weighted, beta 1.0 | 0.8967 | 96.7 | **0/15** |
| `ARCH.W768RG` | reader-generalised | 0.4780 | 230.5 | **0/15** |

**This is the wave's finding and it refines wave 19's law rather than
confirming it.** `W768AW25` has the same parameter count, the same drift band
(0.379 against 0.375), the same per-site quality (median write cos 0.9998), and
retained read energy that differs from the living child's by **0.0015**. It
exchanges **seventeen of 768 directions**. It scores **zero**.

So retained energy is a **ceiling, not a predictor**. Every child under ~0.90 is
dead — 704 at 0.8953, `W768AW100` at 0.8967, `W768RG` at 0.4780, all 0/15 — and
that half of wave 19's law survives intact and is now bounded more tightly. But
*above* the floor the score is not a function of the retained fraction: two
subspaces with the same energy and a 2.2% difference in span are 3/15 and 0/15.
Which directions you keep matters, and the pooled reader moment's own top
eigenvectors are, so far, the only rule that finds the ones that matter.

**The census would have picked the wrong child again, and in the other
direction.** By retained case energy, `W768AW25` is the best basis in the wave.
By review it is the worst of the four that are not obviously broken.

## 3. Which energy the law is about, settled by the leftover nobody built

`E-evidence-pca-768` was sealed by wave 19 and never built against. It is the
sharpest available discriminator, because it moves the two candidate currencies
in **opposite** directions: it retains **more raw** energy than reader-PCA
(0.9636 against 0.9522) and **less read** energy (0.9122 against 0.9193).

Raw says it should be the better child. Read says it should be the worse one.
It scores **1/15** against reader-PCA's 3/15. **The currency is the energy a
reader consumes, not the energy the stream carries**, and that is now a
measurement rather than an argument from RMSNorm.

The reader-generalised basis is the same question asked at the other extreme and
it answers it twice over. Maximising reader energy *per unit of raw energy* — the
top generalised eigenvectors of `M_read v = mu M_raw v`, orthonormalised over
their span — selects the directions the gain amplifies most relative to what the
stream actually carries, which are the directions the stream **barely carries at
all**: 0.4780 of the read energy and **0.1862** of the raw. Its drift is 3.749
against a living child's 0.375, and all sixteen lanes return multilingual token
salad. Wave 19's observation that "reader-weighted energy is lower than raw at
every k" does not mean the readers are looking somewhere better. It means the
gain is largest where the signal is smallest.

## 4. The two leftovers that paid, and the rung that did not

**The writers' λ split is worth a capability, not a score.** `W768LS` scores the
same 3/15 as the child it controls for, on a **different set of cases**. Plain
`W768` holds summarisation, uncertainty and multi-turn; `W768LS` holds
summarisation, multi-turn and **context-retrieval** — the case the attribution
artifact names as the width axis's signature zero, and the only child in the wave
that recovers it:

> The question is to bring a dog and borrow a shovel. The date is Saturday, and
> the last checkout time is 4 p m.

against wave 19's plain child on the same prompt:

> - The 7:30 a.m. is the 4:15 pm. and the 9:00 pm. is the 4:00 pm.

Wave 18 paid four review cases for this fix on the value path; on the width
writers it pays zero cases and one capability, and lowers lambada perplexity
from 8176.90 to 7722.72. **When reporting a fold, report which cases it keeps.**

**The stale arm prices the conditioned solve at one review case.** `W768ST`
installs the closed form and re-solves nothing: **2/15** against 3/15. The
conditioned re-solve is worth one case on this axis — real, positive, and much
smaller than the four cases wave 18's λ fix was worth on the value path. It is
also the wave's cleanest reminder that the closed-form projection is already
most of the child: `E^T W` is the L2-optimal write, and the solve is a
correction, not the construction.

**704 is dead and the cliff is now 64 wide.** `ARCH.W704`, 0.656116x, retained
read energy **0.8953**, **0/15**. The floor is bracketed between 0.8953 (dead)
and 0.9193 (3/15) — half of wave 19's interval. The attribution-weighted basis at
704 is deader still: fifteen of sixteen lanes return the empty string.

## 5. The two never-built slate classes

Both are **probes, research-only**, and both receipts say why in their own words
before any number is quoted.

**CBT (slate §A6) — block-Toeplitz tied ridge at `attn_o`.** The first family
offered to the rank-truncation graves that does not truncate: constrain the
1024×1024 to an 8×8 grid of 128×128 blocks tied by `class(I,J) = I−J`, identity
permutations, 15 base classes. The loss separates over the output index within a
block, so it is 128 independent (15·128)-dimensional systems that share **one**
Gram, and both the Gram and the right-hand side assemble from the sealed input
Gram alone — no target matrix is ever materialised. Storage 245,824 numbers
against 1,048,576 = **4.266x**, every map counted.

* **At L12**: held-out cos **0.9032**, full rank **1024 of 1024**, update_scale
  0.533, against a tied projection with no evidence at all of 0.6937.
* **At L02, where the sealed twins are**: held-out cos **0.9108**, full rank.
  The campaign's own 4x cell at that exact site, `C09.T1.S4.L02`, is a **rank-128
  contraction** storing 262,144 numbers (4.0x) at held-out cos **0.9434**, and
  the other four techniques there run 0.9349–0.9605.

**CBT's central bet loses its own head-to-head.** It stores *fewer* numbers
(4.27x against 4.00x), it is *full rank*, and it reconstructs the parent's map
**worse** than every rank truncation at the same site. Channel space having no
translation structure was the pre-registered risk and it is what the measurement
found.

The frozen review at L12 reads near the parent — 5 useful and 7–8 partial by my
reading against the parent's 5/7 — and **that is the review being insensitive,
not CBT being free**. One site in a fourteen-block model is below this
instrument's resolution. The benchmark is not: one tied tensor moves lambada
from **0.4600 to 0.1800** and its perplexity from **24.39 to 134.66**, and
wikitext word-perplexity from 81.80 to 99.88.

**F-LANDMARK (slate §A8) — verbatim-state landmark memory at L12.** Six of eight
KV groups converted, m=256 actual cached key/value vectors installed as
parameters, RoPE dropped on the converted queries, two worst-reconstructed groups
kept verbatim, `out_proj` refit in place. 4.0x exactly per converted group.

* **One deviation from the spec, stated not smuggled.** Pivoted QR on an `N×64`
  key matrix is exhausted after 64 picks, so the remaining 192 landmarks are
  taken by descending attended mass. Both counts are in the receipt.
* **beta selected on an internal fit split landed at 0.125 for all six groups**,
  which is exactly `head_dim^-0.5` — the parent's own attention scaling,
  recovered by a grid that was never told it. The first run's grid started at
  0.25 and every group pinned to the boundary; the grid was widened to 1/32
  until the choice was interior, and the receipt records which happened.
* `out_proj` held-out cos **0.8585 → 0.9045** under the refit.
* The frozen review reads near-parent, including **correct context retrieval**
  ("The day allowed is **Saturday**. The latest checkout time is **4:15 p.m.**")
  — from a block whose converted heads cannot see the context at all, because two
  KV groups and thirteen other blocks still can. It cannot be benchmarked:
  custom runtime, no `from_pretrained`, no ADDENDUM 8 column, and the honest
  consequence is that F-LANDMARK has **no distribution measurement at all**.

Both probes touch one site. Neither is a compression claim: F-LANDMARK removes
589,824 parameters at one layer (0.26% of the model) and CBT removes none that
a loader can count.

## 6. Benchmarks

Every non-looping child and the one benchmarkable probe. ADDENDUM 8 invocation
unchanged (limit 200, seed 1234, batch 4, max_length 2048, float32, sm86):

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `PROBE.CBT-L12` | 1.000000† | ~12/15 | 0.4550 | 0.4750 | 0.6700 | 0.5850 | 0.1800 | 134.66 | 99.88 |
| `W768` (wave 19) | 0.722610 | **3/15** | 0.3100 | 0.4300 | 0.5200 | 0.4850 | 0.0300 | 8176.90 | 1279.46 |
| `W768LS` | 0.722610 | **3/15** | 0.3050 | 0.4300 | 0.5050 | 0.4600 | 0.0300 | 7722.72 | 1312.49 |
| `W768ST` | 0.722610 | 2/15 | 0.2900 | 0.4350 | 0.5150 | 0.4550 | 0.0350 | 7449.50 | 1297.15 |
| `W768AW25` | 0.722610 | **0/15** | 0.3200 | 0.4350 | 0.5350 | 0.5200 | 0.0200 | 10442.31 | 1419.67 |

† loader-counted; CBT's compression is 4.266x of *storage* at one site.

**The review and the distribution disagree again, and this time on the wave's
central comparison.** `W768AW25` is the **best** of the four width children on
arc_easy, hellaswag, piqa and winogrande — and it is the one that holds **no**
review cases, where the child it is matched against holds three. It is worse only
on lambada and on both perplexities. Under behavioural gating the answer is
unambiguous: the attribution-weighted objective is not a win. But a wave that
reported only the accuracy columns would have concluded the opposite, and that is
worth having in writing.

## 7. Estate

* **Commits.** `1c0c7e3a` (builder, objectives, selectors, λ split, free mode),
  `1d17b482` (CBT, F-LANDMARK, probe review, subspace tool), `c317179b` (seal
  root and packaging repairs), plus the sealing commit for this note. Nothing
  pushed, nothing rebased, staged by explicit path only; the Codex lane's
  uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.
* **Self-test.** `mx20_selftest.py` **17/17**, no model loaded — wave 19's nine
  plus seven of this wave's own, including that the attribution-weighted
  objective at `beta = 0` is bit-identical to the plain reader-PCA basis (so
  every comparison against wave 19 is against the same construction), that
  raising beta never lowers retained case energy on the fit split, that the
  reader-generalised basis is Euclidean-orthonormal, and that the writers'
  internal split is every third fit batch by position.
* **Ledger.** **620 → 636**, sha
  `aa703706990e9d7e06802c4a7987adb6abf425531c941e7d18a50c2838505fc6`. Backed up
  first (`ledger.jsonl.pre-wave20-20260813T093229`, verified by sha **and**
  `cmp`), merged exactly once behind `wave20/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards and again in the repository copy. Sixteen rows:
  8 ARCH run rows and 8 ADDENDUM 6 verdict rows. No superseded rows: the review
  queue now **refuses** a tag it has already sealed, which is wave 19's
  double-seal fixed at the driver instead of at the merge.
* **Seals.** `mx20_verify_seals.py --deployment <deployment>`: **16 roots
  present, 0 undeclared; 636 rows, 632 resolve, 632 verify, 0 fail, 4 do not
  resolve** — the same four pre-existing non-resolvers waves 14–19 named.
* **The blanket rename nearly deleted last wave's seals.** Cutting the wave-20
  driver tree from wave 19's by rewriting every `wave19` to `wave20` rewrote the
  verifier's own root list, dropping `wave19/cells`. The audit function caught
  it — that root exists on disk and was no longer declared. This is the third
  wave that file has warned about this and the first that recorded doing it.
* **Disk.** Server `/` began and ended at 13 GB free. Deployments end at
  current+1 (`1c0c7e3a` and `e7d068da`), with `b372d47c` pruned and named in
  `wave20/deployments-prune-manifest-wave20.json`. Eight children were built;
  **five are kept** (`W768LS`, `W768ST`, `W768AW25`, `W768EV`, `W704`) and three
  are deleted and named with their sealed tensor digests, freeing 0.91 GiB.
  `W768AW25` is kept at 0/15 deliberately: it is the wave's central negative
  result and the one child whose benchmarks disagree with its review in the
  awkward direction.
* **Human-review package, in the repository.**
  `reports/matrix-campaign-20260810/wave20/human-review/` — 68 files: every
  dossier, every child and parent response rendered verbatim, every run receipt,
  every verdict receipt, every per-case reading, every build receipt, the five
  new basis censuses, the free retention report, the subspace comparison, both
  probe receipts and dossiers, and the benchmark table.
* **And wave 19's package is repaired.** Its fourteen `READ-*.txt` renderings
  were committed **empty**: `package.sh` invoked the dossier reader through the
  queue directory, which holds only shell scripts, and `2>/dev/null` swallowed
  the failure. The dossiers were intact, so the renderings are regenerated and
  committed, and wave 20's packager goes through the deployment.

## 8. What this wave did not spend, and what it leaves standing

* **The λ split was not composed with anything.** It is worth a capability at
  768 and costs nothing; whether it moves 896 (4/15) or rescues 704 (0/15) is two
  builds nobody has run.
* **No basis objective was tried between reader-PCA and the ones that failed.**
  The wave moved the objective toward the cases and away from the readers, and
  both directions lost. Whether anything beats the pooled reader moment's own
  eigenvectors is open, and the honest reading of §2 is that the search space is
  narrow: 17 exchanged directions out of 768 was already fatal.
* **The 0.90–0.92 band is unmeasured.** 704 sits at 0.8953 and 768 at 0.9193.
  One rung at 736 (0.9070-ish) would close it, and it is one build and one
  review.
* **CBT's merge schedule was never exercised.** At 4x the 15 base Toeplitz
  classes already clear the budget, so the greedy evidence-weighted merge — the
  only adaptive part of the class — has not been run. The 8x probe would run it,
  and after §5's head-to-head there is no reason to fund one.
* **F-LANDMARK has no distribution measurement** and cannot get one without a
  loader that accepts a custom module. That is a real hole in a probe that read
  near-parent, and it is the reason the class is not being funded off a frozen
  review alone.

## 0. The sentences

1. **The basis objective is a real degree of freedom and moving it is a
   mistake.** At an identical parameter count, an identical drift band, identical
   per-site write cosines and retained read energy that differs by **0.0015**,
   exchanging **17 of 768 directions** takes the width child from **3/15 to
   0/15**.
2. **Retained read energy is a ceiling, not a predictor.** Below ~0.90 every
   child is dead (0.8953, 0.8967, 0.4780 — all 0/15); above it, two subspaces
   with the same energy score 3/15 and 0/15.
3. **The law's currency is the energy a READER consumes, not the energy the
   stream carries.** The evidence-PCA basis retains more raw energy (0.9636 vs
   0.9522) and less read energy (0.9122 vs 0.9193) than reader-PCA, and scores
   1/15 against 3/15.
4. **The hypothesis this wave was built on is false, and one capture pass said
   so.** The case-implicated rows are retained at the pooled rate under the plain
   basis — retrieval rows at **0.9265** against a pooled **0.9193**, i.e.
   *better* — so the width fold is not killing those cases by discarding their
   directions.
5. **Reader-per-raw energy is the worst objective in the wave.** The generalised
   eigenvectors of the reader moment against the raw moment retain **0.4780** of
   the read energy and **0.1862** of the raw, drift 3.749, and return
   multilingual token salad in all sixteen lanes. The gain is largest where the
   signal is smallest.
6. **Choosing the width writers' lambda on an internal fit split buys a
   capability, not a score.** Same 3/15, different cases: it recovers
   **context-retrieval**, the case the attribution artifact names as this axis's
   signature zero, and lowers lambada perplexity from 8176.90 to 7722.72.
7. **The conditioned solve is worth exactly one review case on this axis.**
   `--arm stale` reads 2/15 against conditioned-local's 3/15 at the same
   parameter count.
8. **The cliff is now 64 dimensions wide.** 704 is 0/15 at 0.8953 retained read
   energy; 768 is 3/15 at 0.9193.
9. **CBT loses its own head-to-head against the graves it was built for.** At
   `attn_o`/L02 it stores fewer numbers than the sealed rank-128 twin (4.27x vs
   4.00x), is **full rank**, and fits **worse** (held-out cos 0.9108 vs 0.9434);
   every rank truncation at that site runs 0.9349–0.9605.
10. **A frozen review at one site is not a sensitive instrument, and the
    benchmark says so.** CBT reads near-parent on all sixteen lanes while its
    single tied tensor moves lambada from 0.4600 to 0.1800 and its perplexity
    from 24.39 to 134.66.
11. **F-LANDMARK's beta grid recovered the parent's own attention temperature.**
    Selected on an internal fit split, all six converted groups chose
    **0.125 = head_dim^-0.5**, which is `Lfm2Attention.scaling` exactly.
12. **The review and the distribution disagree on the wave's central
    comparison.** `W768AW25` is the best of the four width children on
    arc_easy, hellaswag, piqa and winogrande, and holds **zero** review cases
    where its matched control holds three.
13. **The frontier is unchanged and was recomputed, not quoted.** Sorted by
    stored-parameter ratio over the sealed rows, `ARCH.CL896` (wave 15,
    0.688431x, 4/15) is still the deepest child with a non-zero score; wave 20's
    `ARCH.W768LS` ties wave 19's `ARCH.W768` at 0.722610x and 3/15 for second.
14. **Wave 19's committed human-review renderings were empty files** — the
    packager called the reader through a directory that holds only shell scripts
    and `2>/dev/null` hid it. The dossiers were intact; the renderings are
    regenerated and committed here.
