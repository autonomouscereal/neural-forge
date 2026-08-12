# NOTES-MX-S — wave 15: the evidence was never the problem, and one config field costs 0.99 of the score

Wave 15, branch `codex/lrc-blank-compression`, source commits `08999e84`
(ADDENDUM 12 + the wave-15 builder) and `1c197ed4` (three fixes found by running
it), on top of wave-14 HEAD `f86ebf0f`. Deployment
`neural-forge-matrix-1c197ed4`, cut with `git archive -c core.autocrlf=false`
from this wave's own commit; review surface verified on extraction and on every
review before a token was generated: protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, prompt pack
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`.

Wave 14 left one question and one debt. The question: fourteen SwiGLU folds, each
inside the sealed alive fidelity band, compose to 0/15 — is that because the
evidence each fold was solved against stops being true the moment the fold before
it is installed? The debt: a child whose *shape* changed could not carry a sealed
machine verdict at all.

The debt is paid. The question is answered, and the answer is **no**.

*§0's sentences are at the bottom.*

---

## 1. The debt: ADDENDUM 12, and three wave-14 children that can now be read

`NOTES-MX-R` §6 recorded that a narrowed child replaces no parent tensor
shape-for-shape, carries no `weight_keys`, and is therefore not a `CellId` under
receipt schema v1 or v2. Receipt schema **v3** closes it in the shape ADDENDUM 4
used for the same problem one level down: one new id form (`ARCH.<TAG>[.<RUNG>]`),
one new `cell_kind`, one new mandatory `architecture` block (config diff, export
statement, parameter arithmetic, per-axis provenance, `passed_through`), and one
new ledger column (`standard_shape`) whose absence means "sealed before this
addendum" for all 461 existing rows.

Three properties are load-bearing and each is a test:

* **`CellId.parse` refuses `ARCH.*` under v1 and v2.** A sealed row cannot acquire
  an identity that did not exist when it was sealed — the same rule ADDENDUM 7
  obeys, in the other direction.
* **The ratio must be its own arithmetic *and* the ledger's.**
  `architecture.storage.net_ratio` is checked against `child/parent`, against the
  count the **loader itself** made, and against
  `solve.stored_parameter_count`/`parent_parameter_count`, which is what the
  ledger's `stored_parameter_ratio` column is computed from. On an ARCH row that
  column is the size of the model that runs. That is the single defect
  `NOTES-MX-R` §1 measured, fixed in the place it was measured.
* **`seal_matrix_behavior_review.py` is untouched.** ADDENDUM 6's verdict and
  ADDENDUM 9's per-lane reading seal on a shape-changing child exactly as they do
  on every other kind. The frozen review runner gained `--child-model` and
  nothing else, so a narrowed child and a parent-shaped one are graded on one
  surface.

20 new tests, `tests/test_matrix_addendum12.py` plus the re-pinned ADDENDUM 7
version test; the whole `-k matrix` suite runs 537 passed against the deployment's
own 517, with the **same 11 pre-existing failures** in both (they are
test-ordering artifacts and reproduce on the unmodified wave-13 deployment).

## 2. The question: conditioning, measured three ways at one width

The harness re-captures activations **through the partially built child** by
hooks, on the same 24 batches, each rebuilt as one 264-token teacher-forced
sequence from the trace cache's own `input.input_ids` + `input.forced_decode_ids`.
That reproduces the sealed row order exactly, and the claim is checked rather than
asserted before any child is built:

```
parity, fresh capture vs the sealed collector's post_swiglu rows
  L00  fit 4751 rows  relL2 3.37e-3  cos 0.9999944
  L05  fit 4751 rows  relL2 1.65e-2  cos 0.9997207
  L13  fit 4751 rows  relL2 2.03e-2  cos 0.9997842
```

(the residual is bf16 storage plus cached-decode versus full-sequence numerics.)

The stale arm then reproduces wave 14 **textually**, not merely numerically: same
λ ladder firing at the same three sites (L00 free 1.555 → λ=1 → 0.456; L01 1.110 →
λ=0.1; L02 1.027 → λ=0.01), same per-site cosines to three decimals, and the same
sentences on the frozen surface (`Mira finished the final product after the final
product.`; `The answer is a dependency-aware plan that covers backups, DNS, TLS,
verification, and rollback. Keep it to six steps.`). The control is a control.

| child | arm | net | per-site cos (median / min) | terminal relL2 | cap | verdict |
|---|---|---:|---:|---:|---:|---|
| `MX15-S640` | stale (wave 14's objective) | **0.640497x** | 0.8563 / 0.8071 | 0.8429 | 6/16 | **dead 0/15** |
| `MX15-CL640` | conditioned, local teacher | **0.640497x** | 0.8527 / 0.8069 | 0.8178 | 9/16 | **dead 0/15** |
| `MX15-CT640` | conditioned, parent trajectory | **0.640497x** | 0.5968 / 0.5221 | 0.9126 | 16/16 | **dead 0/15** |

**Conditioning is not the missing variable.** It does exactly what it was supposed
to do at the level it operates on — the conditioned arm's late-layer solves are
*better* than the stale arm's (L12 0.9263 vs 0.8971, L13 0.9588 vs 0.9400,
terminal drift 0.8178 vs 0.8429) — and the child is the same kind of dead. It
restates its prompt fluently and loops.

The number that explains why is one this campaign had never measured, and it comes
free with a conditioned build:

```
residual-stream drift at each FFN's own input, vs the parent, MX15-CL640
  L00 0.000  L01 0.573  L02 0.667  L03 0.683  L04 0.643  L05 0.654  L06 0.603
  L07 0.596  L08 0.570  L09 0.544  L10 0.532  L11 0.536  L12 0.542  L13 0.569
```

**Narrowing one SwiGLU moves the next layer's input by 0.57 relative.** Not
fourteen of them — one. The drift then plateaus rather than compounding, which is
the residual stream's own normalization doing its job. So the stale-evidence story
was never the mechanism: after site 0, *every* remaining solve in wave 14 was
computed on a distribution 0.57 away from the child's, and fixing that exactly
(the conditioned arm sees the true distribution at every site) changes the outcome
by nothing at all.

What is left is the simplest possible reading and this wave's central sentence:
**a 4x-narrowed SwiGLU cannot carry its block's function, at any solve, on any
evidence.** 0.85 is what the best available 640-channel coreset achieves against
the *correct* target on the *correct* distribution, and fourteen of those is a
model that cannot answer a question.

### 2.1 The trajectory arm is worse than useless, and that is informative

Asking the writer to put the residual stream back where the parent left it —
target = the parent block's output on the **parent's** stream, features = the
child's — makes everything worse at once: per-site cosine falls to 0.52-0.61
against that target *and* to 0.50-0.59 against the local one, drift rises to 0.95
where the local arm holds 0.65, and the child emits `I'm'm'm'm'm…` to the cap on
16 of 16 lanes. The writer spends its capacity on an error it cannot express and
loses the map it could have had. A rank-1024 additive corrector on a late writer
(gap-3) is the same objective in a cheaper package, and this is the measurement
that prices it before it is built.

## 3. head_dim 16: the RoPE constraint costs 0.99 → 0.24, and it is fatal alone

Wave 14 called the score-bilinear q/k census "the best spectrum in the battery"
(top-16 directions carry 0.9926-0.9985 of the whitened score operator's energy)
and flagged one limitation: it is the `Δ=0` member of a family that a real fold
must serve at every `Δ`. Wave 15 built the fold and measured what the limitation
costs.

RoPE's rotation is **block diagonal over frequency planes**, so `q^T R_Δ k` is
*exactly* a sum of per-plane terms — which means a smaller `head_dim` can keep
whole planes and nothing else, and the child's own ladder
(`freq_j = theta'^(-2j/head_dim)`) forces the kept set to be a **uniform stride**
with `theta' = theta^(s·head_dim/64)`. Three ladders are expressible at head_dim
16. Evidence-weighted score energy retained by each, pooled over all six attention
sites and all sixteen heads:

```
stride 1 (theta' = theta^0.25)  0.2379
stride 2 (theta' = theta^0.5)   0.2018
stride 4 (theta unchanged)      0.2394   <- chosen
```

**The score energy is flat across the frequency ladder.** Keeping 8 of 32 planes
keeps 24% of it, which is what keeping 8 of 32 of anything uniform gets you. The
0.99 spectrum is real and it is not available: it lives in a 16-dimensional
subspace that mixes planes, and mixing planes is exactly what RoPE forbids.

Built anyway, with both scale corrections the change forces (softmax `scaling`
moves 64^-0.5 → 16^-0.5, folded into `q_layernorm`; the per-head RMSNorm now
averages over 16 channels, measured per layer and folded into both gains) and
`out_proj` re-solved against the **installed** context:

| child | net | per-site `attention.o` cos | terminal relL2 | verdict |
|---|---:|---:|---:|---|
| `MX15-HD16` | **0.938368x** | 0.8220 median, 0.7446 min | 0.8722 | **dead 0/15** |

It removes **6%** of the parameters and lands a *worse* terminal drift than
`ff640` at 36%. Its answers are short, generic and unrelated (`The other day, the
weather is a bit of a trend.` for the seasons case). The `−14.2M` head_dim bite is
therefore **not available** for the phase-1 arithmetic, and the reason is a
property of RoPE and not of the solve.

## 4. The arithmetic, corrected

The wave-15 brief asked for `intermediate_size 640 (−82.6M) + head_dim 16
(−14.2M) + one block excision (−12.1M) = 0.4666x`. **That sum is wrong twice**,
and the correction is worth more than the target it replaces.

* **The bites are not additive.** The −12.1M excision counts a *parent-width* FFN
  inside the deleted block. Once `intermediate_size` is 640 that block's FFN costs
  1.97M, not 7.86M, so the third bite is worth −6.2M in the presence of the first.
* **The floor is the embedding.** `tie_word_embeddings` is true and
  `embed_tokens` is 65,536 × 1024 = **67,108,864 parameters = 0.2922x of the
  parent all by itself**, untouched by every axis in the R3-standard set. The
  body is 162.58M. A body reduced to *nothing* would still be 0.292x, and 0.5x
  net requires the body to fall to 47.7M — a **3.4x** fold of the entire body.

Built and counted, not projected:

```
MX15-A552 = intermediate_size 640 + head_dim 16 + block 5 excised
            126,795,968 parameters = 0.552023x   (stock Lfm2ForCausalLM)
```

That is the correct value of the three-bite assembly, and `0.4666x` is not
reachable from it. Reaching 0.5x with these axes needs `intermediate_size` ≈ 512
and **two** excisions on top, i.e. every bite in this wave pushed past the width
where one of them already dies alone.

## 5. Excision, and the one measurement that changed a prior

`MX15-X05A` deletes block 5 (`num_hidden_layers` 14 → 13, `layer_types` minus one
entry) and re-solves **every** surviving FFN writer at **unchanged width** against
conditioned evidence — zero parameters spent on the absorption, which is the
strongest form F-EXCISE's absorption can take and still be config-standard.

```
per-site write cos after the re-solve: 1.0000 at all thirteen sites
residual drift at each FFN input:  L00-L04 0.00-0.02, then 0.71 0.65 0.64 0.56
                                    0.70 0.66 0.68 0.70
terminal relL2 vs parent 0.5707 (cos 0.8209) -- the LEAST damaged child of the wave
```

Two readings, both new:

1. **The full-width re-solve is a no-op** (`update_scale` 0.029, cosine 1.0000).
   A writer at unchanged width has no reason to move: the local map it is asked
   to reproduce is the one it already computes. Absorption of a deleted block by
   *local* targets is therefore identically zero, which retires the wave-14
   census's "L05 absorbs +0.665" as a statement about a **trajectory** objective —
   the only objective under which the writer has anything to do — and §2.1 has
   just priced that objective at total collapse.
2. **One deleted conv block costs 0.71 of stream drift in one step**, against
   0.573 for narrowing one SwiGLU and 0.00-0.02 for re-solving five writers at
   full width. The block excision is a bigger single perturbation than the fold
   the campaign has been treating as its unit.

## 6. Every child this wave sealed

All ten are `ARCH.<TAG>` cells under ADDENDUM 12, all ten load as a stock
`Lfm2ForCausalLM` with no `trust_remote_code`, all ten carry a machine verdict
and both ADDENDUM 9 readings, and every ratio below is the count the **loader
itself** made.

| cell | what moved | net | per-site cos | terminal relL2 | cap | score | per-child / per-lane |
|---|---|---:|---:|---:|---:|---:|---|
| `ARCH.S640` | ff 640, stale | 0.640497x | 0.856 | 0.843 | 6/16 | **0/15** | dead / impaired |
| `ARCH.CL640` | ff 640, conditioned-local | 0.640497x | 0.853 | 0.818 | 9/16 | **0/15** | dead / impaired |
| `ARCH.CT640` | ff 640, conditioned-trajectory | 0.640497x | 0.597 | 0.913 | 16/16 | **0/15** | dead / impaired |
| `ARCH.CL896` | ff 896, conditioned-local *(sub-4x diagnostic)* | 0.688431x | 0.891 | — | 3/16 | **4/15** | dead / impaired |
| `ARCH.HD16` | head_dim 16 | 0.938368x | 0.822 | 0.872 | 4/16 | **0/15** | dead / impaired |
| `ARCH.X05A` | block 5 excised, local re-solve *(sub-4x diagnostic)* | 0.947479x | 1.000 | 0.571 | 4/16 | **6/15** | dead / impaired |
| `ARCH.X05T` | block 5 excised, trajectory re-solve | 0.947479x | 0.888 | — | 2/16 | **0/15** | dead / impaired |
| `ARCH.A552` | ff 640 + head_dim 16 + block 5 | **0.552023x** | 0.932 / 0.846 | — | 12/16 | **0/15** | dead / **dead** |
| `ARCH.W14FF1280` | wave 14's, re-sealed *(sub-4x diagnostic)* | 0.760331x | 0.925 | — | 3/16 | **5/15** | dead / impaired |

`ARCH.W14FF640` and `ARCH.W14FF320` re-seal the other two wave-14 children on the
same terms; both were read 0/15 by wave 14 and neither is a frontier claim.

Two rows in that table are the wave's only non-zero scores and **neither is a
frontier claim**: `X05A` is a 1.07x fold (one block of fourteen) and `CL896` is
2.86x per site. The frontier-eligible children — every 4x-class rung — are 0/15.

The bisection now brackets the cliff on both sides, with the conditioned arm and
the stale arm agreeing on where it is:

```
keep 1280  per-site cos 0.925   5/15   (wave 14, stale)
keep  896  per-site cos 0.891   4/15   (wave 15, conditioned)
keep  640  per-site cos 0.856   0/15   (wave 15, stale)
keep  640  per-site cos 0.853   0/15   (wave 15, conditioned)
keep  320  per-site cos 0.781   0/15   (wave 14, stale)
```

**The cliff is between a per-site write cosine of 0.891 and 0.856**, under
simultaneous installation at all fourteen sites. Wave 14 bracketed it at
0.93/0.86 and wave 15 narrows it to 0.89/0.86 — and, more usefully, shows that
*the collapse to zero is not gradual*: 4/15 at 0.891 and 0/15 at 0.856 is a
thirty-five-thousandths-of-cosine step from "answers one question in four" to
"answers none".

## 7. What the wave did not spend

* **Head merge in succession and at S8, CBT, F-DECAY, F-LANDMARK, gap-4, XGRAM,
  Ω-RIDGE, ANTIFOLD, W-ABSORB-SB.** Not reached. All are still deck members; none
  was killed here. **Gap-3's amortized corrector is the exception**: it is the
  trajectory objective in a cheaper package, and this wave ran that objective
  twice — once at 4x width (`CT640`, total collapse) and once at **unchanged**
  width with full capacity and zero parameter cost (`X05T`, 0/15 against the same
  excision's 6/15 with local targets). It should be built only with an
  explanation of why it escapes both.
* **The selector arms.** Wave 14's own `census-vcfold_refit.json` already
  measures all four VC-FOLD selectors at keep=640 at every site: `C_pivoted` wins
  10 of 14 and `A_mass` 4, and the whole spread between best and worst selector
  is **0.005-0.013 of cosine** where the cliff needs **0.035**. The wave-15 brief
  asked for a coreset arm as the strongest untried lead; `mx14_narrow.py:100-101`
  shows wave 14's builder already *was* the coreset arm (`pivoted_columns` selects
  real channels; `w1`/`w3` keep those rows verbatim), so that lead was spent
  before it was proposed and re-spending it on a different selector cannot reach.
* **Benchmarks.** No wave-15 child was benchmarked, under the standing rule that a
  child which loops a token to the cap has no intelligence-per-compression to
  measure. Every graded child this wave loops or answers nothing.

## 7. Estate

* **Disk.** Server `/` began and ended the wave at **14 GB** free. Two
  deployments were cut from this wave's own commits and the first was deleted
  when the second superseded it; `neural-forge-matrix-4d8774f6` (1,641 files,
  90,116,202 bytes) was then pruned under the current+1 policy and manifested at
  `wave15/deployments-prune-manifest-wave15.json`, leaving `63e95cd8` (waves
  13-14) and `1c197ed4` (this wave). The wave's `/` footprint is the `nf-mx15`
  workspace at 144 KB. Every heavy artifact is under
  `/mnt/ai_data/neural-forge-matrix/wave15/` (2.6 GB: eight children, ten
  dossiers, the human-review package and the logs).
* **Ledger.** **461 -> 483**, sha
  `4fc6ba58e1f943828128dca1a93fd2035d4b5ef094712133df0fad0f1f06a6d1`. Backed up
  first (`backup/ledger.jsonl.pre-wave15-20260812T235118`, verified by sha **and**
  `cmp`), merged exactly once behind `wave15/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards, and the committed object verified byte-identical
  with `git cat-file -p HEAD:reports/matrix-campaign-20260810/ledger.jsonl |
  sha256sum`. Twenty-two rows: eleven ARCH run rows and eleven ADDENDUM 6 verdict
  rows. **Six superseded rows were dropped** by the merge and named by digest in
  `wave15/ledger-merge-report.json` -- children that were sealed twice when a
  sealer was fixed mid-wave; the merge **refuses** rather than guesses when a
  duplicated instance has no verdict pointing at one of its receipts, and it did
  refuse until the verdicts were supplied.
* **Seals.** `run/mx15/mx15_verify_seals.py` (wave 14's, with `wave15/cells`
  added and the roots still audited against disk): **11 roots present, 0
  undeclared; 483 rows, 479 resolve, 479 verify, 0 fail, 4 do not resolve** --
  the same four pre-existing non-resolvers wave 14 named. The committed wave-13
  verifier's missing `wave12/cells` root is **restored in this commit**;
  `NOTES-MX-Q` �8 says wave 13 fixed it on the host and the fix never reached git,
  so the committed file had been the broken one for two waves.
* **Human-review package.** `wave15/human-review/` -- 74 files: every dossier and
  its child and parent responses rendered verbatim, every run receipt, every
  verdict receipt, every per-case reading and every build receipt.
* **Commits.** `08999e84`, `1c197ed4`, and the sealing commit for this note.
  Nothing pushed, nothing rebased, staged by explicit path only; the Codex lane's
  ~331 uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.

## 0. The sentences

1. **Conditioning is not the missing variable.** Solving every site against a
   capture taken *through the partially built child* — the exact distribution the
   fold will meet — produces `MX15-CL640` at 0.640497x, with **better** late-layer
   solves than the stale arm (L13 0.9588 vs 0.9400) and a **better** terminal
   drift (0.818 vs 0.843), and it is the same kind of dead: 0/15, restates the
   prompt, loops. Wave 14's per-site numbers were not wrong about the wrong
   distribution; a 4x-narrowed SwiGLU cannot carry its block's function on any
   evidence.
2. **Narrowing one SwiGLU moves the next layer's input by 0.573 relative**, and
   the drift then plateaus at 0.53-0.68 rather than compounding. The stream's own
   normalization is doing that. The damage is per-site and immediate, not
   cumulative and gradual.
3. **The trajectory objective is harmful, twice.** Targeting the parent's own
   stream instead of the parent's map on the child's input costs the local map as
   well (`CT640`: 0.52-0.61 against its target, 0.50-0.59 against the local one,
   drift 0.95, 16/16 at the cap) and it is *still* harmful at unchanged width with
   full capacity and zero parameter cost (`X05T` 0/15 against `X05A` 6/15 on the
   same excision). Gap-3's corrector is this objective.
4. **head_dim 16 is not available, and RoPE is why.** The rotation is block
   diagonal over frequency planes, so a smaller `head_dim` can keep whole planes
   on a uniform stride and nothing else; the three expressible ladders retain
   **0.2394 / 0.2379 / 0.2018** of the evidence-weighted score energy against the
   0.99 that wave 14's `Δ=0` census promised. Built with both scale corrections it
   is **dead alone at 0.938368x** — 6% of the parameters removed for a *worse*
   terminal drift than a 36% fold.
5. **The 0.4666x target was arithmetic, and the arithmetic was wrong.** The three
   bites are not additive (the excised block's FFN is counted at parent width in
   one term and at 640 in another) and the tied embedding is **0.2922x of the
   parent on its own**. Built and counted, the three-bite assembly is
   **`MX15-A552` = 126,795,968 = 0.552023x**, and it is dead. 0.5x net requires
   the body to fall by 3.4x with the embedding untouched.
6. **The debt is paid.** Receipt schema v3 gives a shape-changing child an id, a
   mandatory `architecture` block, a `standard_shape` ledger column and a
   `stored_parameter_ratio` that is the size of the model that runs. Ten children
   carry sealed machine verdicts this wave, including wave 14's three, whose
   readings were re-taken on regenerated dossiers bound to their own receipts —
   `MX14-ff1280` re-reads at **5/15**, the number wave 14 recorded by hand.
