# THE UNBUILT INVENTORY — everything folding-relevant this campaign never behaviourally tested

**Audit, not a wave. Written 2026-08-28 at repo HEAD `c448503a` (wave 39 pre-registered
and in flight; nothing of wave 39's was read into or written out of this audit except its
pre-registration, to avoid duplicating its two X13 rungs).**

## 0. Why this file exists

The campaign's floor is quoted as **0.4116484014x** — `ARCH.V10CL896X09`, sealed 1/15 and
canonical 1/15, with its λ1.0 arm holding 2 at the same ratio (`DEEP_RECORD.md` §2). The
deepest child ever built is `ARCH.V12CL896X09X13` at **0.390512x**, and it holds nothing.

**0.4116x is the floor of what was BUILT.** It is not a measured floor of the fold-only
method, because a large fraction of this estate's registered fold classes were closed by a
census, a spectrum, a cosine, or an arithmetic argument — never by a child and never by a
frozen review. That is the exact move ADDENDUM 11 forbids in one line
(`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md:1502`): *"Children are built and read. Nothing
in this addendum licenses predicting a review."*

The campaign's own record says why a prediction kill is not safe here:

* **Reconstruction fidelity does not predict behaviour.** Fourteen SwiGLU coresets, every
  one inside the sealed alive fidelity band (held-out write cosine 0.81–0.94), composed to
  **0/15** (`wave14/NOTES-MX-R.md:264-269`).
* **The solver's own objective anti-correlates with the review**, eight inversions in
  thirteen ladders (`DEEP_RECORD.md` §8.3).
* **The benchmark column and the review column disagree completely**
  (`DEEP_RECORD.md` §6); `ARCH.CL896X09X13` has wave 38's *best* perplexity and holds
  *zero*.
* **A census ranked the excision axis backwards.** Wave 14's F-EXCISE pre-buy scan cleared
  exactly one of thirteen blocks (L05, +0.6653) and put block 13 at +0.4669 and block 9 at
  +0.0269, below its own 0.50 floor (`wave14/NOTES-MX-R.md:168-184`). Blocks 13 and 9 went
  on to become **the whole deep family** — `ARCH.X13` reads 13/15 — and L05, the one block
  the scan certified, reads 6/15 while `ARCH.X00` at the same removal reads 0/15. The
  census that would have killed the campaign's most productive axis is on disk, and the
  builds falsified it.

So every row below marked **CENSUS-KILL** is a place where the estate stopped on the
strength of exactly the kind of number it has five sealed laws saying it must not stop on.

Fate classes used in the table:

| class | meaning |
|---|---|
| **CENSUS-KILL** | never built; closed by a census, spectrum, cosine, arithmetic or prediction. **Rule violation. Flagged hard.** |
| **BUILT-ONCE** | one cell, one site, one K/λ. Undersampled; a single reading below the instrument's resolution. |
| **NEVER-REACHED** | registered, never contradicted, never funded. Budget, not verdict. |
| **GATED** | blocked on a stated precondition or an unanswered ruling. The gate is named and its current status stated. |
| **BUILD-KILL** | a child was built and frozen-reviewed and it died. **Legitimately closed. Kept off every shortlist.** |
| **RULE-REFUSED** | outside the contract by Cyril's rule. Not a gap. |
| **BUILT** | genuinely executed; listed only where the record needed the fate settled. |

---

## 1. THE INVENTORY

### 1.1 Round-1 slate, Track A — fold classes
`docs/FOLD_CLASS_CANDIDATES_2026-08-12.md` §2–§3, dispatched into wave 14.

| # | item + mechanism | source | fate | cheapest falsifying build now | floor-relevance |
|---|---|---|---|---|---|
| A1 | **VC-FOLD** — verbatim SwiGLU channel coreset, keep w1/w3 rows verbatim, re-ridge w2 | §3 A1 | **BUILT** — `mx14_narrow.py:100-101` *was* the coreset arm (`wave15/NOTES-MX-S.md:279-285`); became `CL*`, the campaign's body axis and 47.9% of the parent | — | — |
| A1-A / A1-B | selector arms **as children**: D²-mass, ridge leverage. (**Arm C, pivoted QR, WAS built** — it is what `mx14_narrow.py:100-101` does, so every `CL*` child in the estate is arm C) | §3 A1 arms | **CENSUS-KILL** — `census-vcfold_refit.json` measured all four selectors at keep=640 at every site; `C_pivoted` wins 10/14 and `A_mass` 4, spread 0.005–0.013 cosine against a 0.035 cliff, so no alternative-selector child was built (`wave15/NOTES-MX-S.md:279-285`) | 1 build + 1 review at `CL896` with `A_mass` instead of pivoted (~451 s GPU + review) | Low. The spread is a *fidelity* spread and fidelity does not rank; but the cliff argument is itself a fidelity argument. Cheap, weak. |
| A1-D | arm D — correlation-cluster medoids with per-cluster α-combined up rows (**the only arm whose child features leave the parent channel set**) | §3 A1 arm D | **CENSUS-KILL** — folded into the same selector census; never built | 1 build + 1 review at CL896 | Low-medium. It is the one arm that is not a subset selection, so the verbatim-vs-synthesis law (wave-14 §2.4) has never been tested on it inside the body. |
| A1-E | arm E — certified-OFF free channel drop | §3 A1 arm E | **BUILD-KILL-adjacent (legitimate measurement kill)** — the OFF certificate was an artifact of L05's single unit carrying 97.15% of gated energy; a scale-free test finds **0** dead units at every site (`wave14/NOTES-MX-R.md:121-136`) | — | None. The mechanism's object does not exist in this parent. Correctly closed. |
| A2 | **F-EXCISE** — block deletion + parent-write splice | §3 A2 | **BUILT**, extensively (X00–X13, pairs, triples; quad gated out at 3/15) | — | Exhausted on this body: `DEEP_RECORD` §8.4 — 9+13 composed reads 0/15 on both halves. |
| A2-B | F-EXCISE **arm B** — re-solve the *following* block's readers against the normed post-block-l stream | §3 A2 step (4) | **NEVER-REACHED**. The *upstream* local re-solve was priced a no-op (update_scale 0.029, `wave15` §5); arm B is the downstream half and was never built | 1 build + 1 review on a retained excision body | Low. §8.4 says the excision axis is exhausted here; arm B would have to revive a closed axis. |
| A3 | **F-C** — conv-block indivisible channel coreset at 8x | §3 A3 | **CENSUS-KILL + export-refused** — the matched-storage control has the coreset **losing** to low-rank on the conv write at 12 of 15 site-levels (`wave14/NOTES-MX-R.md:157-166`), and `Lfm2ShortConv` hardwires `hidden_size` so conv channels are not a stock config axis (`:52-57`). Never a child | Research-probe build + review; ~1 build. The S4 sibling cell the spec required for cause-attribution was never built either | Low-medium as a frontier claim (research probe only), but conv is 20.7% of the body and the *only* structural mass never folded on its own axis. |
| A4 | **S+L** — evidence-scaled sparse-plus-low-rank split | §3 A4 | **NEVER-REACHED** — and its own pre-registered screen said BUILD: the kurtosis screen skipped exactly one component free (C08/L02, 5.94) and found every other heavy (`wave14/NOTES-MX-R.md:99-103`). Last named `wave15/NOTES-MX-S.md:272`; absent from every note after wave 17 | 1 cell at the kurtosis-best SwiGLU site vs its sealed T1/T8 twin at identical storage + 1 review. The eigh/SVD/two-top-k pipeline is ~minutes CPU; cost is the build+review | **Medium-high.** It is the only class designed to survive *at* the graves where every rank truncator died, and heavy tails were measured present. Doomed only if the wave-14 cliff law (14 sites at once) applies; S+L was never asked to run at fewer sites. |
| A5 | **F-B** — GQA value-channel CUR, verbatim v-channels + exact `Wo` re-solve | §3 A5 | **NEVER-REACHED — and it PASSED its own pre-registered gate.** Real channels track the SVD floor within 25% at every site (median ratio 1.129–1.202, max 1.832), which the spec named as the build bar; "the class does **not** close causally" (`wave14/NOTES-MX-R.md:186-192`). Installed relL2/cos measured at five sites. No child, no review | 1 eager-attention research-probe build at L12 + 1 review (the materialization ruling §4.5.3 is still unanswered; zero-padding is forbidden by the kill list) | **High.** C07/C08 are the only components with **zero alive 4x coverage** in the ledger; attention is ~52M and `PLAN_GAPS` GAP 2 says the 0.25x arithmetic has no attention term at all. This is the cheapest attention purchase in the estate that already cleared its measurement. |
| A6 | **CBT** — block-Toeplitz tied ridge, full-rank, no subspace annihilation | §3 A6 | **BUILT-ONCE** — `PROBE.CBT-L12`, wave 20, one site, research probe. Reads near-parent on a frozen review; lambada **0.4600 → 0.1800**, ppl 24.39 → 134.66 (`NOTES-MX-X.md:185-212`). At L02 it loses its own head-to-head to rank truncation (cos 0.9108 vs 0.9434) | The **greedy evidence-weighted merge schedule — the class's only adaptive part — has never run** (`PHASE1…:945-947`); it needs the 8x probe. 1 build + 1 review | Low. Lost its head-to-head where the sealed twins are. Kept off the shortlists; listed because "one site is below the instrument's resolution" is in its own receipt, so the *review* half is not a real reading either way. |
| A7 | **F-DECAY** — mixer → verbatim heads + fixed-spectrum decay bank | §3 A7 | **GATED, and the gate still binds.** Not reached (`wave14/NOTES-MX-R.md:336-339`); precondition unmet — the sealed rows carry a phase label but **no position index**, so causal decay features cannot be built without a recapture | one bounded recapture + 1 build + 1 review | Medium. Attacks the far-mass (rho_far 0.52–0.69) that killed T6. Its designed head-to-head partner was built and it was not — see M-7 below. |
| A8 | **F-LANDMARK** — verbatim cached k/v states as parameters | §3 A8 | **BUILT-ONCE and abandoned mid-verdict.** Wave 20, L12, 6 of 8 KV groups, 4.0x exact per group. **Reads near-parent including correct context retrieval** (*"The day allowed is Saturday. The latest checkout time is 4:15 p.m."*); β selected on a fit split landed at 0.125 = `head_dim^-0.5`, the parent's own attention temperature, recovered by a grid never told it (`NOTES-MX-X.md:214-233`). **No distribution measurement at all** — custom runtime, no `from_pretrained` | a loader that accepts the custom module, then 1 benchmark + 1 review; or the same fold at a second site. The census the wave-16/17 notes budgeted four times was never started | **High as information, low as ratio.** It removes 589,824 params at one layer (0.26%). But it is the only mechanism that ever asked `v` to survive as verbatim points and it *lived*, and its own class-funding decision was left open on a missing benchmark for eighteen waves. |
| A9 | **HODLR-V** — hierarchical block-separable fold, verbatim diagonal | §3 A9 | **CENSUS-KILL** — diagonal interaction coverage clears the floor at exactly one component (C13/L05, 0.9287/0.9138) and that site is the massive-unit artifact; "class does not clear on merit" (`wave14/NOTES-MX-R.md:94-98`) | 1 cell at the screen-best site vs its sealed T1 twin + 1 review | Low. The kill is the honest reading of an energy metric that this estate has caught lying twice. Kept off the shortlists — but note the kill rests on the same L05 artifact that also faked S+L's kurtosis and arm E's OFF census, and no site was screened after the artifact was understood. |
| A10 | **F-D** — verbatim head + k=8 sparse-tail SwiGLU | §3 A10 | **GATED → NEVER-REACHED.** Gates (a) and (b) were satisfied by wave 14 (census sealed, VC-FOLD cell graded); gate (c), the materialization-class declaration, was never made. Not reached (`wave14/NOTES-MX-R.md:336`) | 1 matched-storage within-site contrast cell vs VC-FOLD + 1 review; storage margin is 1,440 params and needs install-time re-verification | Medium. It is the coverage extender for the marginal half of the MLP sites — exactly the sites the CL ladder now runs through. A clean loss is itself the scaling-relevant negative. |
| A11 | **Regime-4b** — certified secant-bilinear CP re-expression | §3 A11 | **GATED, gate never adjudicated.** Its gate (i) was "a fat quasi-linear population net of OFF"; the OFF census was withdrawn as an artifact and the quasi-linear population was never re-measured under the scale-free test | one census pass (CPU) then 1 build + 1 review | Low. Coverage-per-slot, not ratio. |
| A12 | **Gate-Jacobian** — shared rank-k read basis for (C11,C12) in the through-gate metric | §3 A12 | **GATED, dormant by design.** Runs only where VC-FOLD's census FAILS; the census passed at every MLP site, so the gate never opened | 1 build + 1 review at a diffuse-unit site, after the surrogate-fidelity check | Low. Spends its budget in the measured most-dangerous error class. Legitimately dormant. |
| A13 | **FSS** — fixed-spectral sandwich, Hadamard/DCT + sparse core | §3 A13 | **GATED on an unanswered ruling.** §4.5.1 asked Cyril whether fixed, data-independent, seed-free transforms price at zero parameters. **The ruling was never given and never chased** — it appears in no wave note after wave 14 | zero cost to kill (adverse ruling); otherwise a Lorenz pre-check then 1 build + 1 review | Low-medium, but it is free to resolve. An open ruling that has sat 24 waves is an estate defect regardless of the class's merit. |

### 1.2 Round-1 slate, Track B — zero-storage composition classes

| # | item | source | fate | cheapest falsifier | floor-relevance |
|---|---|---|---|---|---|
| B1 | **XGRAM** — pairwise-interference portfolio, quadratic knapsack on the interference Gram | §3 B1 | **NEVER-REACHED**, restricted not killed. The shared G artifact cleared its R²>0.5 floor at **3 of 14 layers** (L10 0.615, L11 0.706, L12 0.874); "running them on L10–L12 only is possible and was not reached" (`wave14/NOTES-MX-R.md:332-335`) | ~30 matmuls + one 30×30 eigendecomposition on L10–L12 endpoints; then 1 prospective child if the retrospective bar clears | Medium. **Superseded in substance:** wave 30's allocator fitted pairwise interference directly off the ledger (depth×vocab **4.44**, depth×mlp 1.39, mlp×vocab **1.17**, MAE 2.156) and that number produced the M2 milestone. XGRAM's per-lane form is finer but the axis-level form already paid. |
| B2 | **Ω-RIDGE** — terminal-transfer-metric folding | §3 B2 | **NEVER-REACHED**, and **partly answered by algebra since.** Wave 30 proved the reader metric **cancels out of an unconstrained writer solve** — `mx30_logitsolve` moved the weight 0.34% and all sixteen review lanes came back byte-identical (`NOTES-MX-AF.md:267-292`). The metric survives only in *rank-constrained* steps | the surviving form is the metric-weighted **selection**, which **was** built (`ARCH.CL1792M`, best body at 0.856199x, 8/15) | Low as originally specified — the algebra closes the writer branch. The selection branch paid and is already in the estate. |
| B3 | **ANTIFOLD** — drift-conjugate sequential folding | §3 B3 | **NEVER-REACHED.** Contingent on B2 step-0; never reached at wave 14, never named again after wave 15 | rebuild MXP2's 6-lane set ANTIFOLD-sequentially vs its sealed twin; 1 build + 1 review | Medium — see PG1 below, which is the same family stripped of the frozen-gate dependency that killed the KKT. |
| B4 | **W-ABSORB-SB** — zero-storage same-block absorber refit | §3 B4 | **NEVER-REACHED.** Its addressable-surface enumeration — the free step that decides whether the class has a surface at all — was never run | the enumeration is free (paper); then 1 child forward pass + 1 ridge + 1 review | Medium. Zero added parameters against a measured 28% same-block linear share. The one purchase whose drift/param ratio is infinite. |
| B5 | **Residual gauge** — exact zero-parameter rotation preconditioner | §3 B5 | **NEVER-REACHED.** Its census ran (pooled Σ̂ top direction 40.53%, top-16 73.7%, rank-for-90% 224, `wave14/NOTES-MX-R.md:145-147`); the ONE paired cell it was budgeted was to run on NKP, and NKP was buried before it | 1 paired cell (native vs rotated, same solve, same budget) on any axis-sensitive lane + 1 review | Low. Its designated beneficiary is dead; it would need a new host. |

### 1.3 Round-1 completeness critique — five gaps
`docs/FOLD_CLASS_CRITIQUE_2026-08-12.md`, registered as `MATRIX_CAMPAIGN_DESIGN` ADDENDUM 11.3.

| # | item | fate | cheapest falsifier | floor-relevance |
|---|---|---|---|---|
| **C1** | **Vocabulary-axis row TYING** — top-h rows verbatim, Zipf tail tied into c clusters with a closed-form shared row. 67.1M / 29.2% of the model | **CENSUS-KILL — and this is the worst one on the list.** Killed at wave 14 on a nearest-neighbour cosine over a 4,096-row sample of the post-12,288 tail: median **0.2977** (`wave14/NOTES-MX-R.md:65-74`). Never built. **The vocabulary axis then became the campaign's single best axis** — `ARCH.V19` 14/15 at 0.794570x, and *every child in the deep family is a vocabulary fold*. The census compared tying against **verbatim**; the mechanism that won compares against **deletion**, and tie-vs-delete is the comparison that was never made. At cosine 0.30 a tied row is a poor stand-in for its members — and a deleted row is a *worse* one | one CPU census pass (frequency-mass Lorenz + intra-cluster coherence already specified) then **one 40-second CPU gather-class build** + 1 review, at the K where the sealed ladder dies | **Highest on the list.** The floor at 0.4116x and the death at 0.4070x are **1,024 vocabulary rows apart** (`DEEP_RECORD` §5.2). Tying is the only registered mechanism that puts *content* into rows the coreset deletes outright, at the exact axis and the exact rung where the campaign's floor is set. §8.5's decay law says the X09 ladder sheds retrieval then uncertainty as K falls; a tied tail is aimed at precisely that. |
| **C2** | **Score-bilinear joint q/k** — factor the observable invariant `Wq_h^T R_Δ Wk_h`, not the factors. C06 has **zero coverage in the entire estate** | **NEVER-REACHED — and it produced THE BEST SPECTRUM IN THE WHOLE MEASUREMENT BATTERY.** Top **16** of 64 singular directions carry a median **0.9948** (L02), **0.9985** (L08), **0.9926** (L12); worst head 0.9414. Wave 14's own words: *"Not built this wave; it is the strongest unspent lead"* (`wave14/NOTES-MX-R.md:107-116`). **It appears in no wave note, ever again.** Twenty-five waves | two stated limitations bound it (it is the Δ=0 member of the RoPE family, and it uses raw projections not the `q/k_layernorm` outputs) — so: one per-RoPE-block two-sided LRA (pure CPU, exact, no capture), 1 build + 1 review at L12 | **Highest never-reached.** Exact 4x per head pair, zero aux, on 6 attention layers. Attention is the unpriced term in `PLAN_GAPS` GAP 2's 0.25x equation, and this is the one class in the estate whose free census came back at 0.99. Field synthesis #25 independently arrived at the same territory from the 2026 literature. |
| **C3a** | **Exact simultaneous terminal re-solve** (one-shot joint block ridge over all installed writers) | **BUILD-KILL.** Built at wave 29 as GLOBAL-KKT on `V19CL896`: 20 installed writers, both arms improved their held-out objective and both read **0/15** against a 2/15 control; the rung-0 discriminator measured *why* — the linearisation's error is **1.5–2.3×** the repair it predicts (`NOTES-MX-AE.md:221-269`) | — | **None, and it is honestly closed** — but only the *frozen-gate* branch. The seal is explicitly narrow: "any global form that does not depend on frozen gates" stays open. |
| **C3b** | **Portfolio-amortized corrector** — one small additive module charged against the whole portfolio (100k against a 12.06M excision = 120:1) | **NEVER-REACHED.** Wave 15 argued it is "the trajectory objective in a cheaper package" and that objective ran twice and died (`CT640` total collapse, `X05T` 0/15) — but the amortized form itself was never built and the note says it "should be built only with an explanation of why it escapes both" (`wave15/NOTES-MX-S.md:270-278`) | 1 child forward pass + 1 closed-form solve + 1 review, on any deep child | Medium. Against a 12M excision the ratio arithmetic is unusually favourable; against a 40-second vocabulary rung it is not. |
| **C4** | **Per-case behavioural attribution artifact** — regress case-level pass/fail over the sealed children on terminal error signatures, yielding sensitivity directions | **NEVER-REACHED as specified.** Budgeted and unrun in waves 14, 15, 16, 17 and 18 by name (`wave16/NOTES-MX-T.md:288-291`, `wave17/NOTES-MX-U.md:299-302`, `wave18/NOTES-MX-V.md:316-320`). Wave 29 ran a *different, narrower* thing — a LIST→SET coordinate attribution — and found attribution **is not local** (top 24 coordinates carry 15.8%, `NOTES-MX-AE.md:170-176`) | ~15–232 ridge probes, zero GPU, restricted to L10–L12 endpoints | Low-medium as a case-buyer, high as an ordering function. It is the only *ex-ante* attack on rare-capability displacement; everything else in the estate addresses it post-hoc. |
| **C5** | **Input-conditional piecewise folds** — two 8x experts + a fixed evidence-derived router | **RULE-REFUSED.** The round-2 contract states "no MoE/conditional execution" (`WAVE23_ROUND2_SLATE_2026-08-13.md:5`). Correctly closed by rule; the §4.5-style ruling it asked for was overtaken | — | Off the shortlists. Named because it is the only *structural axis* absent from the whole estate and the refusal is a contract choice, not a measurement. |
| **C-x** | **G-weighted / terminal-attenuated selection column** (the cross-cutting sub-gap) | **BUILT** at waves 30–31 as the reader-metric selection arm; `ARCH.CL1792M` is the best body at 0.856199x (8/15) | — | Paid. |

### 1.4 Round-2 slate (wave 23)
`docs/WAVE23_ROUND2_SLATE_2026-08-13.md`.

| # | item | fate | cheapest falsifier | floor-relevance |
|---|---|---|---|---|
| S1 | **VOCAB-CORESET** | **BUILT**, and it is the champion axis | — | — |
| S2 | **SEGBASIS** — seam-gauged per-segment reader bases on `hidden_size`, zero added tensors | **NEVER-REACHED.** Its P2 census was costed "CPU only, zero builds" and wave 23 found that wrong — it needs per-endpoint reader moments, i.e. a capture — so census and wave both went unbuilt (`NOTES-MX-AA.md:265-279`). Never named again in 15 waves | P2 census (a capture's worth of machinery) + 1 build + the mandatory T* twin + 2 reviews | Low-medium. It is on the width axis, which wave 30 measured as the *worse* axis at matched ratio (`CL1792` 7/15 vs `W896` 4/15 at three decimals). But it is the only proposal that removes the shared-budget constraint rather than re-weighting inside it, and six objectives have now lost *inside* it. |
| S3 | **ORTHO-PAIR** — cross-error orthogonalization in the reader metric | **CENSUS-KILL, and it is a NAMED RULE VIOLATION.** The slate's FIX 1 is explicit: *"the rho census publishes either way and NEVER gates — the soft arm on the living pair is built regardless of census outcome"* (`WAVE23…:96-97`). P1 was not run at wave 23; a related ρ census ran at wave 24 and its **pooled scalar inverted the premise** (the living pair had the largest ρ; the murdered pair sat *below* additive — `INTELLIGENCE_RETENTION_PLAN…` #13, "Corrected record"), and the soft-arm child that the fix said must be built regardless **was never built** | 1 soft-arm child on `X07KV6JQC` + 1 review; zero storage delta, value-only inside a sealed pair shape | **Medium-high.** L3's cross term is the ~3-case/component tax and it has never been *controlled*, only measured. The per-endpoint reader-metric form the plan's #13 identified as the untested refinement is still untested. |
| S4 | **PAIR-KKT** on the living pair `ARCH.X07KV6JQC` | **NEVER-REACHED on its own pilot.** The mechanism was built at wave 29 on a *different* base (`V19CL896`, the deep pilot) and nulled. The living-pair pilot both the slate and plan #4 named first — 7/15 at 0.926936x with 4-case headroom, byte-gate checkable — was never built | 1 conditioned capture + 1 build + 1 review | Low. Wave 29's null is mechanistic, not count-based: the frozen-gate response's error exceeds its repair. That bound was measured at 0.48x; at 0.93x it is untested, but 0.93x is not floor territory. |
| S5 | **LP-ATLAS + ATTR** — the λ-portfolio capability atlas | **NEVER-REACHED as specified.** Stage-1's gate build (W896 with check rows restricted to conversation/multi-turn evidence) was never built; the ATLAS artifact (profile × 15-case hold vector + Guttman residual) was never produced. What *was* built later is per-writer LS λ variation (`G01/G03/G10`), which is the sealed baseline the fixes explicitly said was **already** per-writer | 1 restricted-arm build + 1 review (Stage-1 is a single gate build) | Medium. `DEEP_RECORD` §8.1–8.2 show the λ arm is worth +1 to +2 and lands at THREE at five of six K values — the largest repeatable free purchase in the deep zone — and the *case-restricted* form of it has still never run. §8.6 warns it can LOSE below 0.437349x, which is exactly where the atlas would be spent. |
| S6 | **29-DIAGONAL RECALIBRATION** — per-reader norm-gain refit across all 28 block RMSNorm gains + final norm | **NEVER-REACHED.** `W768R` was never built; never named again. Note precisely what is untested: the **final** norm has been re-solved dozens of times (every `G` arm); the **28 block** gains have never been re-solved at all | 1 build (`W768R`) + 1 review. Seconds of compute — 29 SPD Choleskys — against the campaign's richest identical-parameter control set | Medium. Zero param delta, zero directions exchanged, outside wave-21's basis-invariance condemnation by construction. Honest expectation 0 ± 1, which the extension bank can now adjudicate and could not in wave 23. |
| S7 | **NKP-VANLOAN** + Monarch control | **CENSUS-KILL.** P3 ran (~1 CPU hour, 42 SwiGLU tensors × 4 grids): top Kronecker term holds **0.25%** of the energy, and the class loses to plain rank truncation at **38 of 42** sites and to Monarch at **all 42**. "The cell was not built" (`NOTES-MX-AA.md:253-263`; `PHASE1…:954-967`). **The kill gate was "flat spectra PLUS a loss to the twin" and the loss to the twin was measured in reconstruction error, not behaviour** — against this estate's own five-times-sealed law that reconstruction does not rank | 1 sealed cell at `w1@L05` vs the sealed rank-truncation twin + 1 review | Low for NKP itself — 0.25% is a very large margin and the flat-spectrum doom is corroborated by the residual spectrum. **But the census's own WINNER was never built:** Monarch beat both arms at all 42 sites, and Monarch is registered as never-priced-behaviourally in `PHASE1…:968-970`. |
| S8 | **EARLY-SPARED ASYMMETRIC WIDTH** — spare the 8 indicted early-block readers + the embedding write | **NEVER-REACHED.** Not built (`NOTES-MX-AA.md:281`); never named again | 2 builds (spared + inverted mirror) + 2 reviews; research probe only | Low for ratio, **medium for the map**: it is a pre-registered binary that either localises L2's wall or stops all future early-block rescue spending. The estate has since spent nothing on early-block rescue anyway, so its value has decayed. |

### 1.5 Round-2 completeness critique — three gaps
`docs/WAVE23_ROUND2_COMPLETENESS_CRITIQUE_2026-08-13.md`, registered as ADDENDUM 15.4.

| # | item | fate | cheapest falsifier | floor-relevance |
|---|---|---|---|---|
| CG1 | **INTERMEDIATE-LADDER** — `intermediate_size` on the standard column | **BUILT** at waves 30–31 (`CL1280/1408/1536/1664/1792`) and it produced the campaign's **M2** milestone: `ARCH.V19CL1792` 7/15 at 0.650769x, with the vocabulary axis costing **ZERO** cases on that body | — | Paid, and it is the single best-value item this audit found already executed. |
| CG1-b | **the 640 → 896 bracket on that same ladder** | **NEVER-REACHED — and nobody has noticed it.** The built rungs are 640, 896, 1280, 1408, 1536, 1664, 1792. `ARCH.CL640` reads **0/15** at 0.640497x; `ARCH.CL896` reads **4/15** at 0.688431x. **Nothing has ever been built inside a 256-channel bracket that spans a 4-case transition** — the exact structure `DEEP_RECORD` §5 names twice on the vocabulary axis and calls "the sharpest transition anywhere in this campaign" | 1 body build at `--ff 768` (~451 s GPU by `CL896`'s own figure) + 1 review; then vocabulary rungs on it at **40 seconds of CPU each** | **Highest cheap item.** Wave 30's fitted interference makes **mlp × vocabulary the cheapest pair in the model (1.17 cases)** and depth × vocabulary the most expensive (4.44) — yet *every child in the deep family spends an excision*. Arithmetic, at 43,008 params per channel over 14 sites, on a model that reproduces `V10CL896X09`'s 94,552,832 and `CL896X09`'s 151,175,936 to the byte: `CL768` = **0.664464x**; `CL768` + K=9,216 = **0.413383x** *with no excision at all*, i.e. the current floor's depth reached on the cheap pair alone; `CL768` + X09 + K=10,240 = **0.387682x**, **below the deepest child ever built (0.390512x)**. A `CL736` body reaches 0.411956x at K=10,240 with no excision — the floor exactly, on the cheapest pair in the model. |
| CG2 | **REBASE-AFTER-EXCISION** — re-derive the residual basis on the *composed child's surviving* reader set | **NEVER-REACHED.** Queued at ADDENDUM 15.4.2, restated as still-queued at `NOTES-MX-AA.md:282-287`, never funded, never named again | 1 conditioned capture through the excised child (shared infrastructure with S3's arm) + 1 build + 1 review, at a sealed pair shape | Medium. It is **the only lever in the estate that reduces the DEMAND side of L2's wall** — fewer endpoints to serve, more retained energy per survivor. Every one of the six condemned basis objectives re-weighted the same 29-endpoint *parent* capture; basis-invariance has never been tested under a changed reader census. Its own honest hope statement predicts a tie, and a tie is a law. |
| CG3 | **RESOLUTION RIDER** — frozen 30-prompt extension bank, gates fire on the sealed 15 only | **BUILT** at wave 29 and **validated**: Spearman 0.8023 on ten sealed checkpoints, 0.8517 on sixteen (`NOTES-MX-AE.md:27-46`); v2 authored at wave 34 | — | Paid — **but note the ordering defect.** The rider's binding rule is that *any kill on a 1-case margin must show extension-bank concordance.* The bank landed at wave 29. **Every census kill in §1.1–§1.5 above predates it, and none was ever subjected to it.** |

### 1.6 Intelligence Retention Plan #1–#16 and PLAN_GAPS 1–4
`docs/INTELLIGENCE_RETENTION_PLAN_2026-08-14.md`, `docs/PLAN_GAPS_2026-08-14.md`, funded by ADDENDUM 16.

| # | item | fate | cheapest falsifier | floor-relevance |
|---|---|---|---|---|
| #1 | INSTRUMENT DEEPENING (extension bank) | **BUILT**, validated (wave 29) | — | Paid |
| #2 | RETOK-BODY — re-solve `CL896` on the retokenized stream | **BUILD-KILL by measurement, and a clean one.** Wave 30 sealed the stream and found the V19 tokenizer's segmentation **identical to the parent's at all 4,181 positions, at every rung of the ladder including the 4/15 cliff child** (`NOTES-MX-AF.md:86-135`). The hypothesis's object does not exist | — | None. Correctly and completely closed. |
| #3 | CHASM BRIDGE | **BUILD-KILL.** Arithmetically void (the parent-shaped bases already carry `C01.T5.S4` owning the whole table) *and* the children read 3/15 and 2/15 | — | None |
| #4 | GLOBAL-KKT | **BUILD-KILL** (frozen-gate branch only) — see C3a | — | None for that branch; PG1 is the surviving branch |
| #5 | MARGIN-RESTORE | **BUILD-KILL by pre-read, legitimate.** The 0/15 child at 0.452734x retains **96%** of the parent's generative margin, so a margin-collapse repair has nothing to correct on the child it was most wanted for (`NOTES-MX-AE.md:178-199`) | — | None as specified |
| #5-succ | the **sink / massive-activation targeted repair** that pre-read aimed at | **CENSUS-KILL (half) + NEVER-REACHED (half).** Wave 30 showed the wave-29 correlation was a coordinate-system artifact — every basis on the ladder keeps the direction to three decimals, so a carve-out buys ≤0.8% of one direction — killing #2(a) *on arithmetic*. #2(b), the targeted diagonal repair, was "not built, and deliberately" (`NOTES-MX-AF.md:439-446`). But the same pass found the channel **is destroyed by 89–98.5% at every width including the one that reads 4/15**, i.e. by a **write** failure, and **no repair has ever been aimed at a write failure** | 1 diagonal ridge on the sink/massive dims against parent cached values, zero storage delta + 1 review | Medium. The carve-out is dead; the write-side repair the same measurement newly implied has never been attempted. |
| #6 | CASE-VECTOR SHOPPING (attribution + masked terminal-gain) | **NEVER-REACHED as specified**; wave 29 ran a narrower LIST→SET attribution and found it non-local | Stage 0 is zero GPU; Stage 1 is 1 build + 1 review | Low-medium (band-lift at 0.742x, not a floor move) |
| #7 | COMPLEMENT-MERGE | **BUILT** (wave 29 `B75`; wave 32 blends). Zero-dimensional swap; moved one rung | — | Paid |
| #8-B | **CHANNEL LEDGER arm B — the giveback rung, `CL896` → 960 or 1024 (~0.495–0.507x)** | **NEVER-REACHED.** The ledger's CL family contains **no 960 and no 1024 rung**. The plan calls it "the registered never-priced shape move" and it is *still* never priced | 1 build + 1 review | Medium. It is the one rung explicitly aimed at "the first alive-degraded child at 0.495–0.507x". Since then the estate has gone far deeper by other means, so it now buys a *shape* datum rather than a frontier move. |
| #8-A | CHANNEL LEDGER arm A — G-selection re-run at 896 | **NEVER-REACHED at 896.** The metric-selection arm was built at 1792 (`CL1792M`, 8/15) and never at the body the deep family actually uses | 1 build + 1 review (~451 s GPU + review) | **Medium-high and cheap.** `CL1792M` beat its own stream-metric control at 1792. The identical one-line change has never been made on `CL896` — the body under *every* child at and below 0.4517x. |
| #9 | DEEP-BODY K DOSE-RESPONSE (K 22,528 / 25,600 / 29,696) | **NEVER-REACHED at those K**; `V32CL896` covers the top of the range | 3 × 40 s CPU + 3 reviews | Low. Expected flat, and flat was the deliverable. |
| #10 | **GENERATION-MANIFOLD CAPTURE — parent free-run states as evidence** | **NEVER-REACHED.** No cell, no capture, no mention in any wave note after the plan | 1 pure-inference capture wave + 1 rebuild + 1 do-no-harm control + 2 reviews | **High.** It is "the only contract-legal way a solve sees autoregression", and **43.6% of everything this campaign scored as a behavioural failure is a compression artifact** — loops, cap-rides, empty strings — i.e. autoregressive failures that teacher-forced evidence weights at **zero** (`ACTUAL_MATRIX…` §3). Waves 34 and 35 then proved decode-time knobs cannot recover them: **20 child-settings, 300 lanes, zero recoveries, every setting net-negative** (`DEEP_RECORD` §9). Whatever fixes those lanes is upstream of the decode, and this is the only registered mechanism that looks there. |
| #11 | LADDER-CLIMB — case-restricted OOS-λ portfolio | **NEVER-REACHED** | 1 build + 1 review | Low. Its own honest prior is the lowest on the slate: no λ variation in 28 waves ever moved arithmetic/logic/coding. |
| #12 | **V-ROUTE — tokenizer-preserving verbatim head + sparse-OMP tail** | **NEVER-REACHED.** No cell in the ledger | 1 solo build (k=16 and k=38 arms) + `V-ROUTE×CL896` in the same wave, per the fix + 2 reviews | **Medium-high.** It is the only vocabulary mechanism that keeps the tokenizer unchanged, so retokenization shift is structurally impossible — and it is the *sibling* of C1 (row tying): both put content into rows the coreset deletes. Anchored on `MXN2`, 10/15 at 0.7805x, the campaign's deepest parent-band child. Its stated thesis — `V-ROUTE×CL896` at ~0.49x reading ≥4/15 — is directly a floor claim. |
| #13 | SIGNED-LEDGER — per-endpoint reader-metric ρ retrospective | **NEVER-REACHED in its untested form.** The pooled scalar census ran (wave 24) and inverted the premise; the per-endpoint refinement never ran, and the **pre-registered closure it specified — sealing "composed error norm does not determine review" and closing ORTHO-PAIR, ANTIFOLD and XGRAM in one stroke — was never fired either.** Those three classes are therefore in limbo: not built, not formally closed | Stage 0 only: leave-one-out over ~30 sealed pair reviews, zero GPU | Medium as an ordering function; **high as estate hygiene** — it is the one action that would legitimately close three of §1.2's rows instead of leaving them un-adjudicated. |
| #14 | REVIEW-ORACLE BEAM SEARCH | **NEVER-REACHED** | library manifest + assembler + controller = one build-class infrastructure deliverable, then a capped pilot | Low-medium. Its own falsifier ("the rim is one-move locally optimal") is a publishable ceiling; but it is the most expensive item on the plan and the rim has moved a long way since it was written. |
| #15 | EVIDENCE-PORTFOLIO — mixture of second moments over K capture distributions | **NEVER-REACHED** | 1 twin at fixed shape + 1 review | Low. Its own stated headwind: every evidence-side change on record priced ≤1 case — and wave 31 measured 34% fewer fit rows costing six cases, which bounds the axis from the other side. |
| #16 | MLP MASS — uniform ladder (frontier) + nonuniform per-block ≥4x probe (research) | **BUILT (uniform half)**; the **nonuniform per-block probe — protected blocks {0,2,4,12}, allocation order 13,7,9,11,1,6,8,10 — was NEVER built**, nor was its named falsifier cell (V19 + block-13 trio at 4x vs the sealed byte-twin `V19X13`) | 1 probe cell + 1 review | Medium. The uniform ladder's own reading is that the axis flattens onto 896; per-block heterogeneity is the escape hatch wave 14 named ("global 4x dies, 3-at-a-time survives") and it was never taken. |
| PG1 | **Residual-corrected sequential solves** — shift B's ridge target by A's computed write-error; zero new capture, no frozen gates | **NEVER-REACHED, and explicitly left open by name at the end of wave 29** (`NOTES-MX-AE.md:334-343`) | one matrix product on existing cached activations + a verbatim re-run of B's sealed ridge + 1 review. **No capture, no operator assembly — strictly cheaper than the KKT that was built** | **Highest of the composition attacks.** It is the surviving branch of the exact seal wave 29 wrote: the KKT null closed the *terminal-metric, frozen-gate* branch and the note says this family "stays open by name". It attacks the ~3-case/component tax per-site and rung-by-rung, compatible with one-component-per-rung — and the tax constant is what stands between 0.4116x and anything below it. |
| PG2 | **Attention mass / KV-head grouping ladder toward depth** | **NEVER-REACHED.** "GAP 2's attention mass, still unpriced" (`NOTES-MX-AE.md:336`). Head merges exist (KV6JQ/KV6JS, waves 17–22) but were never laddered or composed toward depth on a `V`/`CL` base | 1 KV-grouping rung on a V19 base (`num_key_value_heads` is a stock config int) + 1 review | **High.** §5.6's 0.25x equation has three terms and the attention term has never been measured. Attention is ~52M of 162.6M of body. Together with C2 (score-bilinear q/k) and A5 (F-B), this is the same unopened door approached three ways. |
| PG3 | LIST→SET autopsy | **BUILT** (wave 29): attribution is not local | — | Paid |
| PG4 | Downward K rungs + composed cell | **BUILT** extensively (K=12,288 / 10,240 / 9,216 / 8,192 on two bodies) | — | Paid — it is what produced the current floor |

### 1.7 Field research synthesis, 33 deduped findings
`docs/FIELD_RESEARCH_SYNTHESIS_2026-08-28.md`, committed at wave 29 with ADDENDUM 16.6's
statement that everything beyond three adoptions is *"built by no part of wave 29"*.

| # | item | fate | cheapest falsifier | floor-relevance |
|---|---|---|---|---|
| 1 | Evidence-refresh solving (propagate-then-solve + re-rank) | **NEVER-REACHED — ordered and unrun for three consecutive waves** (`NOTES-MX-AF.md:433-438`, `NOTES-MX-AG.md:314-317`) | a three-fold composition solved twice with per-state evidence caches: 2 builds + 2 reviews | Medium. The number it must beat is now *measured* (4.44 / 1.39 / 1.17 per axis pair), so it is better posed than when it was ranked #1. Overlaps PG1. |
| 2a | Sink/massive-activation carve-out | **CENSUS-KILL, and this one is legitimate** — the arithmetic is direct and decisive (≤0.8% of one direction at width 512) | — | None |
| 2b | Targeted diagonal repair on sink/massive channels | **NEVER-REACHED** — see #5-succ above | 1 solve + 1 review, zero storage | Medium |
| 3 | Multi-axis phase-transition budget allocator | **BUILT** (wave 30) → M2 milestone | — | Paid; the best-value adoption on the list |
| 4 | LACE-SVD interpolated-target shrinkage | **NEVER-REACHED — ordered three times (waves 30, 31, 32) and not run each time** (`NOTES-MX-AH.md:361-364`) | a one-line change to existing output-factor solves; 1 A/B build + 1 review | Medium. Cheapest untried change to the solve itself, aimed directly at the additive-interference tax. |
| 5 | EoRA activation-eigenspace-weighted SVD | **NEVER-REACHED** | swap the truncation metric inside a width/rank fold; 1 build + 1 review | Low-medium — wave 30 showed the metric cancels in unconstrained writer solves and lives only in rank-constrained steps, which is exactly what this is |
| 6 | Logit-space / reader-metric solving | **BUILT** (waves 30–31): metric cancels in the writer solve (algebra, byte-identical lanes); the surviving selection form paid (`CL1792M`) | — | Paid |
| 7 | Decision-margin phase analyzer / fold-forbidden map | **PARTIAL** — landed inside wave 30 §6 as per-coordinate margin attribution | the DM(l)/OF(l) profile and the fold-forbidden map proper | Low-medium |
| 8 | CoCurve co-pruning curvature (interference Gram from single-unit ablations) | **NEVER-REACHED** | M forward passes → interaction Gram → one budgeted QP; no gradients | Medium. It is XGRAM's idea with a published closed form and a source of the off-diagonal terms the estate's additive model cannot see. |
| 9 | ReplaceMe depth fold (block deletion + ridge T fused into `W_down`) | **NEVER-REACHED as specified** — the estate built pure deletion; the fused linear replacement `T` was never installed | cache 3 tensors, 1 ridge, fuse, delete; 1 review | Low. §8.4 says the excision axis is exhausted on this body. |
| 10 | Diagonal interface hygiene (per-channel scale absorbed into adjacent weights) | **NEVER-REACHED** as a standing post-excision step | closed-form from cached statistics, zero params; 1 review | Medium. Zero-parameter, applies to every depth cell already in the estate, never tried. |
| 11 | Norm-conserving interface projection (Sinkhorn / Procrustes) | **NEVER-REACHED** | gradient-free iterated normalization on fold interface matrices; 1 build + 1 review | Low-medium |
| 12 | Nested-rank SVD behavioural ladder + matryoshka spectrum probe | **NEVER-REACHED** | frozen review at several nested ranks from one solve — near-free at the solver, N reviews | Medium as a *protocol*: it turns one solve into a behavioural ladder, which is exactly what the estate's cheap vocabulary ladders already prove is the highest-information format. |
| 13 | **Unembedding rank-repair (demotion correction)** | **NEVER-REACHED** | cache parent+child logits on frozen evidence; solve a diagonal scale+bias per vocab row (~2·\|V\| params) restoring parent token ranking; 1 review | **High.** The floor *is* a vocabulary rung, the death at K=9,216 *is* an emission failure, and this is a closed-form repair on exactly the tied table that the whole deep family folds. The richer low-rank tier is budgetable inside net reduction. |
| 14 | **Anti-loop cauterization (param-REDUCING micro-fold)** | **NEVER-REACHED** | difference-of-means repetition direction from cached hiddens at loop onset vs healthy generation (the estate has both, in 846 archived artifact lanes); zero the implicated rows; 1 review | **High.** 43.6% of all sealed misses are artifacts and 61% of wave 38's lanes are; decode-time knobs are sealed dead as a remedy (20 settings, 300 lanes, 0 recoveries). This is the only registered mechanism that removes parameters *and* targets that population. |
| 15 | Entropy-trajectory gate + scalar temperature absorption | **NEVER-REACHED** | 1-D bisection on cached logits, absorb into final-norm gain; costs nothing | Medium. Free, and the estate has no leading indicator for emission collapse. |
| 16 | Parent-as-verifier acceptance-rate gate | **NEVER-REACHED** | run the frozen parent as a speculative verifier; pure inference | Low-medium (instrument, not a fold) |
| 17 | Parent-agreement triage on a pinned token cache | **NEVER-REACHED** | one forward pass per child | Low-medium (triage, not a fold) |
| 18 | **Procrustes K→V tying — delete `W_v` where V is a linear image of K** | **NEVER-REACHED** | per attention layer, one Procrustes/ridge solve on cached pairs; export `W_v' = R·W_k`; 1 build + 1 review | **Medium-high.** A new param-cut axis on the unpriced attention mass, closed-form, no capture. Fourth approach to the same unopened door as C2 / A5 / PG2. |
| 19 | Attention removal + exact FFN fusion + single-seam repair | **NEVER-REACHED** | pairwise block-dependency matrix from cached forwards, then 1 build + 1 review | Medium |
| 20 | Clustered LM head + per-cluster low-rank factorization | **NEVER-REACHED** | spherical k-means on head rows (gradient-free) + per-cluster SVD + ridge repair; 1 build + 1 review | **Medium-high.** It extends the champion axis with a *different* mechanism (cluster + factor rather than delete), on the 67.1M table where the floor is decided. Sibling of C1 and #12 V-ROUTE. |
| 21 | Saliency atlas + logit-KLD depth ranking | **NEVER-REACHED** | scoring only, from existing caches | Low (scoring). Notable: NVIDIA chose logit-KL over activation-MSE for depth, which is this estate's misranking law from outside. |
| 22 | MIP assembly of fold schedules | **NEVER-REACHED** | encode measured behavioural costs, run a MIP | Low-medium; overlaps #14 (beam search) and wave 30's allocator, which already paid |
| 23 | OBS inverse-Hessian compensation vs ridge, head-to-head | **NEVER-REACHED** | A/B on an existing lineage, consuming Grams already cached; 1 build + 1 review | Medium. It is a *different repair family* from ridge — the estate has used exactly one repair family for 39 waves. |
| 24 | Nested-budget joint ridge (MatQuant structural transplant) | **NEVER-REACHED** | stacked/weighted ridge across interacting sites; 1 build + 1 review | Medium. Explicitly aimed at getting below the frontier with no new cell types. |
| 25 | Rotary-pair-preserving block-structured QK solves | **NEVER-REACHED** | split cached QK evidence into rotary and content blocks; SVD/ridge the content subspace only | **High — and it is the 2026 literature independently arriving at C2.** The two should be built as one cell; #25 supplies the RoPE handling that C2's census flagged as its stated limitation. |
| 26 | Frequency-stratified vocab-fold gates + explicit vocab×body interference measurement | **NEVER-REACHED** | a protocol change plus one measurement | Medium. It gates the champion axis with rare-token probes — and the estate's own ladder decays by *losing specific lanes in a fixed order* (§8.5), which is what a stratified gate is for. |
| 27 | Grading-protocol upgrades — state-tracking probes + blind private split | **NEVER-REACHED** | authoring, zero GPU | Medium. The claim it protects against — a fold passing coherence review while silently losing computation — is exactly what `ARCH.KKTD`'s twelve fluent empty lanes look like. |
| 28 | **Nested-parent campaign — fold Gemma-4 E4B** | **NEVER-REACHED** | run the existing fold pipeline on a legitimately nested parent | **Attribution, not ratio — and the synthesis calls it "the single most informative strategic experiment on this list."** Deep sub-0.69x success there would prove the chasm is *parent conditioning*, not fold mechanics. It cannot move THIS parent's floor; it can decide whether the floor is a property of the method or of LFM2. |
| 29–33 | Cross-layer weight tying · sibling fold-child merge (RegMean) · 1-gram embedding repair · anti-symmetry interface diagnostic · ridge-fit linear mixer | **NEVER-REACHED**, all five | 1 exploratory cell each | Low individually. #31 (1-gram embedding repair, fully standard export) is the cheapest and sits on the champion axis; #30 overlaps wave 29/32's blends, which paid once. |

### 1.8 `PHASE1_FOLD_ONLY_FRONTIER` §6 — never priced behaviourally

| item | fate | falsifier | floor-relevance |
|---|---|---|---|
| **Weight-space CUR / interpolative decomposition** (actual rows and columns of the parent + a coefficient block) | **NEVER-REACHED.** The *feature-space* CUR was built (that is `CL*`); the weight-space control the slate specified as "one cell, control only" never ran | 1 matched-storage twin against a `CL` cell + 1 review | Low-medium. A control, not a frontier move — but §6's own sentence stands: *"The fold-only frontier is open until they are priced behaviourally."* |
| **Monarch / block-diagonal + low-rank** | **NEVER-REACHED — and it WON the census that buried its competitor**, beating NKP at 42 of 42 sites and rank truncation on the same surface, at 0.35x storage | 1 cell at the census-best site + 1 review; the P3 census surface already exists | **Medium-high.** It is the closest structural relative of the one thing known to fold here (MLP mass), it is already measured as the best of the three structured families on this parent, and it has never been built. |
| **Tensor-train / tensor-ring on the reshaped weight** | **NEVER-REACHED** | 1 cell + 1 review; TT rank scans under the selection-split discipline | Low. Storage arithmetic is favourable; both spectra on this parent are flat, which is the same doom NKP met. |
| CBT greedy merge schedule | **NEVER-REACHED** (see A6) | the 8x probe | Low |
| M-7: **the F-DECAY vs F-LANDMARK head-to-head at L12** | **HALF-BUILT.** The slate funds "only the winner" of a one-cell-each comparison; F-LANDMARK was built and F-DECAY never was, so the attention estate has never been funded *or* closed on this axis | 1 F-DECAY cell at L12 (after the recapture) + 1 review | Medium. A designed comparison with one arm missing is not a comparison, and it has been in that state since wave 20. |

### 1.9 Cheap unbuilt rungs the record names by ratio

| item | source | fate | cost | floor-relevance |
|---|---|---|---|---|
| **Inside the `V8CL896`/`V9CL896` 0/15 bracket** — 1,024 rows, 0.0045651159 of stored ratio | `DEEP_RECORD` §5.1 | **NEVER-REACHED**; "nothing has been built inside it" | one 40-second CPU pass to halve | Direct. Locates the sharpest transition in the campaign. |
| **Inside the `V9CL896X09`/`V10CL896X09` bracket** — same width, 0.0257 deeper, brackets **the floor itself** | `DEEP_RECORD` §5.2 | **NEVER-REACHED** | one 40-second CPU pass to halve | **Direct — this is the floor's own bracket.** |
| λ1.0 arm on `CL896X09` at K = 14,336 and K = 16,384 (§8.2's named falsifiers) | `DEEP_RECORD` §8.2, `NOTES-MX-AP` §10.4 | **NEVER-REACHED** | ~100 s GPU each | Direct: §8.2's rung account makes a falsifiable prediction at each. |
| λ0.3 arm at K = 8,192 — §8.1's only untried rung, on the campaign's 0/15 vocabulary death | `DEEP_RECORD` §8.1 | **NEVER-REACHED** | ~100 s GPU | Direct: it is the stated falsifier of the "lands at THREE" law. |
| `V9CL896X13` (wave 39 builds 14,336 and 10,240, not 9,216 — stated in its own deviations) | `wave39/prereads/preregistration-wave39.json` | **NEVER-REACHED** (deliberately deferred by wave 39) | one 40-second CPU pass | Direct |
| §8.3's *n* — a wave that spends **two** λ solves instead of one | `DEEP_RECORD` §8.3, `NOTES-MX-AP` §10.5 | **NEVER-REACHED** since wave 37 | ~200 s GPU | Instrument. n has been stuck at 13 for two waves. |
| the 736 → 720 width rung, the max-min basis at 720, the `evidence-pca` basis at 768, `--arm stale`, the 768→640 midpoint | `NOTES-MX-Z` §7, `wave19/NOTES-MX-W` §11 | **NEVER-REACHED** (five items) | 1 build + 1 review each | Low. Width is the measured worse axis (`CL1792` 7/15 vs `W896` 4/15 at matched ratio) and every one of these sits in a 0/15 region. |
| the excision **quad** | `wave18/NOTES-MX-V` §7 | **GATED, correctly** — the gate was a triple at ≥6/15 and the best triple read 3/15 | — | None. §8.4 exhausted the axis independently. |

---

## 2. SHORTLIST 1 — THE CENSUS-KILLED LIST (rule-violation repairs; build these first)

Nine classes were closed without a child. Ranked by *how badly the estate's own laws
undermine the kill* × cheapness.

| rank | item | what killed it | why the kill does not hold | cost to repair |
|---|---|---|---|---|
| **1** | **C1 — vocabulary-axis row TYING** | a nearest-neighbour cosine of 0.2977 on tail rows | The comparison was **tie vs verbatim**. The mechanism that actually won the campaign is **deletion**, which is strictly worse than a 0.30-cosine stand-in, and the axis went on to set the floor. The kill statistic never priced the alternative that shipped. | one CPU census + one gather-class build (~40 s) + 1 review |
| **2** | **S3 — ORTHO-PAIR soft arm** | a pooled ρ scalar (wave 24) that *inverted* the premise | **The slate's own binding FIX 1 says the census NEVER gates and the soft arm is built regardless.** This is a documented rule violation, not an interpretation. And L3's cross term — the ~3-case/component tax — has still never been *controlled*. | 1 build + 1 review, zero storage delta |
| **3** | **S7 — NKP's buried twin: MONARCH** | NKP's own P3 census | The census **buried the loser and never built the winner.** Monarch beat NKP at 42/42 and beat rank truncation on the same surface, and `PHASE1` §6 still lists it as never priced behaviourally. | 1 cell at the census-best site + 1 review; the census surface already exists |
| **4** | **A5 — F-B (GQA value-channel CUR)** | nothing — **it passed its gate** and was then not built | Listed here rather than under never-reached because its measurement *cleared the pre-registered bar* ("the class does not close causally") and the estate behaved as though it had not. C07/C08 have zero alive 4x coverage in the entire ledger. | 1 research-probe build + 1 review; needs the §4.5.3 materialization ruling |
| **5** | **A3 — F-C (conv coreset at 8x)** | coreset losing to low-rank on the conv write at 12/15 site-levels | A reconstruction head-to-head, on the axis this estate has sealed five times as non-predictive. Conv is 20.7% of the body and the only structural mass never folded on its own axis. The S4 sibling cell that would attribute a death to level-vs-mechanism was also never built. | 1 research-probe build + 1 review |
| **6** | **A1-D — VC-FOLD arm D** (α-combined up rows) | a 0.005–0.013 cosine spread across selectors | Same class of statistic. Arm D is the *only* arm that is not a subset selection, so "verbatim beats synthesis" was never actually tested inside the body — it was tested between three subset selectors. | 1 build + 1 review |
| **7** | **A9 — HODLR-V** | diagonal coverage clearing at one site "for the wrong reason" | The kill is honest, but it rests on the same L05 massive-unit artifact that simultaneously faked S+L's kurtosis and arm E's OFF certificate, and **no site was re-screened after the artifact was understood**. | one re-screen (CPU) then 1 cell + 1 review |
| **8** | **A1-A / A1-B — the alternative selector arms as children** | the same 0.005–0.013 spread | Weakest of the nine; the cliff argument is quantitatively strong even if it is the wrong currency. Arm C shipped, so the axis is not unexplored — only unvaried. | 1 build + 1 review |
| **9** | **#5-succ(a) — sink/massive carve-out** | direct arithmetic (≤0.8% of one direction) | **This kill is sound and is listed only to be dismissed.** Its *successor* — the write-side repair implied by the same measurement — is a never-reached item, not a census kill. | — |

**The three worst build-don't-predict violations, named:**

1. **C1, vocabulary row tying.** Killed on a cosine, on the axis that turned out to be the
   campaign's best, at the mass (29.2% of the model) where the floor is decided, against a
   comparison that was never run.
2. **S3, ORTHO-PAIR.** The only kill on this list that violates an *explicit written
   instruction in its own spec* — "the census NEVER gates" — and the census gated it.
3. **S7/NKP.** A one-CPU-hour reconstruction census buried a class *and its own winning
   control*, using precisely the currency `wave14/NOTES-MX-R.md:264-269` proves cannot rank.

**Structural finding.** The extension bank's binding rule — a kill on a 1-case margin
requires extension-bank concordance — was funded at wave 29. **Every kill on this shortlist
predates it and none was subjected to it.** These nine were not merely predicted; they were
predicted with an instrument the estate has since agreed is insufficient for kills.

---

## 3. SHORTLIST 2 — UNDERSAMPLED SINGLES worth one more point each

Built exactly once, at one site or one rung, and never sampled again.

| rank | item | the single point | the one more point |
|---|---|---|---|
| **1** | **A8 — F-LANDMARK** | wave 20, L12, 6 of 8 KV groups. Reads near-parent *including correct context retrieval*; β recovered `head_dim^-0.5` unprompted. **No benchmark, ever** — custom runtime | a loader that accepts the custom module → 1 benchmark; or the same fold at a second site. The class's funding decision has been open on a missing measurement for **eighteen waves** |
| **2** | **M-7 — the F-DECAY arm of a designed head-to-head** | one arm built (F-LANDMARK), one never built | 1 F-DECAY cell at L12 after the bounded recapture. Until then the attention estate is neither funded nor closed |
| **3** | **CG1-b — the `CL640`/`CL896` bracket** | two points, 256 channels apart, spanning **0/15 → 4/15** | one `--ff 768` body (~451 s GPU) + 1 review. The estate brackets its vocabulary transitions to 1,024 rows and has left a **4-case** mlp transition bracketed at 256 channels |
| **4** | **#8-A — metric selection at 896** | built once, at 1792 (`CL1792M`, 8/15, best body at its ratio) | the identical one-line change on `CL896`, the body under every child at and below 0.4517x. 1 build + 1 review |
| **5** | **The floor's own bracket** (`V9CL896X09` 0 / `V10CL896X09` 1) | two points, 1,024 rows apart | one 40-second CPU pass to halve it. `DEEP_RECORD` §5.2: "nothing has been built inside either" |
| **6** | **§8.2's λ1.0 rungs on `CL896X09`** at K = 14,336 and 16,384 | the rung account is fitted on four cells and makes a falsifiable prediction at each | ~100 s GPU each |
| **7** | **§8.1's λ0.3 arm at K = 8,192** | the "lands at THREE" law is fitted on six rungs; this is the only untried one, and its control is the 0/15 vocabulary death | ~100 s GPU |
| **8** | **A6 — CBT** | one site (L12), research probe, near-parent review that its own receipt calls below the instrument's resolution; lost its L02 head-to-head | its greedy merge schedule (the only adaptive part) at the 8x probe. Low expected value — listed for completeness |
| **9** | **§8.3's *n*** | thirteen ladders, stuck at thirteen for two waves | one wave that spends **two** λ solves instead of one |

---

## 4. SHORTLIST 3 — NEVER-REACHED classes, ranked by floor-relevance × cost

| rank | item | floor mechanism | cost | the law that gives it hope, or dooms it |
|---|---|---|---|---|
| **1** | **PG1 — residual-corrected sequential solves** | attacks the ~3-case/component additive tax directly, per-site, rung by rung | one matrix product on existing caches + a verbatim re-run of a sealed ridge + 1 review. **Cheaper than the KKT that was built** | **Hope:** wave 29's seal is explicitly narrow — the KKT null closed the frozen-gate terminal branch and named this family as staying open (`NOTES-MX-AE.md:334-343`). **Doom:** M1's autopsy showed 5/6 local objectives improving while composed drift got worse; this family is still local, just with moved targets |
| **2** | **C2 + #25 — score-bilinear joint q/k with rotary-pair preservation** | exact 4x per head pair, zero aux, six layers; opens C06, which has zero coverage in the estate | one two-sided LRA per RoPE block (CPU, exact, no capture) + 1 build + 1 review | **Hope:** the best spectrum in the entire wave-14 battery (0.9948/0.9985/0.9926 median at 16 of 64 dims) and the ALS locality failure provably does not apply. **Doom:** it is the Δ=0 member of the RoPE family and contracting `head_dim` remaps the frequencies — which is exactly what #25 is for |
| **3** | **CG1-b — a `CL768` (or `CL736`) body, then the vocabulary ladder on it** | puts the floor's whole ladder on **mlp × vocabulary (1.17 cases)** instead of **depth × vocabulary (4.44)** — wave 30's own fitted interference. `CL768`+K=9,216 = 0.413383x with **no excision**; `CL768`+X09+K=10,240 = **0.387682x**, below the deepest child ever built | 1 body build (~451 s GPU) + 1 review, then 40-second CPU rungs | **Hope:** the allocator law (`NOTES-MX-AF` §3–§4) — "keep every axis out of its own collapse and the composition tax collapses with it"; V19 on `CL1792` cost **zero** cases. **Doom:** `CL640` reads 0/15, the mlp axis "flattens onto the sealed 896 reading", and §8.5's decay law says a body already below two cases sheds lanes as K falls |
| **4** | **#10 — generation-manifold capture** (parent free-run states as evidence) | attacks the artifact population that is **43.6% of every sealed miss** and 61% of wave 38's lanes | 1 pure-inference capture wave + 1 rebuild + 1 control + 2 reviews | **Hope:** the entire deep-zone death taxonomy is autoregressive and teacher-forced evidence weights those states at zero; decode-time repair is sealed dead (20 settings, 300 lanes, 0 recoveries). **Doom:** every evidence-side change on record has priced ≤1 case |
| **5** | **PG2 + #18 — attention mass: KV-head grouping ladder, and Procrustes K→V tying** | the unmeasured third term of §5.6's 0.25x equation; ~52M of body | 1 KV-grouping rung on a V19 base + 1 review; 1 Procrustes solve per layer + 1 build + 1 review | **Hope:** `num_key_value_heads` is a stock config int and `X07KV6JQC` reads 7/15 at 0.927x on a KV recipe never laddered toward depth. **Doom:** named ceilings already close `head_dim` and sub-0.763x interface heads |
| **6** | **#13 — unembedding rank-repair (demotion correction)** | a closed-form diagonal on the tied table, at the axis and rung that *set* the floor | cache parent+child logits, one weighted LS; ~2·\|V\| params; 1 review | **Hope:** the floor is a vocabulary rung and the K=9,216 death is an emission failure, which is the exact pathology this repairs. **Doom:** the context-dependent component is unrecoverable statically |
| **7** | **#12 V-ROUTE + #20 clustered LM head + C1 tying** — the three non-deletion vocabulary mechanisms | all three put content into rows the coreset deletes, on 29.2% of the model | 1–2 builds + reviews each | **Hope:** the coverage law is now *measured*, not asserted — the review cliff sits exactly where emission token coverage crosses **0.81** (`NOTES-MX-AF` §2). Every one of these three raises coverage per stored row. **Doom:** the SVD embedding is a sealed 0/15 and the tail is flat (participation ratio 0.979) |
| **8** | **#14 — anti-loop cauterization** | the only registered mechanism that **removes parameters** and targets the artifact population | difference-of-means from 846 archived artifact lanes; zero the implicated rows; 1 review | **Hope:** waves 34–35 proved the artifact lanes are not a decoding bug, so the mechanism must be in the weights. **Doom:** "there is nothing behind the loop" on this population — the penalty diagnostic converted 37 artifact lanes into 37 readable-and-empty ones |
| **9** | **Monarch** (see shortlist 1 rank 3) | the closest structural relative of the one thing known to fold here | 1 cell + 1 review | **Hope:** it beat both comparators at 42/42. **Doom:** both spectra on this parent are flat |
| **10** | **A4 — S+L** | designed to survive *at* the graves where every rank truncator died | 1 cell vs its sealed twin + 1 review | **Hope:** its own screen said build — heavy tails everywhere but one component. **Doom:** the wave-14 cliff law, if it is ever asked to run at many sites at once |
| **11** | **CG2 — REBASE-AFTER-EXCISION** | the only lever that reduces the **demand** side of L2's wall | 1 conditioned capture + 1 build + 1 review | **Hope:** all six condemned basis objectives re-weighted the *same* parent capture; basis-invariance is untested under a changed reader census. **Doom:** basis-invariance has survived six objectives and two widths, so a tie is the honest expectation — and a tie is still a law |
| **12** | **#4 LACE shrinkage · #23 OBS vs ridge · #24 nested-budget joint ridge** — three untried *repair families* | the estate has used exactly one repair family (prior-anchored ridge) for 39 waves | a one-line solve change each; 1 A/B build + 1 review each | **Hope:** cheapest untried changes to the solve itself, all aimed at the interference tax. **Doom:** #4 has been ordered three times and skipped three times on the grounds that other controls closed its lane first |
| **13** | **B4 W-ABSORB-SB · C3b portfolio corrector · #10-field diagonal interface hygiene** — three zero-or-near-zero-parameter repairs | infinite drift/param ratio by construction | the free surface enumeration; then 1 solve + 1 review each | **Hope:** a measured 28% same-block linear share. **Doom:** the trajectory-objective family these resemble ran twice and produced `CT640` (total collapse) and `X05T` (0/15) |
| **14** | **S2 SEGBASIS · S6 29-DIAGONAL · S8 EARLY-SPARED** — the three unbuilt round-2 width waves | zero added tensors; zero param delta; localization | 1–2 builds + reviews each | **Doom, mostly:** wave 30 measured width as the *worse* axis at matched ratio (7/15 vs 4/15), and six basis objectives have lost. S6 is the exception worth keeping: it exchanges zero directions and the **28 block gains have never been re-solved at all** |
| **15** | **#28 — fold a nested parent (Gemma-4 E4B)** | cannot move this floor; decides whether the floor is a property of the *method* or of *LFM2* | run the existing pipeline on a different parent | The synthesis calls it "the single most informative strategic experiment on this list." It is a *phase-2 attribution* question, ranked last here only because the brief is this parent's floor |
| **16** | remaining field items **#1, #5, #7, #8, #9, #11, #15, #16, #17, #19, #21, #22, #26, #27, #29–#33** and plan items **#6, #9, #11, #14, #15, #16-nonuniform**, plus **A2-B, A7, A10, A11, A12, A13, B1, B2, B3, B5, C3b, C4, S4, S5, weight-space CUR, tensor-train, CBT merge schedule** | assorted | 1 cell each | Individually low-to-medium; listed in §1 with their own gates and doom clauses |

---

## 5. THE COUNT

Counting **items that have never had a child built and frozen-reviewed**, per registry.
Mechanisms and cheap rungs are counted separately because they are not the same kind of
thing.

**Mechanisms / classes:**

| registry | items in registry | never behaviourally tested | which |
|---|---:|---:|---|
| Round-1 Track A (A1–A13 + 5 VC-FOLD arms + A2 arm B) | 19 | **13** | A1-A, A1-B, A1-D, A2-B, A3, A4, A5, A7, A9, A10, A11, A12, A13 |
| Round-1 Track B | 5 | **5** | B1–B5 |
| Round-1 critique gaps (C1–C5 + C-x) | 7 | **4** | C1, C2, C3b, C4 |
| Round-2 slate (S1–S8) | 8 | **7** | S2, S3, S4-pilot, S5, S6, S7, S8 |
| Round-2 critique gaps (CG1, CG1-b, CG2, CG3) | 4 | **2** | CG1-b, CG2 |
| Retention plan #1–#16 + PLAN_GAPS 1–4 | 20 | **13** | #6, #8-A, #8-B, #9, #10, #11, #12, #13, #14, #15, #16-nonuniform, PG1, PG2 |
| Field synthesis #1–#33 | 33 | **30** | all but #3, #6, #7-partial |
| `PHASE1` §6 never-priced structural + M-7 | 5 | **3 new** | weight-space CUR, Monarch, tensor-train (CBT-merge ⊂ A6, M-7's arm = A7) |
| | | **77 gross** | |
| less one merged pair: **C2 ≡ field #25** (built as one cell) | | **−1** | |
| | | **= 76 distinct mechanisms** | |

**Cheap named rungs (§1.9): 12** — the two 0/15 brackets, λ1.0 at K=14,336 and 16,384,
λ0.3 at K=8,192, `V9CL896X13`, §8.3's *n*, the 736→720 rung, max-min at 720,
`evidence-pca` at 768, `--arm stale`, and the 768→640 midpoint. (The excision quad is
excluded — it is correctly gated out at 3/15.)

### **TOTAL NEVER BEHAVIOURALLY TESTED: 88 — 76 mechanisms and 12 rungs.**

Of the 76 mechanisms:

* **9 are CENSUS-KILLED** — closed by a census, spectrum, cosine or arithmetic, with no
  child and no review. These are the build-don't-predict violations. **1 of the 9 (the
  sink carve-out, field #2a) survives scrutiny; 8 do not.**
* **5 are GATED**, of which **3 gates still bind** (A7's missing position index; A12's
  census that never failed; A13's ruling that was never given) and **2 are dormant by
  design**.
* **62 are NEVER-REACHED** — registered, uncontradicted, unfunded.
* **2 are BUILT-ONCE and undersampled** (A6 CBT, A8 F-LANDMARK) and are counted in
  shortlist 2 rather than here, since both did get a reading.
* **1 is RULE-REFUSED** (C5, input-conditional folds) and is correctly excluded from the
  76 above.

The registry columns overlap in *territory* without overlapping in *mechanism* in three
more places, and they are kept separate deliberately: A5 (F-B), C2/#25 (score-bilinear
q/k), PG2 (KV-head grouping), #18 (Procrustes K→V) and #19 (attention removal + FFN
fusion) are **five distinct closed-form attacks on the same unopened attention mass**, and
C1 (row tying), #12 (V-ROUTE), #13 (unembedding rank-repair) and #20 (clustered LM head)
are **four distinct attacks on the same tied table**. That two of the model's three largest
masses each have four-to-five unbuilt mechanisms aimed at them is the shape of the finding.

**Legitimately closed by a built child, and kept off every shortlist:** VC-FOLD arm E
(the object does not exist), F-EXCISE (built to exhaustion, §8.4), #2 RETOK-BODY (the
stream shift does not exist), #3 CHASM BRIDGE (void arithmetic + dead children), #4/C3a
GLOBAL-KKT frozen-gate branch (null with a measured mechanism), #5 MARGIN-RESTORE (the
premise is false on the target children), #7 COMPLEMENT-MERGE (zero-dimensional, and it
still moved a rung), field #2a (arithmetic is decisive), field #6's writer branch (the
metric cancels — this is algebra, not a census), and the excision quad (gated out by a
pre-registered rule at 3/15).

**What this means for the floor claim.** 0.4116x is the floor of a search that spent
essentially all of its budget on three axes — vocabulary, `intermediate_size` and block
excision — and closed nine others without building anything. **Two of the three axes it did
spend on were themselves nearly closed by censuses that the builds later falsified:** the
excision scan cleared only L05 (which reads 6/15) and rejected blocks 13 and 9 (which
became the entire deep family), and on the vocabulary axis the tying variant died on a
cosine while the coreset variant went on to set the floor. The honest statement is not
"fold-only bottoms out at 0.4116x"; it is **"0.4116x is where three axes bottom out, and
the estate has 88 unpriced items — 8 of them killed by exactly the kind of number this
campaign has five sealed laws against, and 12 of them one CPU pass or one GPU minute
away."**

---

## NOTES — audit, wave 39 in flight

**What this is.** An inventory, not a wave. No child was built, no solve was run, no
measurement was taken, no ledger row was written, no sealed artifact was read for anything
but citation. The single deliverable is this file.

**Sources swept, in full:** `docs/FOLD_CLASS_CANDIDATES_2026-08-12.md`,
`docs/FOLD_CLASS_CRITIQUE_2026-08-12.md`, `docs/WAVE23_ROUND2_SLATE_2026-08-13.md`,
`docs/WAVE23_ROUND2_COMPLETENESS_CRITIQUE_2026-08-13.md`,
`docs/FIELD_RESEARCH_SYNTHESIS_2026-08-28.md`,
`docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md` §6 and §8,
`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` ADDENDA 11.3, 12, 15.4, 16.6,
`docs/INTELLIGENCE_RETENTION_PLAN_2026-08-14.md`, `docs/PLAN_GAPS_2026-08-14.md`,
`reports/matrix-campaign-20260810/DEEP_RECORD.md` §5 and §8,
`reports/matrix-campaign-20260810/ACTUAL_MATRIX_2026-08-28.md` §3 and §3a, the whole
`NOTES-MX-*` series (41 files, 18,442 lines) with every "what the wave did not spend" /
"what was ordered and not run" / "what was not built" section read, and
`reports/matrix-campaign-20260810/ledger.jsonl` (988 rows) queried directly for the
technique vocabulary and the `CL`-family cell list.

**One machine check was run rather than asserted.** The ledger's entire technique
vocabulary is `architecture` (527), `composition` (154), `T1` (116), `T5` (62), `T3` (45),
`T8` (28), `T2` (26), `T4` (14), `T7` (10), `T9` (4), `copy_exact` (2). **Not one of the
round-1 or round-2 class names appears as a technique id.** The classes that were built
were built *under* `architecture` (`CL*`, `V*`, `X*`, `W*`, `G*`) or as research probes
outside the `ARCH` id space entirely (ADDENDUM 12: custom-module children take no `ARCH`
id), which is why `PROBE.CBT-L12` and the F-LANDMARK probe have no ledger row. The
`CL`-family query returned rungs at 640, 896, 1280, 1408, 1536, 1664 and 1792 — **no 768,
no 832, no 960, no 1024, no 2048** — which is how §1.5 CG1-b and §1.6 #8-B were
established as unbuilt rather than assumed.

**Two things this audit deliberately did not do.**

1. **It did not re-litigate a kill that a built child made.** Everything in the
   BUILD-KILL column is left closed and is named in §5 so the shortlists cannot be read as
   claiming otherwise. `ARCH.KKTD` and `ARCH.KKTS` are real children with real 0/15
   readings and wave 29's bound is narrower and better than "closed-form does not repair
   composition"; §1.6 says so.
2. **It did not rank by expected case-gain.** Every "floor-relevance" cell names the law
   that gives the item hope *and* the law that dooms it, because six of the eight
   shortlist-1 items were originally killed by exactly the kind of single-number
   confidence this audit would reproduce if it scored them.

**What I could not verify.**

* Whether the wave-14 selector census (`census-vcfold_refit.json`) is still on disk. Its
  numbers are quoted from `wave15/NOTES-MX-S.md:279-285`, not read from the artifact — the
  wave-14 artifacts live on the share, which this audit did not touch.
* Retention plan **#6 Stage 0** (the per-case ridge over 232 children × 15 cases). Wave 29
  ran a narrower LIST→SET coordinate attribution; whether the full leave-child-out artifact
  was ever produced is not stated in any note I read. It is recorded as never-reached *as
  specified*, which is the claim the record supports.
* The exact GPU cost of an `F-B` or `score-bilinear q/k` cell. Neither has ever been
  costed by the estate, so §1 quotes their *solve* cost (CPU, exact, no capture, both
  specified as capture-free) and states the build+review as the real expense.
* Whether the `A1` selector spread of 0.005–0.013 cosine was measured at `keep=640` only
  or across levels. The note says keep=640 at every site.

**Deliberately left out.** Stage-2 (`/mnt/ai_data/neural-forge-stage2/`) and every
quantization/codebook item — the first is outside the fold-only contract by design, the
second is refused by Cyril's rule. The `chimera-omni` notes in the field synthesis §2 are
architecture guidance for a from-scratch model and are not fold classes. Wave 39's two X13
rungs (`V14CL896X13`, `V10CL896X13`) are named in §1.9 as *in flight*, not as unbuilt, and
`V9CL896X13` is listed as unbuilt because wave 39's own pre-registration states it is not
being built this wave.

**Estate.** One file added, `docs/UNBUILT_INVENTORY_2026-08-28.md`, staged by explicit
path. Nothing else changed: no builder, no solver, no sealed receipt, no dossier, no
ledger row, no `RESULTS_MATRIX`/`ACTUAL_MATRIX`/`DEEP_RECORD`/`CAMPAIGN_SCORECARD` edit,
no backup needed because nothing was overwritten. **The Codex lane's uncommitted paths
were not touched, and neither was anything under `wave39/` or `run/mx39/`.** Nothing
pushed, nothing rebased.
