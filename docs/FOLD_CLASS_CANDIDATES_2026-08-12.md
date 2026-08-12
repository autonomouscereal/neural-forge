# Wave-14+ Fold-Class Specification — Merged Lens Synthesis
**Neural Forge matrix campaign · dispatch input and design doc · synthesized 2026-08-12**
**Model: LFM2 229,693,184 params · frontier 0.703x/0.729x · phase-1 target coherent ≤0.5x · gap ≈ 8 SwiGLU-lane-equivalents of drift room**

All classes below are closed-form solves on sealed evidence (4,751 fit / 1,584 heldout rows), zero gradients, deterministic, ≥4x parameter reduction on the replaced component with ALL auxiliary tensors counted in full, no quantization. Drift is a prioritization heuristic only (review ≈ 12.96 − 13.16·drift, R²=0.62); every cell is built, composed on the frontier base, and graded by frozen behavioral review.

---

## 1. Dedup and merge decisions

| Merge | Inputs | Result |
|---|---|---|
| **M-1: VC-FOLD** | GLU-CUR (nonlinearity lens, KEEP) + F-A (data-selection lens, KEEP) + F-MERGE (merging lens, fixed) + Regime-4a OFF-drop (nonlinearity lens, fixed) | One class: verbatim SwiGLU channel coreset on the nonlinear feature map, trio-indivisible, with four registered selector arms and a free-drop arm. GLU-CUR and F-A are the same mechanism (select real intermediate channels by evidence-weighted post-nonlinearity mass, keep w1/w3 verbatim, prior-anchored ridge on w2 against sealed write rows). F-MERGE's reviewer proved its scalars/signs are absorbed by the w2 re-ridge, collapsing it to feature-space CUR — its genuinely new content (per-cluster alpha-combined up rows, signed-triplet legality enumeration, coherence-spectrum forecast) survives as arm D. Regime-4a rides the same census at the same sites. **Occupies the wave-14 CUR/interpolative queue slot as its registered refinement — one budget slot, not five.** |
| **M-2: F-D** | F-D (data-selection lens, fixed) | Survives only as the gated tail-extension arm of VC-FOLD at sites the top-640 census disqualifies but top-~900 clears. Knob replaced by the storage-feasibility identity; CSR offsets deleted. |
| **M-3: G-artifact** | Ω-RIDGE + ANTIFOLD + XGRAM (composition lens, all fixed/kept) | Three distinct classes sharing ONE precondition artifact: the per-layer and per-endpoint terminal-transfer surrogates G, built once from sealed captures, published with heldout R² and spectra, serving as the common kill switch. Built before any of the three spends a build. |
| **M-4: Gate-Jacobian** | Gate-Jacobian pair factorization (nonlinearity lens, fixed) | Demoted to census-gated hedge strictly behind VC-FOLD; runs only where VC-FOLD's gated-energy census FAILS. Not independently ranked. |
| **M-5: Residual gauge** | Free residual gauge (nonlinearity lens, fixed) | Demoted from fold class to preconditioner infrastructure attached to one wave-14 structured lane (see §5). Claimed benefit to VC-FOLD/regime census struck — false by gauge invariance of Phi. |
| **M-6: Energy-concentration pair** | S+L + HODLR-V (structured lens) | Both test the same hypothesis (evidence-weighted energy concentration in body matrices). Kept as separate classes but hard-sequenced: S+L first (cheaper plumbing); HODLR-V builds only where its free Gram coverage screen clears a pre-registered floor AND S+L's result does not already answer the question at that site. |
| **M-7: Attention tails** | F-DECAY + F-LANDMARK (merging lens, both fixed) | Alternative tails for the same slot. One head-to-head cell each at L12; the attention estate funds only the winner across remaining sites. |

No proposal re-proposed a sealed class (T1–T9, charts, ALS, sequential conditioning, head-DROP, etc.); zero duplicate kills fired. All reviewer fixes are incorporated into the spec blocks below as binding registration conditions.

---

## 2. Ranked survivors

**Track A — fold classes** (claim parameter reduction directly), ranked by (a) plausibility of a coherent 4x on LFM2, (b) composition story vs the drift wall, (c) scaling transfer:

| Rank | Class | (a) 4x plausibility | (b) composition | (c) scaling transfer |
|---|---|---|---|---|
| A1 | VC-FOLD (verbatim SwiGLU coreset) | Exact 0.2500x, zero aux, error class proven survivable (C13 post-gate recombination) | Strongest: trio-per-block = lane-concentration law made structural; only class with surface to close phase 1 alone (8 sites = 47.2M) | Any GLU FFN at any scale; redundancy grows with width |
| A2 | F-EXCISE (block excision + parent-write splice) | Infinite component ratio, zero added params | Best drift arithmetic: 1 solved component buys ~2 lane-equivalents; 3 excisions + frontier arithmetically cover 58.5M | Depth redundancy grows with model depth; universal residual architectures |
| A3 | F-C (conv-block channel coreset @8x) | Exact 0.1250x, zero aux, structurally exact sub-block | Built AT the two measured conv laws (deeper-is-safer, S8>S4); write-side-only additive | Channel-separable blocks (conv/SSM hybrids); moderate |
| A4 | S+L (evidence-scaled sparse + low-rank split) | Exact 4.0000x verified on C08/C13 with ids counted | Single-lane; verbatim-outlier logic from the one law that revived a dead class; cheapest new mechanism | Heavy-tailed mixers are universal in transformers; strong |
| A5 | F-B (GQA value-channel CUR, v+o pair) | Exact 0.25x on the coupled unit, zero aux | Write-side additive, attention pattern bit-identical; unlocks a zero-utilization lane | GQA is the dominant modern attention layout; strong if it lives |
| A6 | CBT (block-Toeplitz tied ridge, pinned) | Honest 4.23–4.51x on grave components | New failure axis (full-rank, no subspace annihilation) at components where every truncator died; pure new coverage | Prior is weak everywhere equally; methodology transfers |
| A7 | F-DECAY (mixer → verbatim heads + decay-bank tail) | Exact 4x per mixer, 6 sites, 14.2M | Monetizes coverage currently priced infinite; no block-exclusivity conflicts | SSM-tail compilation transfers to any softmax attention |
| A8 | F-LANDMARK (verbatim k/v landmark states) | Exact 4x per converted KV group, smallest purchase (3.5M) | Highest information per GPU-hour: binary verdict on whether the verbatim-rows law generalizes past the interface | Law-localization result transfers regardless of outcome |
| A9 | HODLR-V (hierarchical verbatim-diagonal blocks) | 4.0x expressible but off-diagonal ranks thin (r~9–13) | Lives entirely on a coverage number; redundant with S+L's hypothesis at higher plumbing risk | Moderate; biclustered structure is model-specific |
| A10–A13 | Gated arms, buildable only behind their gates: **F-D** (tail extension), **Regime-4b** (secant/Jennrich CP), **Gate-Jacobian** (hedge), **FSS** (double-conditional) | see spec blocks | — | — |

**Track B — zero-storage composition/solve-time classes** (install nothing or modify solves at constant storage):

| Rank | Class | Value | Gate |
|---|---|---|---|
| B1 | XGRAM (pairwise-interference portfolio) | Prices lane PAIRS the greedy shopper is structurally blind to; failure confirms additivity as physics | Retrospective leave-one-out beat of scalar R²=0.62 on ~15 sealed children before any build |
| B2 | Ω-RIDGE (terminal-transfer metric wrapper) | Steers each lane's unavoidable error into terminal-attenuated directions at constant storage | Step-0 G spectra + heldout R²>0.5 per layer/endpoint |
| B3 | ANTIFOLD (drift-conjugate sequential folding) | The only attack on the additivity constant itself; 25% cancellation → gap shrinks 8→6 lanes | Contingent on Ω step-0; β=0 fallback |
| B4 | W-ABSORB-SB (same-block absorber refit, post-fix) | Zero-storage purchase of the sealed 28% same-block linear share (M2's mechanism with an infinite drift/param ratio) | Addressable-surface enumeration on MXP2/MXO3 first; kill if empty |
| B5 | Residual gauge (preconditioner) | May lower drift-per-param for whole axis-sensitive families at once | Sigma_hat spectrum census; then ONE paired cell on one wave-14 structured lane |

**Execution sequencing principle:** every class above has a free or near-free pre-build measurement. Wave 14 opens by running ALL of them (§4.3) before any child is built; the measurements re-rank within tiers and kill dead classes at ~zero cost.

---

## 3. Survivor spec blocks

### A1. VC-FOLD — verbatim SwiGLU channel coreset (nonlinear-feature CUR, trio-indivisible)
*Merged from GLU-CUR + F-A + F-MERGE(fixed) + Regime-4a. Occupies the wave-14 CUR/ID queue slot.*

**Math.** The 2560 intermediate channels are the dictionary. Feature matrix on sealed evidence: Phi ∈ R^{n×2560}, Phi[t,i] = silu(w1_i·x_t)·(w3_i·x_t), gate evaluated EXACTLY. Select r=640 real channels; keep W1[:,S], W3[:,S] verbatim (survivor pre-activations and gate products bit-faithful to parent); push all error into one closed-form writer solve: W2' = argmin ||Phi_S W2'^T − Y||_F² + λ_rel||W2' − W2[:,S]||² with Y = parent's sealed MLP write rows (the M6 objective family — the campaign's one composition winner), λ relative to the spectrum (R1 lesson). Optional sign canonicalization first (exact, function-preserving): flip (w3_i, w2[:,i]) pairs freely, never w1_i (SiLU is not odd) — the untried signed-triplet legality item, executed for the first time.

**Registered selector arms** (ablation at one site before scaling; all fp64 CPU, pinned deterministic ties by channel id):
- **Arm A** — D²-mass: s_i = E_{D²}[|Phi_{:,i}|]·||W2[i,:]||₂ (operator-product mass; the statistic family that predicted life at the token interface).
- **Arm B** — ridge leverage: diag(Phi^T Phi (Phi^T Phi + λI)^{-1}).
- **Arm C** — D-weighted pivoted QR on Phi (double-MGS per A9, deterministic ties).
- **Arm D** (from F-MERGE, fixed form) — correlation-cluster medoids: complete-linkage agglomeration on 1−corr(Z), cluster cap 8, count 640; gate row = medoid verbatim; up row = per-cluster closed-form least-squares alpha combination w3' = Σ_j α_j w3_j (the only arm whose child features leave the parent channel set; alphas fold into w3', zero aux). The decorative mass-conservation scalar is deleted per verdict.
- **Arm E** (from Regime-4a) — certified-OFF drop: units whose gated-output energy share < ε on BOTH fit and heldout (ε pre-registered) are dropped free; verbatim fill via arms A–C over survivors. Manufactures drift allowance rather than spending it.

**Solve procedure.** (1) Load sealed captures at the MLP input boundary + parent W1/W3/W2 + sealed write rows Y. (2) fp64 CPU: compute Phi (pure replay of parent linears + pointwise silu on cached rows; no model forwards). (3) **Pre-registered census, runs FIRST across all 14 MLP layers, pure CPU:** top-640 mass share; per-row coverage quantiles (fraction of rows with top-640 coverage ≥ 0.8); interpolative residual ||Phi − Phi_S T||_F/||Phi||_F; intra-cluster coherence spectrum (arm-D price forecast); OFF/quasi-linear regime occupancy (arms E and Regime-4b). Alive floor calibrated by analogy to the embedding's measured ~81% occurrence-coverage floor, pre-registered before any numbers are seen. Sites below the bar are not built. (4) Select S per arm. (5) Prior-anchored D² ridge for W2'; receipt publishes update_scale = ||W2'−W2[:,S]||/||W2[:,S]||, prior norm, condition number; **install REFUSED in the kernel (not the writeup) if update_scale > 1.0** (THE LAW as a fail-closed guard). (6) Optional 2-pass B/W2' refinement: exactly linear in B given fixed gates; **separately-identified arm only**, ≤2 passes, prior-clamped, pre-registered stop; never folded into the base result. (7) Materialize a plain narrower SwiGLU (intermediate_size 640) directly to state_dict — no hooks, no runtime indices (unit identity is not needed at inference).

**Storage.** Parent trio 3·2560·1024 = 7,864,320. Child: 2·(640·1024) + 1024·640 = **1,966,080 = 0.2500x exact, zero auxiliary tensors** (the intermediate axis is block-private; the selection permutation vanishes into the layout). S8 probe: r=320 = 0.1250x. Per lane saved: 5,898,240. Eight surviving sites = 47.2M ≈ the entire phase-1 gap.

**Targets.** C11+C12+C13 as one indivisible unit per site. Census re-ranks all 14 MLP layers; priors L4–L6 first (L05 measured 12/15), never L0 first. MLP = 67.7% of body.

**Solo-cell test plan.** Cell 1: census-best site, arm A vs arm C head-to-head (identical storage, one variable), composed on the frontier base, frozen V5 review; drift priced against the actual install base with post-acceptance re-pricing. Cell 2 (same wave): arm D and arm E at one site each. The three-selector ablation and the update_scale guard are registered before the first receipt. Rare-capability displacement is what the selector ablation tests.

**Risks.** Flat SwiGLU mass at some/all sites (census kills free). Rare-channel displacement (ablation covers). L05 tolerance was measured under basis techniques; the verbatim regime may reshuffle site safety — hence census re-ranking, not the old site table. Refinement arm flirts with THE LAW — quarantined as specified.

**Scaling transfer.** Direct: any gated-FFN architecture; channel redundancy is expected to grow with width, and the census transfers as-is.

---

### A2. F-EXCISE — block excision with parent-write splice (depth folding)
*Kept as reviewed; verdict flags carried into registration.*

**Math.** Delete an entire decoder block; repair with ONE closed-form re-solve. Residual structure gives block l = identity + delta_l. The upstream block's writer (mlp_down w2) is re-solved by production ridge so its target becomes the SUM of the parent's sealed writes for both blocks (delta_{l−1} + delta_l). The linearly-predictable-from-upstream-features part of delta_l is absorbed at ZERO added parameters; the unpredictable remainder is the drift price, measured on heldout before any build.

**Solve procedure.** (1) **Pre-buy scan (≈14 ridge solves, ~1 hour):** for each block l ∈ 1..13, A = upstream realized pre-writer features silu(G)⊙U (N×2560) from sealed captures; Y = delta_{l−1} + delta_l from the collector's structural identities (layer_output == post_operator + mlp.down); solve with `_fit_linear`; record heldout unexplained fraction. This is the excisability ranking and the free kill. (2) Excise argmin block — conv-first expectation (attention writes are context-dependent across positions; a per-token ridge absorbs only the position-local component; the scan will show it). Never L0. (3) Install: delete block tensors; overwrite upstream w2 in place. Write-norm-ratio ceiling 2.0 binds by construction (target = sum of two parent writes); A12 collateral guard live. (4) Arm B (separate cell, priced as a second solved component at the standard −3, never smuggled free): re-solve the following block's readers against the parent's normed post-block-l stream. (5) **Before any second excision: RECAPTURE through the accepted 13-layer child** — parent captures are stale for the new base; re-pricing alone is insufficient (verdict flag made binding). Compounding is measured, never inferred.

**Storage.** Removed per purchase: deep conv block = 3,145,728 + 3,072 + 1,048,576 + 7,864,320 + 2,048 = **12,063,744** (5.25% of model). Attention block ≈ 11,010,048 + 2,048 block norms + per-head q/k norm gains (ledger corrected at registration by counting from the checkpoint — the sealed figure undercounts removal). Menu grain: FFN-only 7,864,320; mixer-only 3.15–4.2M. Added: **ZERO** (in-place same-shape refit). Ratio infinite; no aux tensors can exist.

**Targets.** Whole blocks as indivisible units: deep conv region first (L11/L13 neighborhood — deeper-is-safer is a conv-only law), deep attention second (L10/L12), final order from the scan. Never L0.

**Solo-cell test plan.** One excision (scan-best conv block), arm A only, frozen review composed on frontier. Register the harness work item (13-layer child breaks the block-count/layer-type registry) BEFORE the cell. Arm B as a separately-priced follow-on cell only if arm A grades alive. Kill trigger: all 13 blocks scan unpredictable → class dies at scan stage with a sealed negative.

**Risks.** Cross-block linear predictability may be small (the M2 decay-law objection is correctly rebutted — that law measured a fold ERROR's predictability, not a parent block's own write — but the scan is the only arbiter). Depth-distribution shift for all downstream layers (the child is a self-consistent function; frozen review is the gate). Materialization/registry work is real and budgeted.

**Scaling transfer.** Strong: depth redundancy is the best-documented redundancy in large models; the pre-buy scan recipe (upstream-features → summed-writes ridge) transfers verbatim.

---

### A3. F-C — conv-block indivisible channel coreset at 8x
*Kept as reviewed; both verdict demands are binding.*

**Math.** The LFM2 conv block is channel-separable end to end (in_proj emits B/C/x over one shared private axis — 3072 = 3×1024; FIR taps per channel; gating per channel; only out_proj mixes), so a channel subset S is a STRUCTURALLY EXACT sub-block. Keep in_proj row-slices {B_S, C_S, x_S}, taps[S] verbatim (bit-faithful parent gate/FIR temporal dynamics); re-solve only out_proj columns by prior-anchored D² ridge against the parent's sealed block write rows. Selection: s_c = E_{D²}[|ctx_c|]·||out_proj[:,c]||₂ on the gated context stream. Run at S8 (keep 128/1024) — built AT the two measured conv laws.

**Solve procedure.** (1) Evidence: sealed pre-writer conv context endpoint (n×1024) + sealed block write rows; if the endpoint is absent at a site, compute by deterministic fp64 CPU per-channel replay of B/C gating + 3-tap FIR on cached block inputs (evidence prep under the sealed collector — cost, not risk; must go through collector sealing). (2) **Pre-registered measurement, before any build:** per conv site (L0,1,3,5,7,9,11,13) Lorenz curve of s_c, top-128 mass share, per-row coverage quantiles, AND the rank correlation between the three per-role masses (B vs C vs x) — the one place "structurally exact" degrades silently into a compromise if roles disagree. (3) Score, keep top-128, deterministic ties. (4) Install verbatim slices: in_proj' 384×1024, taps' (128,1,3). (5) Ridge out_proj' (1024×128) with prior out_proj[:,S], relative λ, update_scale ≤ 1.0 guard, full receipt telemetry. (6) Materialize child block at width 128; norms untouched.

**Storage.** Parent block unit 3,145,728 + 3,072 + 1,048,576 = 4,197,376. Child 393,216 + 384 + 131,072 = **524,672 = 0.12500x exact (8x), zero aux** (channel axis is block-private). Saves 3,672,704/site; 4 deep sites = 14.7M. **Registered S4 sibling cell** (keep 256: 786,432 + 768 + 262,144 = 1,049,344 = 0.2500x exact) at the same site, so a dead S8 attributes cause (mechanism vs level) instead of burning the class.

**Targets.** C03+C04+C05 as one indivisible block unit. Sites L9/L11/L13 first (deeper-is-safer conv law), subject to the measurement. Conv = 20.7% of body; routes around the C03 stacked-layout kernel debt by construction (slicing + one writer ridge, no stacked solve).

**Solo-cell test plan.** Measurement first (all 8 sites); build S8 + S4 siblings at the measurement-best deep site; frozen review composed on frontier; re-price install base after acceptance.

**Risks.** B/C/x role-mass disagreement (visible pre-build). Whole-block S8 is a new regime vs C05-only S8 evidence (S4 sibling attributes). Evidence-prep sealing cost where the endpoint is missing.

**Scaling transfer.** Moderate-strong for hybrid conv/SSM architectures (channel-separable blocks are the norm there); the exact-sub-block argument transfers wherever separability holds.

---

### A4. S+L — one-shot evidence-scaled sparse-plus-low-rank split
*Kept as reviewed; the offset-count nit is pinned.*

**Math.** W_child = L + S: L = UV^T rank-r, S k-entry sparse in the ENTRY basis, both from one deterministic pipeline under the evidence input metric. G = X^T X from sealed rows; Rt = (G + λI)^{1/2} (one eigh). One-sided weighted LRA is EXACT: SVD(W·Rt), truncate r, map back through Rt^{-1}. Sparse part: residual entries scored by R_ij²·G_jj (diagonal approximation for selection only), top-k with pinned ties (value desc, then row, then col). One fixed second pass (re-SVD of W−S₁, re-pick S₂) — fixed count like OMP's k, flagged-borderline and accepted. Rank-release logic: one large outlier entry costs plain SVD an entire rank-1 term (m+n params); moving it to S costs 2 params.

**Solve procedure.** (1) **Pre-registered kill test, from the checkpoint alone, ~10 min:** per-component kurtosis / participation ratio of D^{1/2}-scaled entries; near-Gaussian ⇒ S buys nothing over T4 ⇒ component skipped free. (2) eigh(G+λI) → Rt, Rt^{-1}. (3) SVD(W·Rt) → L₁. (4) Score residual, top-k → S₁. (5) Fixed pass 2 → L₂, S₂. (6) (r,k) split by the PRE-REGISTERED analytic exchange rule: greedily compare marginal captured weighted energy of the next singular value (cost m+n) vs the best (m+n)/2 not-yet-kept entries (cost m+n incl. ids); stop at crossover. (7) Install L₂ as two factors + S₂ as CSR; direct state_dict materialization ((x@V)@U^T + sparse_mm), no hooks (fixes the MXP1 defect class). **Pinned:** CSR offsets = out_rows+1 per the ACTUAL out×in layout (C13 is 1024×2560 ⇒ 1,025 offsets, not 2,561; k recomputed accordingly at install — the sealed example erred conservative).

**Storage.** r·(m+n) + k values + k ids + (out_rows+1) offsets, all counted. C08 attn_v (512×1024, budget 131,072): r=48 → 73,728; k=28,415 → 56,830; +513 = **131,071 = 4.0000x**. C13 (budget 655,360): r=128 → 458,752; remainder re-split under the corrected 1,025-offset count at install, exact ≤0.25x enforced. 8x probe: halve both terms.

**Targets.** Primary: C11/C12/C13 across the 14 MLP layers, priced per-site against the actual install base. Secondary: C08/C09 at L2/L8/L12 — where every rank-truncator died; S restores exactly the heavy-tailed structure truncation smears.

**Solo-cell test plan.** Kurtosis screen across all candidate components first; one cell at the screen-best SwiGLU site vs the sealed T1/T8 twin at identical storage (single variable: split family); frozen review. Designed contrast with FSS (local outliers verbatim vs global smooth patterns) preserved if FSS's ruling ever clears.

**Risks.** Near-Gaussian scaled entries ⇒ T4-plus-overhead (free kill). Ids cost half of S's budget ⇒ S must beat SVD marginal energy >2x (exchange rule makes explicit). Diagonal-of-G selection misweights if off-diagonal correlation dominates.

**Scaling transfer.** Strong: outlier channels/heavy tails are universal in trained transformers and grow with scale; pipeline is one eigh + one SVD + two top-k regardless of size.

---

### A5. F-B — GQA value-channel CUR (verbatim per-head value dims + closed-form o re-solve)
*Kept as reviewed; the materialization gate is hard.*

**Math.** Attention mixing is per-channel (context[:,(q,c)] = A_h(X Wv_h)[:,c]) and the mixed context is a sealed evidence endpoint, so contracted-v context is an exact COLUMN SUBSET of sealed evidence — no attention replay, no new captures. Per kv-head: keep 16 of 64 real value channels verbatim (Wv rows); re-solve out_proj against sealed attention write rows on the corresponding sealed context columns, prior-anchored at surviving Wo columns. A untouched (q/k verbatim): the fold cannot move WHERE information flows, only its channel resolution.

**Solve procedure.** (1) Evidence: sealed context endpoint Cmat (n×1024) + sealed write rows; parent Wv (512×1024), Wo (1024×1024). (2) Score s_{h,c} = Σ_{q∈group(h)} E_{D²}[|Cmat[:,(q,c)]|]·||Wo[:,(q,c)]||₂; **per-group disagreement between the two q-heads reported in the measurement** (registered). (3) Top-16 per kv head, uniform (rectangular tensors), ties by channel id. (4) C_S = 256 selected sealed context columns. (5) Prior-anchored D² ridge for Wo' (relative λ, update_scale ≤ 1.0, full receipt telemetry). (6) Install Wv' 128×1024 verbatim, Wo' 1024×256. (7) **Materialization ruling sealed BEFORE build:** asymmetric v head_dim (16 vs 64) likely needs a per-site eager attention module (trust_remote_code class); if custom-module is unacceptable the class CLOSES — zero-padding is forbidden (padded zeros are stored parameters, voids 4x). (8) **Pre-build measurement:** per kv-head, per site L2..L12, interpolative residual of sealed context blocks under top-16 real channels vs same-budget SVD residual — real channels track the SVD floor ⇒ build; large gap ⇒ content is genuinely mixed ⇒ attn_v closes WITH A CAUSE.

**Storage.** v+o unit: parent 1,572,864 → child 131,072 + 262,144 = **393,216 = 0.25x exact, zero aux** (uniform per-head selection ⇒ rectangular; within-head order absorbed by Wo'). 1.18M/site; 6 sites = 7.08M.

**Targets.** C08+C09 as one designed coupled pair. First L12 (most tolerant attention site), then L8.

**Solo-cell test plan.** Measurement across all 6 sites; if it clears, one cell at L12 composed on frontier, frozen review. Either outcome is a law: verbatim channels live where every basis died, or attn_v closes causally.

**Risks.** Kernel contract (gated as above). GQA usage disagreement (reported). Rotation-invariant training gives no a-priori channel-axis alignment — expected outcome skews negative; acceptable because the measurement is free and the negative is closure.

**Scaling transfer.** GQA is the dominant modern layout; both the mechanism and the causal-closure result transfer.

---

### A6. CBT — block-Toeplitz tied ridge (LDR-lite, full-rank weight sharing)
*Fix applied: one pinned tying rule replaces the three implicit ones.*

**Math.** Constrain the child to a WEIGHT-TYING linear subspace: partition W' = P_r W P_c^T into q_r×q_c blocks of b×b, tie by the PINNED rule **class(i,j) = i−j (block-Toeplitz, q_r+q_c−1 base classes)**, then a single pinned evidence-weighted greedy merge schedule down to the budget-derived class count c. Because the class is a linear subspace, the evidence-weighted fit is one exact constrained ridge on T1's own normal-equation objects; it decouples by row-offset into b independent SPD solves of dim c·b (verified). Result is generically FULL RANK: the first family offered to the truncation graves that cannot annihilate 87.5% of input directions.

**Solve procedure.** (1) Level-1 arm: IDENTITY permutations (zero index storage) — adaptive spectral co-clustering permutations are an ablation arm, not a confound. (2) c from the storage budget; base Toeplitz classes; pinned greedy merge of diagonals closest in the evidence-weighted metric (deterministic, pinned ties). (3) Per row-offset ρ ∈ [0,b): assemble (c·b)-dim normal equations from the sealed input Gram and target A·W_parent^T; solve with prior-anchored relative-λ ridge. (4) Install c blocks of b×b + class-map ints + (permutation ints if the ablation arm); runtime expands blocks on forward via the **shared structured-module materializer (owed to T4; build first — shared with HODLR-V)**.

**Storage** (all maps/perms counted; internally consistent under the pinned rule): attn_o C09 1024×1024, b=128, 8×8 grid → 15 base classes exactly = 245,760 + 64 + (2,048 perm arm) = 247,872 → **4.23x**. attn_v C08 4×8 grid → 11 base → merge to 7 = 116,256 → **4.51x**. mlp_up C12 b=256, 10×4 grid → 13 base → merge to 9 = 593,448 → **4.42x**. 8x probe: halve c.

**Targets.** The rank-truncation graves: C08, C09, C05 (conv_out, the measured wall), C12. Success is pure new coverage touching no live lane.

**Solo-cell test plan.** Identity-perm arm first, one cell at C09/L12 vs the sealed T1 twin at matched storage; frozen review. Optional pre-registered prioritization heuristic (not a gate): evidence-weighted dispersion of parent blocks along Toeplitz diagonals vs random tying, computed from the checkpoint.

**Risks.** Channel space has no translation structure — the whole bet is full-rank-vs-truncation; the merge schedule is the only adaptation. Thinner effective evidence per unknown than T1 (solver-health note, not a gate). Materializer is a dependency and a bug surface.

**Scaling transfer.** Methodology transfers anywhere; the prior stays equally unjustified at scale — value is the failure-axis information.

---

### A7. F-DECAY — attention mixer → verbatim heads + fixed-spectrum decay-bank tail
*Fixes applied: selection split, GQA fallback pre-committed, sequence-order precondition, far-mass receipt.*

**Math.** Replace one attention mixer (3,145,728 params) with g=2 verbatim parent Q-heads (+ KV group + o-slices, exact) + E=6 causal exponential-decay channels with fixed scanned rates λ_e, input/output maps solved by ridge, stored factored at rank 32: y = Σ_kept(exact head) + Σ_e V_e^T(U_e^T L_{λ_e} x). Tail is linear in its parameters given fixed decays ⇒ ridge + truncated SVD + one factored re-solve. Unbounded-support kernels attack exactly what killed T6 (rho_far 0.52–0.69).

**Solve procedure.** (1) **Preconditions:** verify sequence-ordered captures (decay features are causal convolutions over position); if shuffled, the bounded recapture is a precondition, not a mid-cell patch. **Carve a selection split out of the 4,751 fit rows** for ALL selection (λ grid, head keep/morph ranking, post-truncation refit); the 1,584 heldout rows touch nothing but the final price. (2) Head selection on the selection split: keep the 2 heads with WORST decay-predictability (content-dependent retrieval/induction heads); morph the rest. **GQA constraint resolved up front:** either constrain the kept pair to one KV group (selection-rule weakening stated honestly) or pre-commit the 2-group fallback: E=4 decays at rank 32 (262,144) + 524,288 kept = 786,432, still exact 4x. (3) Decay features on log-spaced grid; E rates by greedy selection-split residual reduction (deterministic 1-D scans). (4) Stack F (N×6144); ridge T against Y minus kept-head contribution. (5) Per-block truncated SVD at rank 32; fix U_e; one final ridge on the stacked right factors. (6) Install kept slices verbatim + {U_e, V_e, λ_e}; per-head q/k norms on kept heads install verbatim (W4B precedent). **Receipt must report the rho_far analog explicitly** — the distinctness-from-T6 claim rides on far-mass coverage.

**Storage.** Kept 393,216 + bank 6·(1024·32 + 32·1024) = 393,216 → **786,432 = 0.25x exact** of 3,145,728, + 6 scalars counted. Fallback arm: 524,288 + 262,144 = 786,432. Per layer removed 2,359,296; six layers = 14.2M.

**Targets.** The mixer as ONE unit (C06–C10 jointly), sites L2..L12; first L12, then L10/L8. Sidesteps the attn_v grave by never asking v to survive a basis contraction alone.

**Solo-cell test plan.** One cell at L12, head-to-head against F-LANDMARK at the same slot (M-7); frozen review; the attention estate funds only the winner. Install machinery precedent: chimera-omni recurrent-mixer kernels.

**Risks.** Content-blind fixed decays; long-generation extrapolation unmeasured (frozen review's free-run tests it); recapture cost if order was shuffled.

**Scaling transfer.** Strong: attention→SSM-tail compilation is directly relevant to any large softmax-attention model; the per-head predictability scan transfers.

---

### A8. F-LANDMARK — verbatim-state landmark memory for attn_k/attn_v
*Fixes applied: fit-split selection, betas counted, component-accounting ruling pre-registered.*

**Math.** For converted KV heads, delete k/v projections and install m=256 ACTUAL cached key/value vectors as parameters, selected by attended-mass-weighted pivoted QR: y = softmax(β·q M_k^T) M_v W_o-slice. Generalizes the deepest sealed insight from tables to states: the value pathway as sparse in a dictionary of its own frequently-attended states. First mechanism asking v to survive as verbatim points on its own manifold.

**Solve procedure.** (1) Pre-RoPE K_h, V_h (N×64) on fit rows; parent attention probabilities recomputed offline from parent weights (no new capture). (2) Attended-mass weights w_s (D²-style weighting driving row SELECTION — the inversion of T8's use). (3) m=256 landmarks by weighted pivoted QR on diag(√w)K_h, CPU fp64, deterministic (T5 nonportability lesson). (4) Install M_k, M_v VERBATIM; converted heads drop RoPE on q (position-free content addressing; position mixing stays with conv layers + kept heads). (5) **β from a deterministic grid on a fit-row selection split** (heldout prices only); with Phi = softmax(βqM_k^T)M_v fixed, o-slice (64×1024) by production ridge against the parent's sealed per-head output. (6) Keep g worst-predicted KV groups fully verbatim; convert the rest. **Pre-registered before the first cell:** the component-accounting ruling for per-head o-slice refits (one C09 component vs several) — obtained, not assumed.

**Storage.** Per converted KV group: parent 131,072 → child 2·(256·64) = **32,768 = 4.0x exact**, no indices (rows copied into parameters). **β scalars counted: ~36 across six layers** (hygiene per A9 standard; still comfortably 4x). 6 of 8 groups × 6 layers = 3,538,944 removed.

**Targets.** C07+C08 jointly per KV group (the only components with ZERO alive 4x coverage in the ledger) + C09 refit in place. First cell L12.

**Solo-cell test plan.** Head-to-head vs F-DECAY at L12 (one cell each); frozen review; fund the winner only. If it lives: C07/C08 open across six layers AND verbatim-state variants at C03 get licensed. If it dies: the verbatim-rows law localizes to the token interface — a sealed physics boundary either way.

**Risks.** Converted heads cannot retrieve from live context (RBF-style FFN in function class) — pre-scan + kept-verbatim groups mitigate; RoPE-drop gauge error only approximately absorbed (selection-split residual prices pre-install); 4,751 prompts bound the memory (the Zipf bet the token head won).

**Scaling transfer.** The law-localization result transfers regardless of outcome; mechanism transfers to any GQA model with cached-state evidence.

---

### A9. HODLR-V — evidence-ordered hierarchical block-separable fold, verbatim diagonal
*Fixes applied: honest gate, pinned orderings, sequenced behind S+L.*

**Math.** Permute channels so evidence co-activation clusters are contiguous; impose one/two-level HODLR: diagonal blocks copied VERBATIM from the parent; each off-diagonal block replaced by evidence-weighted low-rank (exact one-sided weighted LRA per block); ranks by deterministic water-filling under the global budget. Generalizes verbatim coverage from rows to interaction blocks.

**Binding fixes.** (1) The analytic gate is restated as **"fraction of total evidence-weighted interaction energy inside diagonal blocks"** — no zero-drift language (a within-cluster input still routes through g−1 approximated off-diagonal blocks); alive-floor threshold pre-registered by analogy to the 81% floor BEFORE any numbers. (2) Pinned once for square and rectangular alike: output-side ordering = independent Fiedler ordering of output-side evidence energy; g pinned per component class up front. (3) **Sequenced strictly after S+L**; the free Gram coverage screen runs across ALL candidate sites first; build only where coverage clears the floor and S+L has not already answered the concentration question at that site.

**Solve procedure.** Input co-activation Gram; Fiedler ordering (eigh, pinned signs/ties); g contiguous clusters; Hungarian pairing of input clusters to output row-groups for rectangular components (exact combinatorial optimum); diagonal blocks verbatim; per-off-diagonal-block eigh → Rt_blk, SVD(W_blk·Rt_blk), global greedy water-filling; install permutations + dense blocks + factors + rank table.

**Storage** (all counted). C13 g=8: diag 327,680 + perms 3,584 + table 64 + 723 ranks at 448/rank = 655,232 ≤ 655,360 → **4.0x**. C09 g=8: 131,072 + 2,048 + 64 + ~503 ranks at 256 → 4.0x.

**Targets.** C13 at L5 first, then C11/C12, then C09.

**Solo-cell test plan.** Coverage screen (free) → single cell at the screen-best site vs sealed T1 twin; frozen review.

**Risks.** Post-RMSNorm Grams may be near-isotropic (free kill). Off-diagonal is deep rank truncation (mean r~9–13) — the known killer; the class lives entirely on the coverage number, and energy metrics have lied before. Largest deterministic-plumbing surface in Track A.

**Scaling transfer.** Weak-moderate; biclustered interaction structure is model-specific.

---

### A10. F-D — verbatim-head + sparse-tail SwiGLU (gated tail-extension arm of VC-FOLD)
*Fixes applied: feasibility identity, offsets deleted, three hard gates.*

**Math.** Head = h verbatim channels (as VC-FOLD); tail = extra child channels whose w1/w3 columns are k=8-sparse combinations of STORED head columns (tail pre-activations = one sparse matmul reusing head pre-activations); each tail channel gets a dense W2 row; joint writer ridge over [Phi_head, Phi_tail] with prior anchoring. Tests whether the verbatim-coverage law generalizes to body feature space.

**Binding gates (hard, in the cell registration):** (a) VC-FOLD's concentration table sealed; (b) one graded VC-FOLD cell at the contrast site; (c) materialization class declared (HF-materializable vs research-only) BEFORE build (MXP1 lesson). **Knob replaced by the feasibility identity:** tail_max(h) = floor((1,966,080 − 3·1024·h)/(1024+16)), re-verified per split at install against the ACTUAL sparse layout. 576/229 is dead; 576/189 is the legal neighbor. **CSR offsets deleted** (fixed k=8 makes extents implicit).

**Solve procedure.** As VC-FOLD steps with h=512; tail: fp64 CPU OMP (double-MGS, documented deterministic ties — T5's sealed determinism recipe reused and cited in the receipt) approximating each dropped channel's two pre-activation traces over the head trace dictionaries; joint prior-anchored D² ridge with update_scale ≤ 1.0.

**Storage** (512/371, no offsets): 1,048,576 + 524,288 + 379,904 + 2·(371·16)=11,872 → **1,964,640 = 0.24982x**, margin 1,440 params; install-time re-verification mandatory.

**Targets.** MLP sites where VC-FOLD's top-640 coverage is marginal but top-~900 clears — the coverage extender for the marginal half of the 14 sites, run head-to-head vs VC-FOLD at one such site.

**Test plan.** One matched-storage within-site contrast cell; frozen review. A clean loss to VC-FOLD is itself the scaling-relevant negative (verbatim beats synthesis at interfaces but not in body feature space).

**Risks.** Tail features ARE synthesized mixtures through SiLU — the exact mechanism behind every dead body basis fold; sub-1,500-param margin; custom-module debt.

---

### A11. Regime-4b — certified secant-bilinear CP re-expression (deferred arm)
*Split per verdict; 4a already merged into VC-FOLD arm E.*

Gated on: (i) VC-FOLD's census showing a fat quasi-linear population net of OFF; (ii) **beta handling = EXCLUSION only** (certify only units whose evidence-weighted intercept-term energy < pre-registered ε; ridge absorption is not a permitted install path — a narrower SwiGLU has no linear bypass); (iii) Jennrich independence/conditioning gates pre-registered, HOSVD fallback fail-closed, re-installed CP units regime-certified on heldout with failures returning to the verbatim bin. Storage: same triplet arithmetic (≤640 stored units → 0.25x, zero aux); the gain is coverage per slot, not count. Worst case: never runs; 4a already banked free allowance inside VC-FOLD.

---

### A12. Gate-Jacobian metric pair factorization (census-gated hedge)
*Demoted per verdict; runs only where VC-FOLD's census FAILS.*

Shared rank-k read basis for (C11,C12) in the through-gate metric G_bar = mean M(x)^T M(x), M(x) = W2[D_a W1 + D_b W3]; exact whitened generalized eigendecomposition; W2 re-ridge (single-pass, prior-anchored) against cached block outputs through the folded readers' TRUE gate features. Storage: 6144k; k=213 → 1,308,672 = 4.006x on the 5,242,880 pair; S8 k=106. **Two pre-build gates:** (1) whitened-pencil census only at diffuse-unit sites; (2) surrogate-fidelity check — metric-derived U_k must beat an equal-k plain Frobenius baseline on the TRUE sampled objective Σ_t||M(x_t)(I−P)x_t||² on heldout, else the class is T4 with makeup and dies without a build. M-level grading before any X4. Spends its budget in the measured most-dangerous error class (pre-gate read error through silu) — hence hedge status.

---

### A13. FSS — fixed-spectral sandwich (double-conditional probe)
*Fixes applied; two preconditions, both external.*

W_child = H_m^T Δ H_n with fixed seed-free transforms (Hadamard/DCT) and a sparse core Δ solved linearly (energy-rule arm + A9-verbatim OMP arm with swapped dictionary). Storage: k values + k ids + (m+1) offsets; C09 k=130,559 → 262,143 = 4.0x — **entirely contingent on Cyril's explicit written ruling that fixed, data-independent, seed-free transforms price at zero parameters** (RoPE/silu precedent; adverse ruling = immediate kill at zero cost). **Precondition 2:** Lorenz coefficient-energy pre-check must beat the entry basis by a pre-registered margin at a named component (start C05). 2560 handling (DCT-II vs 5×512 block-Hadamard) pinned per component NOW. Selection weight diag(H G H^T) carries the same diagonal-approximation caveat as S+L's selector (stated). Value even on death: the S+L/FSS designed dual localizes where body matrices keep behavioral mass.

---

### B1. XGRAM — pairwise-interference portfolio selection
*Kept; four registration conditions enforced.*

**Math.** Terminal error signature per candidate lane: ε_i = G_ℓ(i)·E_i on heldout rows (E_i from sealed isolation artifacts; G from the shared artifact §4.1). Interference Gram K_ij = ⟨ε_i, ε_j⟩; composed-drift predictor D(S) = sqrt(Σ||ε_i||² + 2Σ_{i<j}K_ij); lane selection = deterministic quadratic knapsack under D(S) ≤ 0.3765. Negative K_ij pairs are free sub-additive capacity per-lane accounting cannot see.

**Registration conditions.** (1) ε_i for C11/C12 lanes uses endpoint transfers G_e (2560-dim writes cannot be left-multiplied by a 1024 metric); endpoints below the R² floor are EXCLUDED from the universe, not mispriced. (2) Retrospective bar pre-registered precisely: leave-one-out predicted-vs-sealed correlation vs the scalar law's 0.62 on the ~15 sealed children (MXL11, MXN4, MXO3, MXP2, MXO4's collapse, the 0.4977x wreck) — not an in-sample number. (3) K regularized to rank ≤3, no per-pair free parameters; deterministic tie-breaking for greedy-with-swaps (T5 double-MGS precedent). (4) Interface×body handled multiplicatively per sealed physics.

**Plan.** ~30 matmuls + one 30×30 eigendecomposition; go/no-go retrospective; if calibrated, emit the predicted-optimal ≤0.5x portfolio and build exactly ONE prospective child for frozen review; publish K as a program asset pricing wave-14 CUR/NKP/Monarch/TT lanes. Failure mode informative: diagonal K confirms additivity as physics.

---

### B2. Ω-RIDGE — terminal-transfer-metric folding
*Fix applied: endpoint type-checking.*

**Math.** Per-layer transfer surrogate G_ℓ (ridge of h_final on residual.post_layer at ℓ, from sealed parent captures — no new capture); output metric Ω_ℓ = G_ℓ^T G_ℓ + τI; any fold solved in that metric via SVD in the chol(Ω)-rotated whitened space — exact, same cost class as T1; error steered into the directions the parent itself attenuates before the head. τ→∞ recovers plain T1 (never-worse-by-construction at solve level).

**Binding fix.** Wave-1 cells restricted to residual-writing components where the 1024 metric type-checks — **C13, C05 (with the post_operator first-order caveat stated), C09** — OR endpoint-specific transfers G_e regressing h_final on the component's own write rows (1024×2560, only 1.86x overdetermined ⇒ stronger relative ridge required), with the same pre-registered heldout R² > 0.5 floor per endpoint. No pulled-through metrics across the gate nonlinearity (data-dependent Jacobian, out of scope). G/R are solve-time artifacts, NEVER shipped.

**Step 0 (free falsification, runs before everything in Track B).** Fit G_ℓ for all 14 layers + endpoint transfers; publish heldout R² and σ_min/σ_max per layer/endpoint. Flat spectra everywhere (identity carry dominates) ⇒ the class — and ANTIFOLD's dependence — dies for the cost of 14 ridge solves.

**Plan.** First cell: C13.T1Ω.M4.L05 vs sealed twin C13.T1.M4.L05 — identical storage, one variable; frozen review. Storage identical to the base technique at every level; zero aux.

---

### B3. ANTIFOLD — drift-conjugate sequential folding
*Fixes applied: endpoint fix, claim rewritten, contingent registration.*

**Math.** After lanes 1..k−1: measure composed terminal residual r on the sealed corpus (one child forward pass, ALS licence); solve lane k against the shifted target Y_k' = Y_k − β·G^+·r̂ (λ-regularized pullback, closed form; β ∈ {0, 0.25, 0.5, 1}); the lane's unavoidable error cancels accumulated drift instead of adding — the non-local objective the M1 postmortem named as missing, in the only gradient-free form available. Trust region ||Y_k'−Y_k|| ≤ ρ_max||Y_k||.

**Binding fixes.** (1) Pullback restricted to lanes with 1024-dim residual writes (C13/C05/C09) or endpoint transfers G_e for C11/C12. (2) Non-regression claim rewritten: **cannot regress on the heldout drift metric only**; frozen review remains the sole verdict and can still differ — β selection is prioritization, never a gate. (3) Registered contingent on Ω-RIDGE step-0 (shared G artifact, shared R²>0.5 floor; β forced to 0 wherever the floor fails). Order dependence bounded at 2 registered orders (decreasing drift price; depth-ascending). Wreck cap: β=0 whenever the base's heldout terminal relL2 > 0.3765.

**Plan.** First experiment: rebuild MXP2's exact 6-lane set ANTIFOLD-sequentially (bit-comparable 0.703409x; C11/C12 lanes via G_e or excluded per the endpoint rule) vs sealed 6/15 twin; frozen review. Storage unchanged from base lanes; zero aux; targets = incremental lanes 7–12 (C11/C12/C13 S4 on L03/L08–L12, C05@S8 second sites, C07@L12).

---

### B4. W-ABSORB-SB — same-block absorber refit (rebuilt per verdict)
*Original downstream placement and original solve target are killed (see §6); this is the surviving version.*

**Math.** Zero-storage refit of an unfolded SAME-BLOCK writer where the fold's own error has crossed no block nonlinearity — the sealed 28%-linear regime, bought at zero params instead of M2's 525k/route. **Retargeted solve:** Y' = h_post_block^parent − h_pre_write^child (parent rows from the sealed cache; child rows from the same capture pass producing X_c), so the ridge regresses the actual gap, not the parent write (the original target provably left the residual carry untouched).

**Procedure.** (1) **Enumerate the addressable surface on MXP2/MXO3 FIRST:** legal absorbers = unfolded C13/w2 beneath folded C11/C12 in the same FFN; unfolded C05/out_proj above folded in-operator conv tensors. Blocks folding the full trio have no legal absorber; empty surface ⇒ kill without building. (2) One deterministic child forward pass of the sealed fit corpus → X_c. (3) Closed-form prior-anchored relative-λ ridge; ρ = ||ΔW||_F/||W_a||_F trust region swept upward from 0.02 ∈ {0.02, 0.05, 0.10}; λ raised until compliant. (4) Fail-closed: install refused unless heldout terminal relL2 improves; A12 collateral non-regression. (5) Alive-base-only (W13A self-consistent-function trap); installs in registered depth order with re-capture between. Kill triggers: larger ρ grades worse than smaller (LAW breach signature); best absorber Δdrift < current marginal lane price.

**Storage.** Exactly zero added parameters; the fold set's ratio bit-unchanged.

---

### B5. Residual gauge — exact zero-parameter rotation preconditioner (infrastructure)
*Demoted per verdict; not a fold class.*

Exact global orthogonal reparameterization of the residual stream (gamma absorption exact; tying survives; conv-FIR and RoPE untouched). **Struck:** claimed benefit to VC-FOLD/regime census (Phi is gauge-invariant; T1 rotation-covariant). Real beneficiary set: T5 routes and the queued axis-sensitive structured classes only. **Protocol:** (1) free census — eigenspectrum of pooled Sigma_hat with a pre-registered concentration threshold; (2) if passed, ONE gauge (chosen by census, not both); (3) ONE paired cell — native vs rotated, same solve, same budget — on the single most axis-sensitive wave-14 structured lane (NKP is the natural pick); expand only on a measured behavioral win. Tied-head seam: prefer the exact fused R^T·diag(γ_F)·R with its 1,048,576 params charged IN FULL against the lane the gauge claims to improve (no charge survival ⇒ no win); the γ' diagonal-refit variant only with its own frozen behavioral grade first. bf16 round-trip audit fail-closed.

---

## 4. Shared infrastructure and preconditions

### 4.1 G transfer artifact (common kill switch for Track B)
Built once, before any Track-B build: per-layer transfers G_ℓ (14 ridge solves, residual.post_layer → h_final) AND per-endpoint transfers G_e for 2560-dim writers (stronger relative ridge; 1.86x overdetermination noted). Published: heldout R², σ_min/σ_max per layer/endpoint. Pre-registered floor R² > 0.5. XGRAM, Ω-RIDGE, ANTIFOLD all consume this one artifact and die together if it fails — which is why it runs first and costs ~an hour.

### 4.2 Structured-module materializer (owed to T4)
One shared runtime install path for factor/block/sparse children: two-factor low-rank (T4 debt), block-expansion (CBT), block-pattern + per-block factors (HODLR-V), CSR sparse add (S+L), sparse-tail MLP (F-D), decay-bank op (F-DECAY), landmark head (F-LANDMARK), eager asymmetric-v attention (F-B). Built once before the first cell that needs it. Every class above materializes directly to state_dict / declared custom module — no forward hooks (MXP1 defect class closed).

### 4.3 Free pre-build measurement battery (wave-14 opens with ALL of these)
1. VC-FOLD concentration census, 14 MLP sites (gates A1, A10, A11, A12).
2. F-EXCISE 14-block predictability scan (gates A2).
3. F-C conv concentration + B/C/x role-mass rank correlation, 8 sites (gates A3).
4. S+L kurtosis/participation-ratio screen, checkpoint-only (gates A4).
5. F-B interpolative-vs-SVD context residual, 6 attention sites (gates A5).
6. F-DECAY/F-LANDMARK per-head predictability scans on the selection split (gates A7/A8).
7. HODLR-V Gram coverage screen (gates A9).
8. G artifact + spectra (gates B1–B3).
9. XGRAM retrospective calibration on ~15 sealed children (gates B1's build).
10. W-ABSORB-SB addressable-surface enumeration (gates B4).
11. Sigma_hat eigenspectrum (gates B5).
12. FSS Lorenz curves — only after the ruling (gates A13).

### 4.4 Protocol disciplines binding on every class
- **Selection/heldout split:** any data-driven selection (decay rates, head rankings, β, thresholds) uses a split carved from the 4,751 fit rows; the 1,584 heldout rows PRICE only, never choose.
- **Determinism:** fp64 CPU solves, pinned tie rules, receipts citing the hardened A9/T5 machinery where reused.
- **THE LAW as kernel guards:** update_scale ≤ 1.0 / trust regions fail-closed in code, not prose.
- **Storage:** every auxiliary int counted; install-time re-verification against the ACTUAL layout wherever margins are thin (F-D) or layouts were mis-stated (S+L offsets).
- **Verdicts:** frozen behavioral review only; drift and all censuses are prioritizers.

### 4.5 Rulings required from Cyril before the affected cells
1. **FSS:** do fixed, data-independent, seed-free fast transforms price at zero parameters (RoPE/silu precedent) or as stored matrices? Adverse ⇒ A13 killed at zero cost.
2. **F-LANDMARK:** component-accounting for per-head o-slice refits (one C09 component vs several) for interference pricing.
3. **Materialization policy:** which custom-module classes (4.2) are acceptable as HF-materializable installs vs research-only declarations — decided per class BEFORE build.

---

## 5. Already-queued wave-14 classes: lens refinements

**CUR / interpolative decomposition.** Superseded-in-place: **VC-FOLD is registered as this slot's refinement** (feature-space CUR on the nonlinear map, trio-indivisible, D²-mass selection instead of reconstruction-optimal criteria, write-row prior-anchored objective, zero-aux install, go/no-go concentration census the queue entry lacked). If a weight-space CUR control is still wanted, run it as a matched-storage twin against a VC-FOLD cell with leverage/pivoted-QR selection and deterministic ties — one cell, control only. F-B and F-C are the same refinement pattern (verbatim substructure + closed-form writer re-solve) instantiated on attention and conv.

**Nearest-Kronecker-product (Van Loan).** No lens proposed a direct replacement; three transferable additions: (1) S+L's pre-registered analytic marginal-energy exchange rule transfers to Kronecker rank selection; (2) NKP is the most basis-dependent queued class (the Van Loan rearrangement) — the designated target for the residual gauge's one paired cell (B5); (3) Ω-metric applicable only at residual-writing endpoints per the type-check rule.

**Monarch (block-diagonal + low-rank).** Keep queued, but sequence AFTER CBT: CBT is the designed adjacent contrast (single linear tying subspace + one-shot exact ridge vs Monarch's bilinear factor product), and a CBT result at the graves re-prices Monarch's prospects before its (harder, bilinear) closed-form question is confronted. Shares the structured-module materializer (4.2).

**Tensor-train on stacked tensors.** Distinct from Regime-4b's Jennrich CP (functional contribution tensor of a regime slice vs stacked weight tensors — different objects; no double-counting). TT rank scans obey the selection-split discipline (4.4); TT lanes get priced by XGRAM's K matrix once published; gauge-B incoherence is a plausible benefit for mode unfoldings (same B5 protocol if NKP's paired cell wins).

**All four:** axis-sensitive ⇒ candidate Ω-metric consumers where endpoints type-check; every cell gets a sealed twin at matched storage where one exists.

---

## 6. Kill list

1. **F-MERGE as a standalone class** — mass-conserving scalar and sign flips are absorbed by the W2 re-ridge; function class collapses to feature-space CUR; surviving content merged into VC-FOLD arm D.
2. **W-ABSORB downstream-block placement** — built on a misquoted decay law; the only never-folded downstream datapoint (L13) measured 0.0% linear share with heldout HARM; both named first cells sat in the sealed dead regime.
3. **W-ABSORB original solve target (match parent writes on drifted inputs)** — removes only pass-through amplification; provably cannot touch the residual-carry gap it quoted as its prize.
4. **W-ABSORB two-blocks-deep control cell** — re-measures a sealed row (10.0% at L07).
5. **Ω-RIDGE/ANTIFOLD 1024 layer-metric applied to C11/C12** — R_ℓ cannot left-multiply 2560-dim pre-gate write errors; type mismatch; endpoint-transfer variant survives.
6. **ANTIFOLD's "cannot underperform by construction"** — holds only on the heldout drift proxy; frozen review can regress; claim rewritten.
7. **F-D 576/229 registered split** — computes to 2,011,756 params = 0.2558x, a contract violation inside a knob; feasibility identity replaces it (576/189 is the legal neighbor).
8. **F-D CSR offsets** — deleted; fixed k=8 makes row extents implicit (margin widened 696 → ~1,440).
9. **FSS unconditional build** — dead unless fixed seed-free transforms are ruled zero-parameter AND the Lorenz pre-check beats the entry basis by a pre-registered margin; both preconditions external.
10. **Gate-Jacobian as a primary class** — spends its entire error budget in pre-gate-through-silu error (the measured most-dangerous class); dominated by VC-FOLD at equal price wherever the gated-energy census passes; demoted to census-gated hedge.
11. **Residual gauge as a fold class** — 1.0000x by construction; claimed benefit to VC-FOLD/regime census false by gauge invariance of Phi; demoted to preconditioner infrastructure with a single paired-cell budget.
12. **HODLR-V "zero drift on within-cluster signal"** — false as stated (within-cluster inputs route through g−1 approximated off-diagonal blocks); gate restated as diagonal coverage fraction with a pre-registered floor.
13. **CBT cyclic tying with per-component implicit rules** — three examples, three tying rules = the hidden-DOF class the program fences everywhere; replaced by pinned block-Toeplitz class(i,j)=i−j plus one pinned merge schedule.
14. **Heldout-driven selection anywhere** (F-DECAY λ/head ranking, F-LANDMARK β/group ranking, Ω/ANTIFOLD refits) — protocol contamination; heldout prices, never chooses; selection splits carved from fit rows.
15. **Zero-padding fallback for F-B's asymmetric v head_dim** — padded zeros are stored parameters and void 4x; the honest fallback is closing the class.
16. **Funding both attention tails (F-DECAY + F-LANDMARK) across all six layers pre-comparison** — one L12 head-to-head decides which tail the estate buys.
17. **Regime-4b "final ridge absorbs beta intercepts"** — a narrower SwiGLU has no linear bypass to express the leaked rank-limited linear term; EXCLUSION is the only permitted install path.
18. **F-DECAY "2 verbatim Q-heads + 1 KV group" without the GQA constraint** — silently breaks exact-4x if the kept heads span two groups; single-group constraint or the pre-committed E=4 fallback budget replaces it.
19. **S+L C13 offset count of 2,561** — wrong layout (actual out×in is 1024×2560 ⇒ 1,025 offsets); error was conservative but the count is pinned to the actual layout at install.
20. **Duplicates of sealed classes T1–T9 / closed paths** — none were proposed; recorded so the matrix shows the check ran.

---

## 7. Suggested wave-14 execution order

1. **Measurement battery (§4.3), all items, before any build** — expected to kill 2–5 classes at ~zero cost and re-rank the rest.
2. **Rulings (§4.5) requested in parallel.**
3. **First build slate (contingent on measurements):** VC-FOLD arm A vs C cell at the census-best MLP site; F-EXCISE scan-best conv block; F-C S8+S4 siblings at the measurement-best deep conv site; S+L at the kurtosis-best SwiGLU site vs its sealed twin.
4. **Second slate:** F-B at L12 (if the interpolative measurement clears and the materialization ruling lands); CBT identity-perm cell at C09/L12; F-DECAY vs F-LANDMARK head-to-head at L12; XGRAM's one prospective child (if retrospective calibration passes).
5. **Wrappers:** Ω-RIDGE twin cell at C13/L05; ANTIFOLD MXP2 rebuild; W-ABSORB-SB if the surface enumeration is non-empty — all only after the G artifact publishes above-floor.
6. **Gated arms last:** F-D, Regime-4b, Gate-Jacobian, HODLR-V, FSS — each strictly behind its named gates.

Phase-1 arithmetic this slate can support: VC-FOLD (up to 47.2M across 8 sites) + F-EXCISE (12.06M/purchase) + F-C (14.7M across 4 deep sites) individually or jointly cover the 58.5M gap; the attention classes (7.1–14.2M) and Track B add coverage and drift room from lanes currently priced at zero or infinity. The frozen review, not this arithmetic, decides what is real.