# WAVE-23 SLATE — Round-2 Discovery Synthesis

Input: 4 lenses x 3 proposals = 12. Verdicts applied: **3 killed, 1 merged, 8 survivors -> 7 funded waves** (LP-ATLAS absorbs ATTR-LAMBDA). All specs below have reviewer fixes ALREADY APPLIED — dispatch from this document, not from the raw lens outputs.

Contract restated for every wave: closed-form solves on sealed evidence only (ridge/SVD/OMP/Procrustes/eigen), zero gradients, parameter-count reduction with all aux tensors counted, no quantization, no MoE/conditional execution. Frontier claims require standard-shape export (stock `Lfm2ForCausalLM`, head_dim=64 pinned); custom runtime = research probe only. Parent = LFM2 229,693,184 params.

---

## 0. CYRIL ADJUDICATION QUEUE (block claims, never builds)

- **A1 — Frozen-gate linear response operators** (PAIR-KKT's J_k; ORTHO-PAIR constraint form (ii)): does a fixed linear map assembled from sealed weights + sealed activations, entering a ridge KKT with no differentiated objective and no update rule, count as "a gradient"? One ruling covers both waves. Pre-declared fallback: demote to research probe by mechanism (artifacts remain standard-shape either way).
- **A2 — Vocab row coreset + tokenizer remap**: fold or not? The wave-19 dismissal ("a tokenizer change, not a fold") is a one-line category claim with zero builds behind it, and vocab is not on the refused-by-rule list. Ruling gates FRONTIER PROMOTION only; the build proceeds as research probe regardless.
- **A3 — Priority vector p over the 15 review cases** for LP-ATLAS/ATTR (stated want implies conversation-natural-greeting, multi-turn-correction, instruction-three-actions high; explanation-seasons/logic-ordering low).

---

## 1. RANKING

Rank = hope-under-the-laws x cheapness-of-falsifying-build x standard-export status.

| # | Wave | Hope under laws | Falsifier cost | Export |
|---|------|-----------------|----------------|--------|
| 1 | **VOCAB-CORESET** | Med-high: anchored to the campaign's best solo (MXN2 10/15); L2 and L3 untouched by construction; the re-tokenization shift is the whole bet and is unpriced | Near-zero: zero-solve build (gather rows, emit tokenizer) + 1 review | Stock config axis (`vocab_size`), loader-counted 0.7809x; fold-category ruling A2 pending on the claim only |
| 2 | **SEGBASIS** | Best-reasoned L2 attack extant: removes the shared-budget constraint instead of re-weighting inside it; medium hope on the floor half, no promise on the ceiling half | 1 build + mandatory T* probe twin + 2 reviews, at an existing control shape | Full standard, ZERO added tensors (structure lives in tensor values) |
| 3 | **ORTHO-PAIR** | First direct measurement of L3's cross-term in 652 rows; lever family has a sealed +1 inside the exact murdered composition (W896LSX13 2/15 vs W896X13 1/15) | rho census is FREE (existing sealed captures, zero builds); then 1 soft-arm child | Standard (value-only re-solve of sealed pair shapes) |
| 4 | **PAIR-KKT** | The instrument the M1 postmortem named as missing; 4-case headroom on a living pair; expect L5 swap over pure addition; worst case re-seals 7/15 in bytes | Zero-build pre-read, then 1 build + 1 review | Standard (value-only, 212,910,848 params); A1 pending on mechanism |
| 5 | **LP-ATLAS + ATTR** (merged) | L5 direct license (case-swap sealed 4x at these bases); serves the handoff decision directly; expected magnitude 1–2 cases | Gate build = 1 child (W896 conv-restricted arm); worst case 6 builds | Standard, zero storage delta |
| 6 | **29-DIAGONAL** | Low-moderate (0 ± 1 case honest expectation); the null is load-bearing (kills last cheap solver-side explanation of reader disagreement) | 1 build (W768R) against the richest identical-param control set | Standard, zero param delta |
| 7 | **NKP-VANLOAN** | Low ("probably a grave, cheap to dig") but no law forbids it; closes frontier-report §6's open clause either way | Census FREE (~1 CPU-hr); 1 sealed cell | Research probe permanently (parent-shaped storage-format column) |
| 8 | **EARLY-SPARED** | Medium as mechanism localization, zero frontier progress; buys one law | 2 builds (spared + inverted mirror) | Research probe only (per-block width not expressible in stock config) |

Co-scheduling note: PAIR-KKT and ORTHO-PAIR share the conditioned capture through ARCH.X07KV6JQC and adjudication A1 — run as one paired wave.

---

## 2. FREE PRE-READS (CPU only, zero builds — dispatch immediately; results ORDER, never gate: L1)

- **P1** ORTHO rho census: rho = <e_A,e_B>_M / (||e_A||_M ||e_B||_M), pooled and per-endpoint, over every sealed two-body composition with both solo captures. Predictions pre-registered: rho > 0 for W896+X13A, BX13, X13KV6J; rho ~ 0/negative for X07+KV6JQ.
- **P2** SEGBASIS J(c) census: all 13 cuts, per-endpoint retained read energy printed (never pooled-only), from sealed captures via the residual identity (no new capture).
- **P3** NKP rearrangement-spectrum census: 42 SwiGLU tensors x admissible grids, spectra + evidence-weighted energy-at-4x; Monarch blockwise-SVD control through the identical census.
- **P4** PAIR-KKT linearization pre-read: predict the ALREADY-KNOWN composed damage r_parent − r_child from the pair's sealed dW on held-out rows; publish R2 per tensor group.
- **P5** VOCAB post-closure coverage census: closure-corrected kept set, re-tokenize sealed corpus, assert zero kept-token occurrences lost, publish pre- vs post-closure coverage and tokens-per-byte.
- **P6** ATTR Stage A: D(block, c) from the complete case x block excision-survival matrix (pure arithmetic over the ledger).

---

## 3. SURVIVOR SPECS

### S1. VOCAB-CORESET — vocab_size as the missing standard axis (closure-corrected)

**Wall:** the tied-embedding floor (L6: 29.2% of the model) and the standard-shape frontier at coherent scores. Removes 15–26% of parameters touching NEITHER L2 (hidden stays 1024, all 29 readers bit-identical) nor L3 (zero body folds in the base arm).

**Math:** tied table 65,536x1024 counted once. K=32,768 -> 196,138,752 = 0.8539x; K=16,384 -> 179,361,536 = 0.7809x; K=8,192 -> 170,972,928 = 0.7444x — all standard-shape, loader-counted. Rung 2 (gated on rung 1 >= 8/15): vocab-16k + X13 excision ~167.3M = 0.7284x. Calibration: MXN2's 12,288-row verbatim head covered 80.99% of occurrences at 10/15 with a LOSSY tail; here the tail is exact chains of kept rows.

**Solve (FIX APPLIED — derivation closure):** (1) Token-occurrence census over the sealed corpus. (2) **kept = top-K-by-census ∪ ALL merge-tree ancestors of every kept token ∪ specials/chat-template tokens ∪ all 256 byte-fallback tokens**; only then restrict the BPE merge list to its closure (a merge survives iff product and both operands kept). This repairs the original spec's fatal flaw: BPE intermediates have near-zero surface frequency and an unclosed census silently destroys formation of kept tokens. If the tokenizer is Unigram: vocab restriction + Viterbi re-tokenization instead. (3) Recompute occurrence coverage POST-closure; choose census K so the POST-closure row count meets the parameter budget; publish both numbers. Receipt must prove, by re-tokenizing the sealed corpus, zero kept-token occurrences lost. (4) Gather kept rows verbatim into the new tied table, dense id remap, emit stock tokenizer.json. Base arm is zero-solve — the strongest identity gate in the campaign. (5) Optional arm B: final-norm gain diagonal re-solved by prior-anchored ridge against parent logits on kept tokens (the sealed 0.54 -> 0.17 solve), lambda on internal fit split.

**Storage:** K x 1024 rows verbatim (tied, counted once); everything else parent-identical. Merge table SHRINKS; remap lives in tokenizer.json (parent's tokenizer never counted; child's is a strict subset). Sequence length +5–15% at K=16,384 is compute, not parameters — print measured tokens-per-byte beside the ratio on every dossier.

**Export:** stock `Lfm2Config.vocab_size`, stock tokenizer file, no trust_remote_code. Labeled research-probe until A2 lands; frontier promotion only after Cyril rules.

**Pre-registered harness changes (must land BEFORE first frozen review, wave-18-style gated assertions):** per-child tokenizer acceptance; per-token perplexity columns replaced by bits-per-byte; token-match diagnostic prints N/A; lm-eval loglikelihood tasks remain valid (string-level sums).

**Kill gate:** closure-corrected K=16,384 child reads <= 4/15 -> the re-tokenization input-distribution shift dominates, the coverage law does not survive tokenization change, axis dead at every K worth having — before Cyril even needs to rule.

**First build:** the K=16,384 arm-A child + one frozen 15-prompt review. Report the per-case mix (L5 watch: rare-string lanes — coding-deduplicate, context-retrieval).

---

### S2. SEGBASIS — seam-gauged per-segment reader bases on hidden_size (zero added tensors)

**Wall:** L2 at its root — one global D->H projection serving 29 disagreeing readers (B01.ffn argmin 0.8801 in all seven sealed bases; blocks 0–3 under-served at every width; spread 0.065 rising at death). Payoff rungs: 736/720/704 where the global basis reads 1/0/0 of 15, targets 0.6726x/0.6561x — below CL896's 0.688431x at nonzero score.

**Math:** K depth segments at cut c; segment s keeps E_s = top-H eigenvectors of its pooled reader moment; within a segment, folds are the sealed wave-19 recipe verbatim (W_c = sqrt(D/H) W diag(g) E diag(1/g_c), g_c = diag(E^T diag(g) E)). Seam: exact orthogonal gauge Q_{s+1} by seam-weighted Procrustes (SVD of E_{s+1}^T M_seam E_s); identity at the cut costs 2−2cos(theta) vs the optimal transition's sin^2(theta) — same order, so the ~0.6–0.8M-param transition tensor buys only second-order correction. Exact absorption of a non-orthogonal T is impossible (residual trunk is one accumulator); the remaining absorber is the conditioned-local sequential ridge (parent's map on the child's OWN input — the blessed form, explicitly not trajectory targeting). Tied embedding: writer form E_1^T W_emb stored; tied-read moment folded into the last segment's Procrustes; final-norm diagonal solved closed-form.

**FIX 1 APPLIED — through-traffic pinned, knobless:** segment A's pool = the endpoint-equalized pool over ALL 29 endpoints, where each post-cut endpoint enters as its reader moment restricted to r_pre. No mixture scalar exists. Degenerates exactly to the sealed global objective when the segment is the whole model, which strengthens the self-test. E_B(c) = pool{M_i : i >= c} unchanged.

**Solve:** (1) P2 census J(c) over 13 cuts from sealed rows (residual identity, no new capture), per-endpoint printed; orders, never gates. (2) Build at the argmax cut with mx22_build.py lineage unchanged except per-segment E_s Q_s; conditioned-local sequential pass; lambda on the internal fit split (wave-18 law; L5 lever). (3) **Self-test:** at E_A = E_B the construction must reproduce the sealed global W-child bit-for-bit (Q = I, seam = I). (4) **FIX 2 APPLIED:** the T*-transition probe twin (ridge-optimal transition installed as a forward hook at the cut) is built and reviewed WITH child (a), not deferred — a null must be attributable (segment bases buy nothing vs the identity seam ate the gain). (5) Report directions-exchanged and per-case capability mix on every child. K=3 gated on K=2 signal (L3: one seam at a time).

**Build order:** (a) H=768, K=2 + T* twin vs ARCH.W768 3/15 at identical 165,978,624 params; (b) H=896, K=2 vs ARCH.W896 4/15 at 197,311,616; (c) if (a) or (b) >= control: strike H=720 (0.672632x) and H=704 (0.656116x) where the global arm is 0/15 — nothing to lose.

**Storage:** standard arm ZERO added tensors — byte-identical tensor sets and counts to the sealed W-children. Probe arm +H^2 per boundary (589,824 at 768), diagnostic only, charged in full if ever claimed.

**Export:** stock Lfm2ForCausalLM, hidden_size moved, head_dim=64 pinned — the sealed W-child export path. Probe arm research-only.

**Kill gate:** BOTH arms of build (a) read <= 3/15 with no per-case mix gain -> the depth-varying-basis lens is dead at the width with the largest measured reader disagreement; the 736/720 revival claim is struck without building it.

**Honest ceiling caveat (carried):** MM896 already cleared under-floor readers once and lost two cases — floor-clearing is necessary, not sufficient; only the build answers (L1).

---

### S3. ORTHO-PAIR — cross-error orthogonalization in the reader metric (census-first)

**Wall:** L3's additive-interference arithmetic: ||e_A + e_B||^2_M = ||e_A||^2_M + ||e_B||^2_M + 2<e_A,e_B>_M — the cross term has never been measured or controlled in 22 waves. Targets: murdered pair W896+X13A (4/15 + 13/15 -> 1/15) and living pair X07+KV6JQ (7/15).

**FIXES APPLIED:**
1. **Rule-3 repair:** rho census publishes either way and NEVER gates — the soft arm on the living pair is built regardless of census outcome. Census sets hard-vs-soft priority and target order only.
2. **Constraint form pinned:** primary form (i) — constraint at B's OWN write sites with A's error field measured THERE from the existing sealed solo captures. Fully legal, no linearization anywhere, no dependence on A1. Form (ii) (terminal-read via frozen-gate operators) is only permitted co-adjudicated with PAIR-KKT under A1.
3. **Affine constraint:** the evidence term leaves an offset — the constraint is C vec(W_B) = c0, and the bordered KKT [[X^T X + lambda I, C^T],[C, 0]][w; nu] = [X^T y; c0] must carry it, or the exact-retreat assertion fails on a correct implementation.
4. **X13A degeneracy stated:** with A = X13A (last block) the shared downstream endpoint set degenerates to the final read; the published error-spectrum of A decides hard vs soft arm (flat spectrum -> soft arm only, mu ladder, one linear system).

**Solve:** (1) P1 rho census (free). (2) W896+X13A: fold A = X13A (rank-by-solo, wave-18), fold B = width-896 writers re-solved under the constraint; basis frozen at plain pooled reader-PCA — L2's point is NOT touched, only the writer solve. X07+KV6JQ: A = X07, B = merge k/v/o ridge under constraint, relation phase gauge re-anchored (wave-17). (3) r at 80% cumulative energy of A's stacked reader-projected error, spectrum published, never gated. (4) lambda/mu on internal fit split; trust region veto-only. (5) Retreat assertion: constraint inactive (rho ~ 0) must return B's sealed solo tensors. (6) Frozen review, per-case mix, benchmarks. Two-body only.

**Storage:** zero delta — value-only inside sealed pair shapes (187,214,592 = 0.815064x; 212,910,848 = 0.926936x). Multipliers are solve-time objects.

**Export:** config-standard (both pairs already exist as sealed standard children).

**Kill gate:** if rho fails to separate murdered from at-better-component pairs, linear error superposition is falsified for one capture's price — itself the first sealed mechanism evidence on L3 (damage composes nonlinearly). The soft-arm child then still runs; if it also moves nothing, the lever is priced dead.

**First build:** P1 census (free), then one soft-arm child on X07+KV6JQC.

---

### S4. PAIR-KKT — one-shot coupled terminal solve on the living pair ARCH.X07KV6JQC

**Wall:** L3's cross-term at the solve level — the first single, non-alternating, joint objective over two fold classes' tensors in 652 rows (ALS was local+alternating; wave-15 harmful arms were per-writer trajectory-chasing; wave-17's joint solve was within one merge group; conditioning moves evidence only). Target: 7/15 at 0.926936x, 4 cases below its better component (X07 solo 11/15).

**Math:** freeze attention probabilities, SwiGLU gates, RMSNorm scalars per position from a conditioned capture through the sealed child (wave-17 precedent for the freeze class); final read becomes affine in each re-solvable tensor; drop bilinear cross-terms; solve ONE convex quadratic over both folds' free tensors jointly, prior-anchored at the sealed pair tensors, in the final-norm gain-weighted reader metric M. One bordered KKT system. lambda -> infinity reproduces the sealed 7/15 child in BYTES (identity gate and exact retreat).

**HARDENINGS APPLIED (binding):**
1. **A1 ruling lands before any contract-compliance claim** — a multi-block frozen-linearization design matrix is functionally a Jacobian; wave-17's one-block precedent does not automatically cover it. Pre-declared fallback: research probe by mechanism, artifact stays standard-shape.
2. **The "both sites sit in the L10–L12 regime" claim is struck as overreach** — X07's surviving writers and early attention sit where terminal linear prediction failed its R2 floor. Publish held-out R2 per tensor group and expect weakness for the shallow groups.
3. **P4 pre-read first** (prioritizer, never gate): if linearization cannot reproduce EXISTING composed damage on held-out rows, it cannot shape it.

**Solve:** conditioned capture (three standing calibrations); unknowns = exactly the tensors the pair's recipe already re-solves (KV6JQ merged k/v/o group; X07 excision-conditioned surviving FFN writers); J_k operators by forward products of sealed activations through frozen dense weights; two-eigh preconditioning (wave-17: unpreconditioned CG needed 20,000 iterations and a capped solver silently returns the prior); synthetic-identity verification of the assembled system; lambda ladder on internal split (every third fit batch withheld), trust region veto-only; byte-identity of the lambda->infinity rung asserted; adopt split-selected rung only; frozen review with per-case mix; expect L5 swap, report it.

**Storage:** zero delta — 212,910,848 params, no aux, no routes; the 355x route economics has no denominator here.

**Export:** config-standard (num_hidden_layers + layer_types with the causal-mask assertion re-run; standard head axes). Value-only motion.

**Kill gate:** child reads <= 7/15 -> the first sealed measurement that no closed-form coupling beats selection on a living pair; worst case re-seals the frontier pair unchanged. Either outcome is a law.

---

### S5. LP-ATLAS + ATTR-LAMBDA (merged) — the lambda-portfolio capability atlas

**Wall:** the quality axis at fixed storage — the lambda profile selects WHICH capabilities survive (L5, sealed 4x: 0/−1/−1/+1, same capability every time), and no wave has mapped the portfolio. Cyril's wanted lanes (conversation-natural-greeting 0.25, multi-turn-correction 0.10 campaign-wide) sit where default profiles treat as expendable.

**FIXES APPLIED:**
1. Per-class arm DROPPED — the sealed LS baseline is already per-writer at all 28 sites (build-receipt-MX21-W896LS.json: chosen lambdas {0.1:5, 1.0:21, 10.0:2}, none at grid boundary); a per-class arm would coarsen sealed practice.
2. Cost model corrected: the conditioned pipeline invalidates downstream cached Grams when an upstream lambda changes — **each profile is a full build pass**, not two matmuls. Budget honestly (below).
3. ATTR-LAMBDA absorbed: Stage A rescoped to the ONLY identified data — the case x block excision-survival matrix (X13A's 13 surviving FFN writers map 1:1 to blocks with clean presence/absence rows). Width-writer attribution is UNIDENTIFIED (no sealed row varies one width writer) and is dropped until single-writer perturbation children exist. s_w = sum_c p_c D(block_w, c) with p from A3.

**Staged build plan:**
- **Stage 1 (gate, 1 build):** W896 with check rows restricted to conversation/multi-turn evidence rows (wave-20 closed-form token rules; rows never from review prompts). If its held-case vector equals sealed W896LS's — case-targeted selection is dead; cancel ALL remaining restricted arms; only X13A {LS, forced protect-band} proceed (no lambda variation has ever run on the X13A base, so those stay informative).
- **Stage 2 (if the mix moves, 5 builds):** X13A x {per-writer LS, forced protect-band (selection overridden — blocks 0–3 lambda large, wave-18 move-almost-not-at-all regime), ATTR priority allocation, ATTR INVERTED allocation}; W896 x {instruction/retrieval-restricted}.
- **ATTR pre-registration:** if the case-mix delta between priority and inverted children does not track the allocation's sign (same held set or symmetric noise), the excision census does not transfer to lambda allocation even on its own axis — Stage A is noise, program dies in one wave.

**Solve:** cached per-site Gram eigendecomposition, prior-anchored ridge W(lambda) = (Y X^T + lambda W0) Q diag(1/(mu_i+lambda)) Q^T per site, rebuilt per profile in depth order (conditioned); admission per site = internal fit-split check with trust region veto-only; boundary-landing rule (widen until interior). Output artifact: the ATLAS (profile x 15-case hold vector + Guttman residual per profile — a direct measurement of the 10.4% non-ladder axis). Cyril's profile is a lookup, not an optimization; case mix, not score, is the selection criterion.

**Storage/export:** zero delta; bit-identical shapes to sealed baselines (217,629,440 = 0.947479x X13A; 197,311,616 = 0.859023x W896). Config-standard. Buys zero compression — buys the capability mix at compression already paid for, which is what the handoff needs (§2.12: "report which cases it keeps").

**Kill gate:** Stage-1 null (case vector identical to W896LS) collapses the atlas to coarse knobs already priced at ±1, and cancels 3 restricted builds before GPU. Side effect: a Stage-1 or Stage-2 swing >= 2 cases arms the CLIFF resurrection (see kill list).

---

### S6. 29-DIAGONAL RECALIBRATION — per-reader norm-gain refit on the width axis

**Wall:** L2's reader-disagreement structure, attacked in the one per-reader degree of freedom the architecture provides that no global subspace can express — the 28 block RMSNorm gains + final norm (only the final norm was ever re-solved; worth 3x in relative logit SSE, wave 19). Exchanges ZERO directions — outside wave-21's basis-invariance condemnation by construction.

**Math (NOTATION FIX APPLIED):** per endpoint i, solve diagonal d minimizing ||(N_c diag(d) − N_p diag(g) E) W_i^T||_F^2 with normal equations **(N_c^T N_c) ⊙ (W_i^T W_i) d = rhs** — the Hadamard is with the H x H reader-COLUMN Gram (W_i is out x H), one SPD Cholesky per endpoint, 29 endpoints, seconds of compute. Reader Grams: q/k/v concat (attn operator_norms), w1/w3 (ffn_norms), in_proj (conv operator_norms), lm_head (final). Target = parent's LOCAL post-norm reader input transported through E (conditioned-local family, not trajectory targeting). Prior-anchored at the gauge-transported g_c; lambda on internal split, trust-region veto-only. Identity gate: at E = I, child == parent, the solve returns g bit-exactly.

**Build order (GATING FIX APPLIED):** W768R FIRST — the rung with the campaign's richest identical-parameter control set (W768 3/15, W768LS 3/15, W768ST 2/15, three dead basis arms, all at 165,978,624). W896R and W736R only if W768R shows >= +1 case or a sealed case-swap. The W896LS+recal cross (fourth build) gated the same way — wave 22 already showed solver dials failing to add across mechanisms.

**Storage/export:** zero delta (overwrites 29 existing diagonals, 30,464 params already counted); stock tensors, standard export; solver-refinement arm of an existing axis, not an L3 step.

**Kill gate:** W768R <= 3/15 AND no case swap (context-retrieval stays dead per the §2.12 signature) -> the 28-diagonal family is behaviorally inert like every above-floor proxy; remaining rungs cancelled. The null still pays: eliminates gain-shape mismatch, leaving early-block residual demand (S8's question) as the standing suspect. Honest expectation 0 ± 1 case; MM896 (improved the same proxy 0.1752 -> 0.1513, lost two cases) is the standing caution — but the diagonal refit is not zero-sum across readers.

---

### S7. NKP-VANLOAN — nearest-Kronecker pricing wave on the SwiGLU stack (+ Monarch control)

**Wall:** none re-fought — converts frontier-report §6's named open backlog ("highest-leverage unplayed class by storage arithmetic alone") into priced rows. Rearrangement rank and matrix rank are independent: a flat channel-space spectrum (the property that kills every low-rank lane) says nothing about the never-measured rearrangement spectrum.

**Math:** Van Loan–Pitsianis: min ||W − sum_t A_t (x) B_t||_F = truncated SVD of the rearranged R(W) — closed form from the checkpoint alone. w1 (2560x1024), grid (40x32)(x)(64x32): 3,328 params/term, r = 196 at the 4x bound (top 196 of 1280 rearrangement directions). Evidence enters through SELECTION only (grid/r/site by evidence-weighted error tr[(W−What) M_x (W−What)^T], M_x = sealed ffn-input moment) — the solve stays pure SVD. Monarch control: exact blockwise-SVD projection, identical census.

**Solve:** (1) P3 census (~1 CPU-hr, no GPU, no capture); publish beside the channel-space spectra; orders, never gates. (2) Single sealed cell at the S4 bound on w1@L05 (safest body-fold site) vs the sealed rank-truncation twin at same site and matched storage; full 16-lane surface PLUS benchmark row (wave-20: one-site reviews are below the instrument's resolution). (3) Lane-set ladder (one lane per block, MLP side only, one class) ONLY if the cell wins. (4) fp64 reconstruct-and-compare identity gate; per-site evidence-weighted error in the receipt. 4x-eligibility predicate (correction #11) on any claim.

**Storage:** factor pairs counted at r(m1 n1 + m2 n2); no runtime indices. Installed parent-shaped — every ratio is a STORAGE-FORMAT ratio, labeled, competing on the parent-shaped column (MXO3 0.729103x at 8/15).

**Export:** research probe under R3 permanently — no stock config axis, no absorption path to standard shape.

**Kill gate:** flat rearrangement spectra at the 4x budget + a loss to the twin -> class buried in one day, mirroring the block-Toeplitz grave. A priced grave is a deliverable; an unpriced backlog is not.

---

### S8. EARLY-SPARED ASYMMETRIC WIDTH — partial-depth width probe (confound-corrected)

**Wall:** WHERE L2 lives. The census indicts blocks 0–3 at every width (B01.ffn argmin in all seven bases; blocks 6–9 ~0.99); every failed rescue taxed good readers inside a fixed budget; sparing the early readers outright is the untested categorical alternative. L4 independently marks the same blocks (first-of-kind criticality).

**FIX APPLIED — the confound is closed:** as originally specced the design spared NINE readers (the final-norm/lm_head read was silently unserved), making a YES unattributable (early-blocks vs interface). Corrected design: **the final-norm read is SERVED — project the pre-final-norm stream through the same E** (parent-shaped runtime permits it; embedding write stays full-width so out-of-E embedding content feeds only blocks 0–3). Spared set = exactly the 8 indicted early block readers + the embedding write. If the projection is awkward, the fallback is the third arm: interface-only narrowing, so the triplet {spared-early, inverted, interface-only} decomposes the wall into early-block vs interface vs late-block shares in one wave.

**Math/solve:** runtime residual stays 1024; blocks 4–13 read/write a shared k-dim subspace (k=896 arm first; k=768 if informative). E = top-k eigenvectors of the pooled reader moment over the SERVED endpoints (20 block endpoints + final read) — uniform service over the served set, same estimator, not a seventh objective; the receipt must nonetheless **publish the direction-exchange count vs the sealed plain 29-endpoint basis** (wave-20/21 table format) so review sees how far E moved rather than litigating in prose. Readers fold as W diag(g) E with standard gauge, writers as E^T W, conv channels coreset channelwise-gated (wave-19 law); blocks 0–3 bit-identical to parent; prior-anchored ridge, internal-split lambda, conditioned capture; E = I bit-exact gate. **Inverted mirror control pre-registered** (narrow blocks 0–3 only): if the wall is early-local it must die immediately.

**Storage/export:** parent-shaped storage-format fold (~0.94x at k=896, every index counted, §1.3 convention); **research probe ONLY** — per-block residual width is not expressible in stock config. Budgeted explicitly as a knowledge-buy against ship-to-done: it buys one law, not ratio. The law is load-bearing either way: YES localizes the wall, licenses S6's diagonals aimed at blocks 0–3, and explains L2+L4 with one mechanism; NO kills the last cheap structural explanation and stops all future early-block rescue spending.

**Kill gate (pre-registered binary):** corrected spared-early child at k=896 reads ~4/15 or below vs sealed W896's 4/15, with early readers at retention exactly 1.0 -> the wall is not localized in the early readers.

---

## 4. KILL LIST

- **FFN-CORESET** (depth-varying-basis lens) — KILLED: sealed duplicate founded on a false reading of the record; ARCH.CL896 (mx15_build.py line 849: pivoted selection; line 889: prior-anchored ridge on w2; conditioned-local) IS the proposed recipe, so the "challenger" is selection vs itself, and solver micro-variants at fixed shape are twice priced at <= 1 case. Salvage rider only: the column-normalized ranking arm may ride as a one-cell probe on another wave's GPU time.
- **ABSORB-THEN-FOLD** (composition-error-shaping lens) — KILLED: its core cell is sealed 0/15 twice (X05T, CT640); the "L4 existence proof" is false (wave-15 §5: excision local re-solve is an identical no-op, update_scale 0.029); the written target is the trajectory form the wave-15 law prices as harmful; unflagged block-conflict landmine with KV6JQ. Resurrection only as an honestly-labeled "X05T re-run under wave-18 lambda-split with a gap target," baselined against a sealed 0/15 — not proposed here.
- **CLIFF-LAMBDA** (capability-portfolio lens) — KILLED: the falsifier already exists on disk — build-receipt-MX21-W736LS.json shows per-writer, out-of-sample, interior-converged lambda selection was already installed at the cliff and read 0/15 (a sealed −1); the reader-floor protect-band is contradicted twice (MM768/MM896); the consolation datum was published in wave 21. **Resurrection pre-registered: if LP-ATLAS restricted arms swing >= 2 cases at 896 or X13A, ONE W736 case-restricted child (no protect-band, no W720) is a justified follow-up.**
- **ATTR-LAMBDA (standalone)** — MERGED, not killed: width-axis Stage A is unidentified (no sealed row varies one width writer); the identified X13A half survives as two arms inside the LP-ATLAS wave (S5).

---

## 5. SHARED DISCIPLINE RIDERS (every child, every wave)

1. **L1:** censuses and pre-reads ORDER, never gate; every funded child is built and frozen-reviewed (16-lane surface, 15 prompts).
2. **L5:** per-case capability mix reported on every child — the score is never the only reading; lambda selected on an internal split of fit rows (wave-18 law), trust region veto-not-vote; grid-boundary landings = not converged, widen until interior.
3. **Identity gates in bytes, not tolerances** (wave-19): SEGBASIS E_A=E_B; PAIR-KKT lambda->infinity; ORTHO inactive-constraint retreat (with the c0 offset); 29-DIAG E=I; VOCAB zero-solve row identity; NKP fp64 reconstruct; EARLY-SPARED E=I.
4. **Per-endpoint printing, never pooled-only** (wave-21); directions-exchanged tables wherever a basis differs from the sealed plain pooled reader-PCA.
5. **Benchmark row on every single-site or single-child cell** (wave-20: one-site reviews are below instrument resolution); 4x-class eligibility predicate stated with any frontier claim (correction #11).
6. **Storage honesty:** all aux counted; storage-format vs standard-shape ratios labeled on every dossier; VOCAB dossiers additionally print tokens-per-byte.
7. **L3:** no wave adds a second composition step beyond what its spec declares; gated escalations (SEGBASIS K=3, VOCAB rung 2, ORTHO hard arm, 29-DIAG rungs 2–4) fire only on their pre-registered signals.

---

## 6. BUDGET

| Wave | Builds+reviews (gated minimum -> worst case) |
|------|---------------------------------------------|
| S1 VOCAB | 1 -> 2 (rung 2 gated on >= 8/15) |
| S2 SEGBASIS | 2 (child a + T* twin) -> 5 (b, then 720/704 strikes) |
| S3 ORTHO | census free + 1 -> 3 (hard arm, W896X13A target) |
| S4 PAIR-KKT | pre-read free + 1 |
| S5 LP-ATLAS+ATTR | 1 (Stage-1 gate) -> 6 |
| S6 29-DIAG | 1 (W768R) -> 4 |
| S7 NKP | census free + 1 -> 2 (Monarch cell) |
| S8 EARLY-SPARED | 2 -> 3 (interface-only arm) |
| **Total** | **9 minimum -> 26 worst case; all six free pre-reads dispatch day one** |

Dispatch order: P1–P6 immediately (CPU, parallel); builds in rank order S1 -> S8, with S3/S4 co-scheduled on the shared X07KV6JQC capture and A1–A3 put to Cyril before any frontier or contract-compliance claim is written.