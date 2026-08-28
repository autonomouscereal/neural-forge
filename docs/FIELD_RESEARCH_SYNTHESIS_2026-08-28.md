# Field Synthesis: Fold-Only Compression Campaign — 2025/2026 Literature Sweep

Deduped from 45 raw findings (EoRA appeared 2x, ReplaceMe 2x, the LG error-propagation paper 3x; the LinearPatch/Prune&Comp diagonal-repair family merged into one cell). Ranked by contract-fit x novelty-vs-sealed-record x leverage against the named walls: **additive interference (AI)**, **interface-x-body multiplication (IxB)**, **emission collapse (EC)** — the sharpest tax — the **reader-projected 0.92 energy floor (FLOOR)**, the **sub-0.69x chasm (CHASM)**, and **solver-criteria-cannot-rank (RANK)**.

Entries tagged `[EMISSION-COLLAPSE ATTACK]` directly target the repetition->echo->empty pathology.

---

## 1. FOLDING-APPLICABLE TECHNIQUES

### Tier 1 — run first (clean contract, cheap, hits a named wall dead-center)

**1. Evidence-refresh solving (propagate-then-solve + re-rank between folds).** Contract: clean. Walls: AI, RANK.
Four independent 2025-26 sources converge on the same finding: solving fold k against the PARENT's cached activations is measurably wrong once folds 1..k-1 exist (NMSE up to 0.27 by late layers), and importance rankings computed pre-fold go stale (importance drift). A chunk of our "additive interference" may literally be stale-evidence error, removable for free.
*Cell:* order folds front-to-back; after each accepted fold, one forward pass regenerates downstream evidence caches through the folded prefix; solve the next fold against propagated activations; re-rank the remaining fold menu each round. Version evidence caches per fold-state, never reuse across states. Testable at exact-4x: does refresh reduce the measured interference coefficient?
*Source:* arXiv:2608.08506 (LG, Aug 2026); same lesson independently in Prune&Comp arXiv:2507.18212, Nemotron Nano 2 arXiv:2508.14444 (iterative re-scoring), LACE-SVD arXiv:2607.03057.

**2. Sink-channel / massive-activation carve-out + targeted diagonal repair.** `[EMISSION-COLLAPSE ATTACK]` Contract: clean. Walls: EC, CHASM.
2025-26 mechanistic work establishes that attention sinks and massive-activation channels are the model's emission-regulation machinery (when/how strongly heads write into the residual stream) — not knowledge storage. Folds that perturb those few outlier channels break generation while bulk-channel knowledge is intact: plausibly a large fraction of our chasm and the best published account of why repetition precedes knowledge loss.
*Cell:* from cached activations, detect massive-activation dims (top-magnitude, trivial) and sink-dominant heads (first-token attention mass); (a) exempt those rows/dims from width-fold rank reduction (they are few — near-zero cost); (b) post-fold, ridge-solve a diagonal/low-rank output calibration ONLY on those channels against parent cached values. Falsifiable prediction: sink-protected folds show later emission collapse at equal x-factor.
*Source:* arXiv:2505.06708 (NeurIPS 2025 Oral, Qwen gated attention); arXiv:2601.22966; arXiv:2603.05498.

**3. Multi-axis phase-transition budget allocator.** Contract: clean. Walls: CHASM (head-on).
Survey+experiments over 30+ methods: each compression axis (structured prune ~30-45%, low-rank ~17-30% reduction, unstructured ~55-65%) has its own critical threshold with power-law-then-exponential collapse beyond it — and thresholds stay stable under combination, so stacking orthogonal axes each below its own PTP compounds safely. Their low-rank PTP band brackets our frontier (0.795x fine, 0.742x marginal, <0.69x chasm) almost exactly: independent evidence the chasm is a per-axis phase transition, not a solver deficiency.
*Cell:* stop deepening one axis. Fit the piecewise curve per axis (vocab / depth / width) from cheap cached-evidence sweeps; a budget-allocator picks the split hitting the target ratio while keeping every axis pre-transition.
*Source:* npj Artificial Intelligence, Feb 2026 (s44387-026-00072-8); code github.com/whucs21Mzy/Model-Phase-Transitions.

**4. LACE-SVD interpolated-target shrinkage + acceptance gate.** Contract: clean. Walls: AI.
Complement to #1 when full refresh still overfits: solve each fold's output factor against a blended target T = M_folded + a(M_parent - M_folded), a~0.7 — closed-form shrinkage that stops the fold from chasing a target the degraded input stream can no longer reach — and keep the solve only if a frozen-evidence fidelity check improves (their acceptance gate maps 1:1 onto our behavior-gated grading).
*Cell:* one-line change to existing output-factor solves; A/B on the 0.742x lineage.
*Source:* arXiv:2607.03057 (Jul 2026). LLaMA-7B at 0.6 ratio: PPL 32.57 vs SVD-LLM 53.74.

**5. EoRA activation-eigenspace-weighted SVD as the fold solver.** `[EMISSION-COLLAPSE ATTACK]` (evidence: disproportionate generative-task recovery). Contract: clean when absorbed. Walls: FLOOR, EC.
Eigendecompose the calibration Gram XX^T = QLQ^T, project with Q' = Q sqrt(L), SVD in that space, map back — provably minimizes OUTPUT-space error where plain weight-space SVD does not; rank stops being wasted on low-eigenvalue directions. Its headline recoveries are generative/multi-step (+11.8% GSM8K), i.e., output-space weighting attacks exactly the failure mode that dies first in our children.
*Cell:* (a) swap the truncation metric inside width/rank folds to activation-covariance-whitened coordinates (zero param cost — may push the 0.92 floor down); (b) optional rank-r compensation of dW = W_parent - W_fold, absorbed into fold factors where shapes allow, else budgeted in the ledger.
*Source:* arXiv:2410.21271v3 (NVlabs, Feb 2025 rev.); github.com/NVlabs/EoRA.

**6. Logit-space / reader-metric solving for reader-adjacent folds.** `[EMISSION-COLLAPSE ATTACK]` Contract: needs light adaptation. Walls: EC, FLOOR, RANK.
Two convergent results. LFQ (ICML 2026): block-wise activation-MSE calibration preserves MMLU but kills long-form generation, with a worked example where lower reconstruction error still flips the argmax — an independent derivation of our verbatim-beats-reconstruction and solver-cannot-rank laws; their fix optimizes the FINAL block against parent logits. IO-SVD: two-sided whitening with output metric C = E[J^T H J] (softmax Hessian) — the field independently inventing our reader-projection law as a full metric; PPL 27.7 vs 53.7 for SVD-LLM at 40% retention.
*Cell:* for the last block and any fold adjacent to the reader, solve in logit space — ridge against W_head·h_parent targets, optionally Fisher/softmax-Hessian-weighted as the closed-form CE surrogate; build C forward-only from actual downstream reader weights (no Jacobians needed). Also lift IO-SVD's greedy budgeted rank allocator with dense-crossover accounting. Prediction: logit-space final-block folds move the 0.795x frontier more than equal-budget mid-stack work.
*Source:* LFQ arXiv:2605.29756 (ICML 2026); IO-SVD arXiv:2605.15626 (May 2026).

**7. Decision-margin phase analyzer (fold-forbidden map + distributional-collapse early warning).** `[EMISSION-COLLAPSE ATTACK]` (early-warning component). Contract: clean. Walls: CHASM, EC.
Layer-pruned models have a Silent Phase then an abrupt Decisive Phase; transition depth predicts pruning robustness at r = -0.96; ablating transition-boundary layers is catastrophic while deep Decisive-Phase layers are nearly free; SFT cannot rescue models pruned past structural margin depth — matching our "no healing will save it" reality. Distributional collapse (one option dominating regardless of input) is measurable from cached logits BEFORE visible repetition.
*Cell:* pre-fold analyzer: logit-lens DM(l)/OF(l) profiles from cached hiddens through the frozen reader; mark boundary-adjacent layers fold-forbidden, deep Decisive layers as the cheap menu; add OF-collapse as an automatic early-warning gate ahead of frozen review.
*Source:* arXiv:2605.07271 (May 2026).

**8. CoCurve co-pruning curvature: joint combination scoring.** Contract: clean. Walls: AI (its closed-form theory), RANK.
Second-order expansion of pruning risk under KL yields a quadratic form whose OFF-diagonal entries are exactly the interference between removal decisions — recovered as a Gram matrix of single-unit ablation features from M forward passes, no gradients, no pairwise sweeps. Selection is one budgeted QP over attention units + FFN groups sharing a budget pool.
*Cell:* per-candidate ablation passes on the evidence set -> interaction Gram -> score fold COMBINATIONS by the quadratic form instead of ranking children independently; optional ridge compensation from precomputed edge blocks. Practical at 350M scale.
*Source:* arXiv:2607.17568 (Jul 2026). Up to 14.5x lower PPL than best baseline at 40% on Mistral-Small-24B.

### Tier 2 — new cell families and repairs

**9. ReplaceMe depth fold: block deletion + ridge linear replacement fused into W_down.** Contract: clean (LS variant only). Walls: IxB (sidesteps it structurally).
Delete n contiguous blocks; solve T = argmin ||M_i T - (L_{i+n} - Y_i)||_F on three cached tensors; fuse T into the preceding block's down-projection. Zero added params, standard checkpoint, ~25% depth on LLaMA-2-7B keeps ~92.5% with no training. Key structural point: the correction lives on the MLP branch — the residual identity path is never re-projected, so the interface itself is never damaged.
*Cell:* window selection by our behavioral criteria + DM map (#7); cache (M_i, Y_i, L_{i+n}); ridge-solve; fuse; delete. The cosine-objective variant needs Adam — do not port.
*Source:* arXiv:2505.02819 (NeurIPS 2025), code released.

**10. Diagonal interface hygiene (channel-scale compensation absorbed into weights).** Contract: clean, free. Walls: IxB.
Two papers agree most layer-removal damage is per-channel magnitude mismatch at the seam. The diagonal fix is closed-form from cached statistics and absorbs exactly into RMSNorm gamma / next projections — a genuinely zero-param repair, and it quantifies the interface component of IxB as a scale problem (outlier channels may also explain part of the 0.92 floor).
*Cell:* default post-fold step for every depth cell: moment-match per-channel scales on cached activations, fold into adjacent weights, re-measure (per #1) before the next fold.
*Source:* LinearPatch arXiv:2505.24680 (diagonal-only reduction); Prune&Comp arXiv:2507.18212.

**11. Norm-conserving interface projection (mHC/Sinkhorn).** Contract: clean, parameter-neutral. Walls: IxB, AI (at the source).
DeepSeek's mHC diagnosis from the training side matches ours from the fold side: unconstrained residual mixing maps destroy signal conservation. Sinkhorn-Knopp projection onto doubly-stochastic matrices is gradient-free iterated normalization; orthogonal Procrustes is already in our kit.
*Cell:* after ridge/Procrustes-solving fold interface matrices (readers/writers between folded and retained widths), project onto a norm-conserving manifold. Hypothesis: small reconstruction-error cost, large downstream emission-damage reduction.
*Source:* mHC arXiv:2512.24880; accelerated Birkhoff projection arXiv:2606.07574; Qwen Gated Residual (Qwen3.8-Flash-Next card).

**12. Nested-rank SVD behavioral ladder + matryoshka-ness spectrum probe.** Contract: clean, essentially free. Walls: RANK, FLOOR, CHASM diagnosis.
SVD is inherently matryoshka: one rank-K solve contains every rank-k truncation optimally. And MatFormer/EmbeddingGemma prove prefix-truncatability is trained-in, not free — flat-spectrum frozen parents ARE the chasm.
*Cell:* (a) per fold site, run frozen review at several nested ranks from a single solve to locate the behavioral knee (where emission collapse begins) at zero extra solver cost; (b) pre-solve probe: eigen-solve reader-projected activation covariance per site — spectrum decay predicts width-fold headroom against the 0.92 floor BEFORE spending a solve.
*Source:* EmbeddingGemma (Google, Sep 2025); MatFormer/Gemma 3n docs; Gemma 4 TR arXiv:2607.02770.

**13. Unembedding rank-repair (demotion correction).** `[EMISSION-COLLAPSE ATTACK]` Contract: clean core; verify expressiveness. Walls: EC.
The Benchmark Illusion: pruned models keep MC accuracy on questions they can no longer answer generatively — correct answers are DEMOTED in ranking, not erased (they return under alternative decoding). PruneCD shows the parent-vs-pruned logit delta is literally "what was removed," recoverable from logits alone.
*Cell:* cache parent and child logits on frozen evidence; solve (a) cheap tier: diagonal scale+bias per vocab row (~2|V| params, effectively free) restoring parent token ranking via weighted LS on logit gaps; (b) richer tier: low-rank map C ~ argmin sum||C·h_child - (z_parent - z_child)|| absorbed into the unembedding, rank budgeted inside net reduction. Expect partial recovery (context-dependent component is unrecoverable statically); grade on whether it moves the loops->echo->empty ladder even when benchmarks are flat.
*Source:* arXiv:2606.17609 (Jun 2026); PruneCD arXiv:2509.16598.

**14. Anti-loop cauterization (param-reducing micro-fold).** `[EMISSION-COLLAPSE ATTACK]` Contract: needs adaptation (identification only). Walls: EC stage 1 specifically.
Repetition is a LOCALIZED mechanism: identified neurons/SAE features causally induce loops; zeroing them fixes looping while approximately preserving tasks. Published identification uses SAEs/gradients — substitute closed-form: our archive has looping vs healthy child transcripts; compute the repetition direction as difference-of-means (or top PC) of cached hiddens at loop onset vs healthy generation; rank neurons by alignment.
*Cell:* for a folded child that regressed into loops: permanently zero the implicated rows of up/gate and columns of down_proj (exportable, REMOVES params), or project the layer output W' = (I - b·dd^T)W. Grade behaviorally.
*Source:* arXiv:2504.14218 (Repeat Curse); arXiv:2606.13705 (single-neuron repetition).

**15. Entropy-trajectory gate + scalar temperature absorption.** `[EMISSION-COLLAPSE ATTACK]` Contract: clean extracts. Walls: EC (leading indicator).
Repetition = entropy lock-in; rambling = entropy drift — generation health is a trajectory-level distributional property.
*Cell:* (a) diagnostic: entropy-profile divergence of child vs parent on cached evidence — likely fires before any review score moves; (b) repair: solve one scalar matching mean child entropy to parent (1-D bisection on cached logits), absorb into final norm gain or lm_head. Bolt onto every fold; costs nothing.
*Source:* arXiv:2601.01714 (Jan 2026).

**16. Parent-as-verifier acceptance-rate gate.** `[EMISSION-COLLAPSE ATTACK]` (as meter). Contract: clean. Walls: EC detection, RANK.
Run the frozen parent as a speculative-decoding verifier over the folded child's continuations; token acceptance rate is an automatic generative-coherence meter — should catch repetition->echo->empty earlier and more sensitively than perplexity, zero training, zero review cost.
*Source:* Gemma 4 MTP machinery, arXiv:2607.02770; vLLM PR 41745.

**17. Parent-agreement triage on a pinned token cache.** Contract: clean. Walls: RANK (operationalizes behavior-first at sweep scale).
MiniPuzzle cut 400 candidate architectures to 3 using greedy next-token PARENT AGREEMENT on 1M tokens — one forward pass per child. Their Table 6 independently confirms our collapse ordering (MMLU 84.2->81.5 while GSM8K 91.4->60.7, code 67->23).
*Cell:* standard pre-review triage so 4x sweeps stop spending frozen-review budget on dead children.
*Source:* arXiv:2504.03624 (Nemotron-H / MiniPuzzle).

**18. Procrustes K->V tying: delete W_v where V is reconstructable from K.** Contract: clean solve; novel fold family. Walls: none directly — new param-cut axis for the multi-axis allocator (#3).
Gemma 4 reuses keys AS values on global layers at no reported quality cost — the value subspace is largely a linear image of the key subspace in trained attention.
*Cell:* per attention layer, solve Procrustes/ridge R with V ~ R·K on cached pairs; where residual is tight, export W_v' = R·W_k (delete W_v — ~1/4 of that layer's attention projection params). Rank layers by residual; schedule inside the shared attention interference budget.
*Source:* Gemma 4 TR arXiv:2607.02770.

**19. Attention removal + exact FFN fusion + single-seam repair.** Contract: needs adaptation. Walls: IxB (collapses k interfaces into one).
Pairwise block-dependency matrix (cosine distance of block j's contribution with/without block i in its input — pure cached forwards) finds effectively-parallel runs; delete attention there (the param cut, scored Puzzletron-style); fuse the leftover k sequential FFNs into ONE exactly (concatenation theorem); ridge-repair one seam instead of k. Their finding that the LAST block of a run is the fusion-sensitive one tells you which member to repair hardest.
*Source:* FFN Fusion arXiv:2503.18908; Puzzletron attention->no_op scoring (Model-Optimizer, May 2026).

**20. Clustered LM head + per-cluster low-rank factorization.** Contract: needs adaptation. Walls: extends the vocab frontier (our champion axis).
Gemma 4 drafters replace the d x 262k projection with cluster routing + within-cluster logits (d x 4,096 matmul). Spherical k-means over head rows is gradient-free on cached evidence; the param cut comes from SVD-factoring each cluster's row-block independently (semantically coherent clusters should be far lower-rank than the global head), ridge-repaired against cached logits.
*Source:* Gemma 4 TR arXiv:2607.02770; Google MTP blog. Corroborating: Fast Gemma Challenge competitors independently identified the LM-head path as the small-Gemma bottleneck.

**21. Saliency atlas + logit-KLD depth ranking.** Contract: clean (scoring only). Walls: RANK.
Minitron-SSM's exact one-shot formulas for every prunable axis (head channels, heads, FFN neurons, embedding dims, layers) — all computable from our caches. Notably NVIDIA chose logit-KL (reader-side) rather than activation-MSE for DEPTH decisions — consistent with our misranking law. Embedding-channel importance summed across ALL layers is a ready-made width-fold criterion vs the 0.92 floor. Their group-permutation constraint generalizes: folds that permute structured sub-units must respect broadcast groups.
*Source:* arXiv:2504.11409 (Minitron-SSM); axis-ordering prior (depth -> FFN width -> embedding; skip SSM heads at mild ratios) from Nemotron Nano 2 arXiv:2508.14444.

**22. MIP assembly of fold schedules.** Contract: clean. Walls: AI (exploits its additivity), RANK.
Puzzletron's training-free path: per-layer variant library, one-shot activation scoring, MIP picks one variant per layer under param/memory constraints. Our additive-interference law is precisely the assumption that makes per-block additive costs + MIP valid — and CoCurve (#8) supplies the off-diagonal correction when additivity bends.
*Cell:* encode each fold candidate (site x intensity) with measured behavioral cost; MIP assembles the mix hitting a target ratio instead of hand-picked schedules.
*Source:* Puzzletron README (NVIDIA Model-Optimizer, May 2026); Puzzle arXiv:2411.19146 (ICML 2025).

**23. OBS inverse-Hessian compensation vs ridge re-solve: head-to-head.** Contract: needs adaptation. Walls: AI (may distribute damage more evenly).
When a fold deletes columns/neurons/heads, the SparseGPT-style closed-form update perturbs SURVIVING weights minimally via (XX^T + lambda·I)^{-1} instead of re-fitting them — a different repair family from ridge, consuming Grams we already cache.
*Cell:* A/B against ridge repair on the 0.742x lineage; hypothesis: OBS delays the interference blowup.
*Source:* Model-Optimizer sparsity guide (2025-26 releases). The shipped 2:4 pattern itself is dead (see kill list).

**24. Nested-budget joint ridge (MatQuant calibration math, structural transplant).** Contract: adaptation of quantization math (allowed per standing rule). Walls: AI (from the other side).
MatQuant's +10% at int2 comes from optimizing the INTERACTION between nested precision levels rather than each alone.
*Cell:* stacked/weighted ridge across interacting fold sites where the aggressive-budget solution is constrained to be a prefix/subspace of the mild-budget solution — keeping deep folds on the manifold shallow folds define. Candidate mechanism to get below 0.742x with no new cell types.
*Source:* arXiv:2502.06786 (ICLR 2025 oral).

**25. Rotary-pair-preserving block-structured QK solves.** `[EMISSION-COLLAPSE ATTACK]` (plausible hidden contributor). Contract: clean constraint. Walls: EC-adjacent damage in attention folds.
GLM-5.3 (NoPE sparse layers) and Qwen3.8 (RoPE on 64 of 256 dims) agree most QK dims carry content, not position. Generic SVD of W_Q/W_K smears positional phase across content dims — corrupting position before content, surfacing as coherence loss.
*Cell:* split cached QK evidence into rotary-carrying and content dim blocks; preserve rotary pairs exactly; SVD/ridge only the content subspace.
*Source:* GLM-5.3-Flash deep-dives; Qwen3.8-Flash-Next card; lmsys blog 2026-08-26.

**26. Frequency-stratified vocab-fold gates + vocab-body interference measurement.** Contract: clean (protocol). Walls: protects the champion cell; verbatim/coverage law.
Google's 270M bet (63% of params in a 256k-vocab embedding, rationale: rare-token capacity) predicts vocab folds fail first on rare-token verbatim recall — invisible to aggregate metrics head tokens dominate.
*Cell:* gate vocab cells with rare-token-stratified verbatim probes; separately, explicitly measure vocab-x-body interference — embedding rows are row-separable, so it should be ~zero; any nonzero term localizes where IxB enters through the tied head/reader.
*Source:* Gemma 3 270M blog (Aug 2025).

**27. Grading-protocol upgrades: state-tracking probes + blind private split.** Contract: clean. Walls: RANK, EC (separability).
(a) Emission collapse and algorithmic state-tracking collapse are SEPARABLE axes with different layer signatures — a fold can pass coherence review while silently losing computation; add a tiny arithmetic/bracket-matching battery to the 4x/component review. (b) The Fast Gemma Challenge's private-blind-set gate caught a self-reported-clean but degraded submission — adopt a per-campaign frozen-review split that solver-side tuning never touches.
*Source:* arXiv:2602.01997 (Limits of Layer Pruning); Fast Gemma Challenge (HF x GDM, Jun-Jul 2026).

**28. Nested-parent campaign: fold Gemma-4 E4B to isolate the chasm's cause.** Contract: clean on this parent. Walls: CHASM (attribution experiment).
Mix-n-Match extraction (per-layer FFN width selection + layer skipping, standard-checkpoint export, zero gradients) IS a fold cell — but only works because the parent was nested-trained; this externally explains our 0.92 floor on ordinary parents. E4B is a legitimately nested parent.
*Cell:* run the fold pipeline (slice + ridge seam repair) on E4B; deep sub-0.69x success there would prove the chasm is parent-conditioning, not fold mechanics — the single most informative strategic experiment on this list.
*Source:* Gemma 3n/MatFormer docs; Gemma 4 TR arXiv:2607.02770.

### Tier 3 — exploratory / smaller

**29. Cross-layer weight tying via block-ridge shared solve.** GLM's IndexShare/IndexPool prove attention-adjacent score maps tolerate 4-way sharing nearly free. Cell: for k adjacent layers, solve one shared W* = argmin sum_i ||X_i W* - X_i W_i||^2 (+ per-layer diagonal rescales); param reduction requires shared-tensor export or treats it as prep for a layer-merge fold. Priority: Q/K first, V/O second, FFN last. *Source:* arXiv:2603.12201; GLM-5 repo. Contract: needs adaptation.

**30. Sibling fold-child merge (RegMean with real Grams).** If two children of one parent preserve different capabilities, covariance-weighted closed-form merge of surviving weights may yield one child with both — use our cached activation Grams (better than ACE's data-free Fisher proxy). Unproven on decoder LLMs; one exploratory cell, not a lane. *Source:* arXiv:2603.02945; RegMean lineage. Contract: needs adaptation.

**31. 1-gram embedding repair after vocab folds (Engram degenerate case).** Row values = conditional-mean/ridge of parent-child hidden residuals per token over the calibration corpus, folded into the existing embedding matrix. Fully standard export, squarely in lane, small. *Source:* Memory Grafting arXiv:2605.20948; Qwen3.8 Engram lineage. Contract: clean.

**32. Anti-symmetry interface diagnostic (Ghosted Layers).** The optimal full-space boundary operator after layer removal has substantial anti-symmetric components that symmetric patches cannot express. Cell: solve full-space W = I + X_pre^† Δ on boundaries, split into the branch-absorbable part (into W_down, ReplaceMe-style), and measure whether IxB damage correlates with anti-symmetric energy of the ideal correction. Naive deployment (dense CxC hook) is dead — see kill list. *Source:* arXiv:2605.15491. Contract: diagnostic clean.

**33. Ridge-fit linear mixer replacing mid-ranked attention layers.** Jet-Nemotron's validated decomposition (knowledge in MLPs; attention exchangeable; only a few attention layers load-bearing) suggests: rank attention layers by removability; delete bottom-ranked (per #19/#22); for mid-ranked, ridge-fit a cheaper linear token-mixing map on cached (pre-attn, attn-out) pairs with MLPs frozen — param cut vs full QKV+O. Speculative: post-hoc attention surgery is exactly what IxB punishes; one exploratory cell. Their MLP-freezing success is also independent evidence that coherence machinery lives in the mixing layers. *Source:* arXiv:2508.15884. Contract: adaptation only (published version trains).

---

## 2. CHIMERA OMNI NOTES

Non-MoE design observations worth carrying into the from-scratch multimodal model. The 2026 field verdict (GLM-5.3-Flash and Qwen3.8-Flash-Next are FRESH pretrains, not derivations): efficiency must be designed in before pretraining — plan the small config from the start.

**Architecture.**
- Residual: 4-stream norm-conserving widened residual (mHC doubly-stochastic mixing, or Qwen Gated Residual). Credited with stabilizing very deep sparse hybrids; a mass-conserving residual should also make the finished model dramatically more foldable.
- Attention: output-gated attention from day one — empirically produces a sink-free, massive-activation-free trunk, removing the fragile emission-outlier machinery that folding and quantization keep tripping over.
- Mixing stack: 3:1 linear-recurrent (channel-wise decay, KDA-style) to full-attention layers; NoPE on the full-attention layers (recurrence carries order) or partial RoPE (~25% of dims). Cross-lab consensus in 2026.
- Memory recipe (Gemma 4): 5:1 local:global attention, cross-layer KV sharing, K-reuse-as-V on globals, pp-RoPE on globals.
- Keep every block's interface uniform (drop-in variant menus stay cheap forever — Puzzletron retrofit); blocks read a shared stream rather than strictly chaining (low pairwise dependency = near-free depth folding/parallelization later); plain linear down-projection so every block has a fusion slot for absorbed linear corrections.
- Declare a protected-layer set (the few global-mixing/attention layers) and explicit, identifiable compute blocks in the middle region (where algorithmic state-tracking circuits live) in the architecture spec — future folds must know what never to touch.

**Output head and vocab.**
- Pick vocab size deliberately: big vocab + small trunk is Google's small-model recipe; embedding params are cheap at train time and offloadable (PLE-style) at inference.
- Cluster the LM head from day one; budget a ~4-layer drafter sharing the trunk's KV cache (~1-2% of params — Google ships this as standard equipment).
- Give training/distill losses a logit-level term on the final block (LFQ lesson: hidden-state-matching-only calibration bakes in generation fragility).

**Capacity without FLOPs.**
- Early-layer hash n-gram/Engram table with a gated delta-residual write (and patch-hash for vision) — the 2026 way to buy knowledge capacity that lives in host RAM, offload-friendly.

**Foldability insurance (training losses).**
- Matryoshka losses on widths/FFN and MRL on every embedding/projection head — makes every future width fold a free slice; this is exactly the property whose absence is our 0.92 floor.
- MatQuant-style nested-precision QAT: one checkpoint serves int8/int4/int2 tiers.
- Explicit per-axis redundancy budgets so post-hoc compression has known safe margins instead of discovered cliffs.

**Telemetry and eval (bake in from day one).**
- Persist per-layer activation Gram matrices XX^T at every eval checkpoint — every closed-form repair family (ridge, OBS, EoRA, RegMean) consumes exactly this object.
- Log DM(l) decision-margin profiles during training (transition depth is a design variable predicting compressibility); log which units correlate with repetitive sampling (a known loop-circuit map makes cauterization surgical).
- Eval harness: generation-mode probes only as gates (never MC/classification as proof a checkpoint works); greedy-agreement vs a frozen reference on a pinned 1M-token cache; entropy-trajectory monitoring; a TPS-under-PPL-budget rig with a private prompt set.
- Evidence caches versioned per model-state, never reused across states.

---

## KILL LIST

Shiny but dead — contract violations, param-neutral compute tricks, or duplicates of the sealed record. Where an adaptation survives, it is cited by entry number above.

- **2:4 sparsity as shipped (SparseGPT/Wanda patterns)** — zeros weights, doesn't remove them; checkpoint never shrinks. OBS compensation math survives as #23.
- **KDA / post-hoc linear-attention conversion** — cuts KV cache and FLOPs, not params; every published linearization needs finetuning to recover coherence (our IxB law, confirmed).
- **Lightning-indexer / QSA sparse attention** — ADDS a module, targets compute; block-pooled score-regression survives only as a redundancy probe.
- **MTP drafters as shipped** — jointly trained; the verifier acceptance gate survives as #16.
- **Mix-n-Match on arbitrary (non-nested) parents** — the extraction is free only because the training wasn't; valid solely on E-model parents (#28).
- **MatQuant as quantization** — reserved by contract; joint nested-objective math survives as #24.
- **PostNAS / JetBlock as published** — gradient-trains the replacement attention; layer-placement analysis and ridge-mixer variant survive as #33.
- **Every distillation recovery stage (Minitron, MiniPuzzle 63B tok, Nemotron Nano 2 135B tok, FFN Fusion healing)** — the exact thing our program exists to replace; their scoring math was harvested, their healing discarded.
- **FFN fusion alone** — parameter-neutral concatenation; only pays after attention removal (#19).
- **Qwen 51B hash n-gram table as shipped** — adds 51B params with nonstandard lookup export; only the 1-gram embedding-repair degenerate survives (#31).
- **PLE token-conditioned correction cell** — adds params and quietly swaps the metric to "resident params"; PARKED pending Cyril's explicit metric decision, not built.
- **Fast Gemma Challenge winning stack (vLLM/MTP/sliding-window tuning)** — pure serving throughput, zero param reduction; only the blind-set gate methodology survives (#27).
- **GLM-5.3-Flash / Qwen3.8-Flash-Next as derivation comparators** — fresh 30T-token pretrains wearing "small sibling" branding; never benchmark fold x-factors against them. Corollary: fold-only derivation remains unclaimed territory.
- **Ghosted Layers naive deployment** — un-exportable dense CxC forward hook; anti-symmetry diagnostic survives as #32.
- **ReplaceMe cosine-objective variant** — needs Adam; LS/ridge variant only (#9).
- **Gated attention / output gates as a retrofit fix** — training-time intervention; the sink mechanistic account survives as #2.
- **PruneCD runtime contrastive decoding and per-step entropy control** — cannot be stored in weights; static extracts survive as #13 and #15.
- **SAE / gradient-attribution loop-neuron identification** — gradients; difference-of-means substitute survives inside #14.
- **Gemma 4 KV-sharing / K-as-V as architecture surgery** — design-time only for existing checkpoints; Procrustes W_v-elimination probe survives as #18.
- **Gemma 3 270M** — a design position, not a technique; survives only as gate-sharpening for vocab folds (#26).
- **ACE data-free Fisher covariance proxy** — we HAVE calibration activations; use real Grams in any merge (#30).
- **EmbeddingGemma MRL training itself** — training; spectrum probe and nested-rank ladder survive as #12.
- **Duplicates collapsed silently:** EoRA (2 findings -> #5), ReplaceMe (2 -> #9), arXiv:2608.08506 (3 -> #1), LinearPatch/Prune&Comp diagonal family (2 -> #10).