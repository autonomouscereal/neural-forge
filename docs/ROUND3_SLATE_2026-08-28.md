# ROUND-3 SYNTHESIS — RANKED BUILD SLATE (waves 40+)

Contract held throughout: closed-form solves on cached evidence only, zero gradients, no quantization, no MoE/routing, all params counted, standard-checkpoint export for frontier claims, no class closes without a BUILT child.

---

## 1. Dedup / merge actions (fixes applied)

**M1 — ROWSET** ← merge of floor-mechanics #1 (second-census tie-break, KEEP) + emission #2 (TIEBREAK-KILL, FIX applied) + the one live fragment of the killed knapsack (second census as tie-value). Same machinery (re-rank the discretionary fill in `choose_kept` at fixed K), same bodies, same death rungs, byte-identical storage to sealed 0/15 controls. Becomes one three-arm cell family. Fixes carried in: receipt citation corrected to `receipt-V12CL896X09.json` (K=12,288, not 10,240); body named exactly as the wave-37/38 retained CL896X09 (MX37/MX38 lineage, not "MX35"); wave-9 `NOTES-MX-M.md` §2.3 cited as prior art with the Arm-C failure prediction pre-registered; all scoring corpora sealed by hash, disjoint from the frozen 15 AND the 30-prompt extension bank; rank-divergence vs parent-id order published in the receipt so a null is interpretable.

**M2 — NORM-ESTATE** ← merge of structural #4 (BAND, KEEP) + emission #3 (READER-RECAL-27, KEEP). Identical free-capacity estate (the never-touched norm-gain diagonals, forward conditioned install, zero storage delta, same control pair); they differ only in objective (plain per-coordinate stream match vs reader-Gram-weighted) and arm granularity. Merged into one arm grid that straddles the wave-29 free-parameter hazard from both sides. Fix carried in: the w3-absorption argument is deleted and restated correctly — the ffn_norm gain is input-side on BOTH branches (right-multiplies w1 and w3), so the entire diagonal is outside the sealed w2-solve span; the truth is stronger than the original claim.

**M3 — shared child**: BISECT's bracket-midpoint child (K=9,728 on CL896X09, 0.409364x) is byte-identical to MLP-QUANTUM's required mid-bracket axis-attribution control. One build serves both specs; it is counted once.

**T1 — ROWSET-LANE** ← floor-mechanics #3 trimmed per verdict to ONE cell (retrieval-steered census at K=10,240 on CL896X09 vs its sealed 1/15 generic twin). X13 members blocked until wave 39's V10CL896X13 control is sealed in the ledger. λ1.0 arm dropped (kill list).

**Distinctness ruling (why ALIAS, ROWSET, BISECT, MLP-QUANTUM all survive on one axis-neighborhood):** they are four orthogonal single-variable probes of the same death bracket — ALIAS varies the dropped-type MAPPING (no re-segmentation, sealed rows kept), ROWSET varies WHICH rows (re-segmentation kept), BISECT varies K in the bracket interior (rows and mechanism kept), MLP-QUANTUM enters the bracket ratio from a second axis (vocab machinery untouched). Joint interpretation matrix, pre-registerable now:
- ALIAS pulses, ROWSET null → death is re-segmentation, not row identity.
- ROWSET pulses, ALIAS null → death is row identity (tie-break block), not segmentation.
- BISECT interior pulse → floor moves in 0.0011-ratio steps; "quantized" was the ladder.
- All vocab-side null + V14CL768X09 null → first evidence the bracket is a RATIO wall (axis-independent); L2 promotes from tokenizer statement to ratio statement.
- All vocab-side null + V14CL768X09 alive → bracket is vocab-axis-specific; the floor moves via mlp x vocab.

All other fixes from the verdicts are embedded in the spec blocks below.

---

## 2. Ranked slate

Rubric: floor-relevance x error-class-novelty x cost(inverse). Grades H/M/L; rank follows the product. Note the two places the rubric and campaign value diverge — they are resolved in build order (§6), not by bending grades.

| # | Item | Floor | Novelty | Cost-inv | Note |
|---|------|-------|---------|----------|------|
| 1 | ALIAS | MAX (at the open question, sealed 0/15 twin) | H (type-identity coarsening, new class) | MAX (CPU) | Both outcomes are laws |
| 2 | ROWSET (M1) | MAX (moves floor a full rung to 0.4071x) | M (selection-order at fixed K, never varied) | MAX (CPU) | Two lenses converged independently |
| 3 | XCHANGE | MAX-deepest (0.3796x PLANE-independent; 0.3317x with PLANE) | MAX (operator substitution, rank 1) | M | Highest variance on slate; lowest prior; carried by near-free go-gate |
| 4 | PLANE | H (0.3911x composed if d=48 pulses) | H (band-slide knob new; rest is wave-15 machinery, stated honestly) | H | Stock-clean, source-verified |
| 5 | BISECT | MAX (only path to life below 0.4116x on-axis) | L-M (finer step + off-corpus meter) | MAX (CPU) | Midpoint doubles as MLP-QUANTUM control |
| 6 | MLP-QUANTUM | H (bracket entry from second axis; 3 rungs, 2 below deepest ever built) | L (same CL class; new QUESTION: axis-independence) | M (one ~320s GPU body) | Rubric under-prices it: highest expected floor move per GPU-second; wave-A anyway |
| 7 | NORM-ESTATE (M2) | ZERO by construction (never quote as ratio) | M (gauge repair, 38 untouched sites) | H | The only repeatable +1, extended or closed with cause |
| 8 | ROWSET-LANE (T1) | L (ratio held, claim is capability) | M | MAX (CPU) | Unlocks §4 if it moves |
| 9 | SC-CORESET | RESEARCH only (non-stock; adjudication-gated) | M (new site class, old mechanism) | M | Prices the 31% mass either way |

---

## 3. Spec blocks

### 3.1 ALIAS — vocabulary fold by row aliasing (type-identity coarsening)

- **Mechanism:** keep the sealed census's top-K rows UNCHANGED; map every dropped type to its nearest surviving type instead of re-segmenting. Document token count and segmentation preserved; what degrades is type identity. Removes exactly the mechanism L7's corpus floor describes.
- **Law hope:** L2 (a twice-reproduced 1,024-row bracket is either tokenizer artifact or information — this cell decides), L7 (coverage preserved by construction), L10 (only a built child can adjudicate the vocab axis).
- **Solve:** closed-form, one CPU pass, no capture. Assignment a(v) = argmin over kept u of ||E_v − E_u||²_M, M = tied-unembedding Gram (L11's reader projection as metric); deterministic id tie-break; one 65,536 x K fp64 distance pass. ARM-V: kept rows verbatim (zero solve). ARM-C: census-frequency-weighted centroid, one k-means M-step, no iteration. Alias remap ships in the tokenizer artifact; config is stock `vocab_size: K` (wording fixed per verdict — vocab_size = K, not "same vocab_size").
- **Storage:** byte-identical to the sealed truncation fold at the same K. K=9,216 on retained MX38 CL896X09 body: 93,504,256 = 0.4070832855x. K=10,240: 94,552,832 = 0.4116484014x.
- **Export:** model side fully stock. One ruling registered BEFORE build: does an id-remapped tokenizer count as standard-checkpoint (precedent favorable — sealed V folds already ship re-derived tokenizers; a remap is strictly smaller). Probe-only fallback still decides L2.
- **Pre-registration notes (from verdict):** name the asymmetry — aliasing makes dropped-type strings ungeneratable by construction while truncation can emit them re-segmented, so the retrieval read must distinguish context-READING recovery from literal-emission recovery.
- **Kill gate (built children):** ARCH.V9CL896X09-ALIAS ARM-V and ARM-C both 0/15 at K=9,216 → re-segmentation is not the escape hatch; L2 hardens from observation to mechanism law.
- **Cheapest first build:** ARCH.V9CL896X09-ALIAS ARM-V at K=9,216 vs the sealed 0/15 / canonical 0/15 twin. One CPU pass, one review. Any signal = new double-witnessed floor one bracket deeper. K=10,240 twin (retrieval-lane diagnostic) nearly free in the same wave.

### 3.2 ROWSET — fixed-K kept-row re-ranking at the sealed vocab death rungs (merged M1)

- **Mechanism:** more than half the kept vocabulary at the floor is chosen by the tie-break (parent-id = merge/frequency order; receipt-V12CL896X09.json: 4,633 observed tokens admitted, tie-break decides the rest; NOTES-MX-AA §9: ~14,000 of 16,384 rows). The exact 1,024 rows separating dead V9CL896X09 from living V10CL896X09 are one contiguous blind tie-break block. Vary the fill order at fixed K; weights untouched, verbatim gather, torch.equal gate.
- **Law hope:** L10 verbatim (solves provably blind to the vocab axis — this fixes the blindness at the selection step), L1 (the vocab census has never been audited by a built child), L2 (a resurrection at a death rung means death is quantized by the tie-break, not tokenizer structure), L5 (selection, not reader metrics), L7 (~2,000–3,000 discretionary rows at these K).
- **Solve:** zero solve on weights. Mandatory set held exactly as sealed (specials, 256 ByteLevel alphabet, transitive closure, all evidence-observed tokens) — coverage stays 1.000000, below-floor gate cannot fire. Arms fill only the discretionary budget:
  - **Arm A (second census):** key = (−evidence_count, −heldout_count, parent_id); held-out generic+QA+analysis corpus sealed by hash, disjoint from frozen 15 and extension bank; rank-divergence vs parent-id published (parent-id IS a frequency proxy — a null must be interpretable).
  - **Arm B (segmentation participation):** score candidates by parent-canonical positions participated in over sealed register-matched held-out text; attacks the 0.0992 review-prompt position agreement; the arm wave-9's own measurement favors (coverage is the currency that discriminated).
  - **Arm C (leverage s_r = e_rᵀ M e_r, Cholesky pivot order):** built LAST with the pre-registered PREDICTION that it fails like wave-9's row-norm ordering (2.34% coverage at M=4,096) — predicted AND built, never skipped.
  - Receipt: pre/post-closure coverage, tokens-per-byte, review-prompt position agreement beside sealed 0.0992.
- **Storage:** byte-identical to controls. K=9,216 on CL896X09: 93,504,256 = 0.4070832855x (vs sealed 0/15). K=8,192 on CL896: 93,902,592 = 0.408819x... (two-body replication vs sealed ARCH.V8CL896 0/15).
- **Export:** identical to every sealed V-child; frontier-eligible.
- **Kill gate (built children):** all three arms 0/15 at K=9,216 AND best arm 0/15 at K=8,192 → row-set identity is inert at fixed K; L2's quantization claim gets its first real support; future vocab waves argue only about how many rows, never which.
- **Rider discipline:** any 1-case resurrection requires 30-prompt extension-bank concordance; Arm B's scoring text must also be disjoint from the rider bank or the rider is void.
- **Cheapest first build:** Arms A and B at K=9,216 (two 40-second CPU passes + two reviews); C and the K=8,192 replication follow in-wave.

### 3.3 XCHANGE — conv block → attention block via layer_types (operator substitution)

- **Mechanism:** replace a surviving ShortConv block (3-tap causal FIR + channelwise gating, fixed-offset mixer) with a RoPE attention block constructed to concentrate at fixed relative offsets. First operator-KIND change in 988 rows. What is lost is input-dependent B/C gating — that loss is the experiment.
- **Fixes applied (all three from verdict):** (a) builds from / explicitly supersedes `docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md` (design-only, never built), running its FIR-vs-softmax residual per site pre-build for attribution; (b) Stage-1 rebuilt — single-plane heads alias (cos re-peaks at Δ+2πk/ω; at plane 0, m−Δ=44 gives cos≈0.9998, no τ separates it): each head gets ≥2 incommensurate planes phase-aligned at Δ_h (32 planes / 16 heads permits pairs) OR ridge-fit q/k against cached FIR score targets per the design doc; A's concentration verified on a 4k-position grid before install; (c) the "substitution preserves strictly more information than deletion" falsifier is DELETED as logic (a bad operator pollutes the stream where a hole does not).
- **Law hope:** L6 (names the sites and supplies deletion baselines), L12 (the installed operator is the one the campaign understands), L4 (width closed in every combination — conv mass reachable only by leaving the width axis).
- **Solve:** Stage 1: fixed positional score geometry by construction (multi-plane) or ridge fit; per-head τ_h by deterministic 1-D bisection matching softmax mass at Δ_h to normalized tap magnitude, absorbed into existing q_layernorm. Stage 2: with A fixed the block is linear — W_v init verbatim from in_proj x-branch rows, W_o prior-anchored ridge against the block's conditioned residual write, ONE alternating pass, pre-registered stop (VC-FOLD step-6 discipline).
- **Storage:** per conversion at d=64: 4,197,376 → 3,145,856 (saves 1,051,520). XA13 solo on parent: 228,641,664 = 0.995422x. All-7 on floor body at d=64: 87,192,192 = **0.379603x — below the floor with NO dependence on PLANE**. With PLANE d=32: 76,181,696 = 0.331667x. Partial schedules independently priced.
- **Export:** stock — layer_types is a stock Lfm2Config field; state_dict change is layers.N.conv.* → layers.N.self_attn.*. Caveat retained: head_dim is global, so converted and native layers must share d.
- **Kill gate (built children, two-cell pair verdict):** XA13 (go-gate, vs its own deletion baseline X13A 13/15) + XA09 (value test, vs ARCH.X09 sealed 10/15, a site where deletion actually costs). XA13 < 13/15 AND XA09 < 10/15 → class dies by two built children. XA09 ≥ 10/15 → operator carries value where deletion costs; fund the 7-block schedule.
- **Cheapest first build:** ARCH.XA13 at d=64 on the full parent (operator is the only variable), 0.995422x — one CPU construction + one ridge + one review; baseline already sealed.

### 3.4 PLANE — RoPE frequency-band fold, opened at d=48

- **Mechanism:** whole-frequency-plane head_dim moves via the (d, m, s) family: head_dim + rope_theta' = θ^(m·d/64) + rope_type "linear" factor θ^(s/32) sliding the band window. Novelty restated honestly per verdict: the (d, m, θ') mechanics are wave-15 machinery (`mx15_build.py rope_plane_sets`); the genuinely new knob is the linear-factor band slide s; the new rungs (d=48, d=32) were never visited. Error class: positional-bandwidth truncation — residual stream keeps full width, so L11's wall does not apply.
- **Fixes applied:** open at d=48 not d=32 (m=1 forced: stride·23 < 32 permits only stride 1; planes 0..23, θ' = 1e6^0.75 = 31,622.78); pre-build, run the existing plane_energy census (`mx15_build.py:431`) on the d=48 and d=32 sets and PRE-REGISTER the retention numbers (HD16's receipt read ~0.24 retained score energy ≈ 8/32 uniform → predicts ~0.75 at d=48, ~0.50 at d=32) so a death attributes to mechanism, not rung. The census publishes either way and never gates. HD32 is rung two; the three-arm band portfolio (m=2 vs m=1,s=0 vs m=1,s=16 at byte-identical storage) is contingent on a d=48 pulse.
- **Law hope:** L12 (permission granted, only the extreme rung ever spent), L6 (band choice as capability lever — §4), L5 (verbatim rows + one writer re-solve), L2 (no attention fold exists anywhere in the deep family; attention is 19.96% of the floor child).
- **Solve:** verbatim q/k row slice on S ∪ {s+32}; q/k layernorm sliced; the only inexactness is the RMSNorm denominator over d coords, absorbed into existing q/k layernorm vectors by a plain normal-equation ridge in d unknowns per layer against cached pre-softmax logits; v/o: verbatim value-channel subset scored s_c = E_{D²}[|ctx_c|]·||W_o[:,c]||₂, W_o by prior-anchored D² ridge on the exact column subset of sealed context (no attention replay). update_scale ≤ 1.0 fail-closed.
- **Storage:** per-layer 3,145,856 / 2,359,392 / 1,572,928 at d=64/48/32. HD48 solo on parent: 224,974,400 = 0.979456x (derived from verified per-layer numbers). Composed on floor body: d=48 → 89,834,048 = 0.391105x (below the floor); d=32 → 85,115,264 = 0.370561x.
- **Export:** stock-clean, source-verified (modeling_lfm2.py:234 getattr head_dim; rope_parameters + "linear" stock). Frontier-eligible.
- **Kill gate (built children):** HD48 solo ≥ 12/15 funds HD32 and the composed d=48 pass; HD48 < 10/15 at 0.9795x solo re-seals the axis WITH the pre-registered energy attribution (a solo rung this shallow must be near-parent to fund a 6-layer composed spend); 10–11/15 → build composed d=48 directly and let it decide.
- **Cheapest first build:** ARCH.HD48 solo (CPU slices + d-unknown ridges + W_o ridge off retained L2..L12 sealed context + ~100s decode) + the CPU-only plane-energy census.

### 3.5 BISECT — sub-quantum interior rungs + off-corpus segmentation meter

- **Mechanism:** both brackets are exactly 1,024 rows wide because that is the LADDER's step; nothing has ever been built inside either (DEEP_RECORD §5.1/5.2, "one 40-second CPU pass to halve"). Build 256-row interior rungs; pair with the off-corpus meter (on-corpus it is saturated by construction — wave 30 found coverage 1.0 down to 8,190 rows; off it, per mx34_vocab.py's own header, "it is the whole experiment").
- **Fixes applied:** knee criterion stated NUMERICALLY in the preregistration before any build (max second difference of changed-position fraction over the 256-step grid — kept sets nest, so the curve is monotone and a vague "knee" would be post-hoc); any single 1/15 interior survival must be TWINNED (G-arm λ1.0 or a ROWSET arm at the same K) before a floor-move claim, matching the double-witness discipline of the sealed floor.
- **Law hope:** L2 (the law itself flags the quantum), L7 (same step-size confound at the solo peak's shoulders), L11-by-analogy (a pre-pricing wall for the vocab axis built from segmentation instead of energy).
- **Solve:** meter = in-memory choose_kept + remapped tokenizer per K, held-out capability bank tokenized, ids mapped via `injection()` from mx34_retokcensus.py verbatim; record tokens_per_byte(K), changed_positions(K); predict death K; then verbatim-gather builds. Zero GPU for builds.
- **Storage:** K=9,472/9,728/9,984 on CL896X09 → 93,766,400/94,028,544/94,290,688 = 0.408223/0.409364/0.410506x; K=8,448/8,704/8,960 on CL896. Each 256-row step = 262,144 params = 0.0011413 of ratio.
- **Export:** fully standard; frontier-eligible for any surviving interior rung.
- **Kill gate (built children):** all interior rungs of a bracket 0/15 at 256-step resolution (sequential stop rule pre-registered: bracket halts after two consecutive 0/15, K-descending) → the cliff is real and located; the meter becomes a free pre-pricing instrument. Any twinned interior survival → the floor moves below 0.4116x and "quantized" was the ladder.
- **Cheapest first build:** midpoint K=9,728 on CL896X09 (0.409364x) + the CPU meter pass. **This child is simultaneously MLP-QUANTUM's required mid-bracket axis-attribution control (M3) — build once, grade once, cite twice.**

### 3.6 MLP-QUANTUM — CL768 rungs and axis-independent bracket entry

- **Mechanism (rewritten per verdict):** the load-bearing claim is AXIS-INDEPENDENCE, not reachability — the false "the vocab axis cannot land there" premise is dropped (K is not constrained to 1,024-multiples). One SwiGLU channel on the 13-block body = 39,936 params = 0.00017387 of ratio, a 26x finer quantum; the 768/832/880 hole in the measured ladder (1792→7, 1280→4, 896→4, 640→0 both arms) has never been built. Entering the bracket at 0.40766x from the mlp axis, against the same-wave vocab-axis mid-bracket control, is the first test of whether the bracket is a ratio wall or a vocab-axis artifact.
- **Law hope:** L4 (mlp x vocab mildest at 0.76 — and note the sealed floor child is itself a triple, so the X09 rungs are not law-doomed), L5 (channel selection composes), L2 (bracket entry from a second axis is the only way to distinguish walls), DEEP_RECORD §8.5 (2/2/1/0 ladder leaves headroom at K=14,336).
- **Solve:** sealed CL recipe at keep=768: conditioned-local capture through the partially built child, pivoted-Cholesky coreset of the SwiGLU hidden Gram, w1/w3 rows VERBATIM in pivot order, w2-only prior-anchored ridge, λ laddered to update_scale ≤ 1.0. Vocab folds are the existing zero-solve CPU gather. MANDATORY pre-build gate: write both intermediate_size and block_ff_dim, round-trip through stock from_pretrained, assert ints and every w1/w2/w3 shape (LFM2 ff auto-adjust hazard); 880 attempted only if it round-trips.
- **Storage (all recomputed, all counted):** ARCH.CL768X09 = 146,064,128 (0.635908x); ARCH.CL768 unexcised = 152,622,848 (**0.664464x** — corrected per verdict). Composed: **V9CL768 = 94,951,168 (0.413382x)** promoted per verdict to co-equal first rung (pure mlp x vocab, the measured mildest pair, no depth tax); **V14CL768X09 = 93,635,328 (0.407655x, INSIDE the bracket)**; V12CL768X09 = 91,538,176 (0.398524x); V10CL768X09 = 89,441,024 (0.389393x, below the deepest child ever built).
- **Export:** standard — 15 waves of intermediate_size precedent; frontier-eligible subject to the round-trip gate.
- **Kill gates (built children):** body-level — ARCH.CL768X09 0/15 vs CL896X09's sealed 1/15 kills the sub-896 shelf and prices the 640–896 cliff to within 128 channels for one build. Bracket-level — V14CL768X09 graded against V14CL896X09 (sealed 2/15) and V9CL896X09 (sealed 0/15), WITH the K=9,728 vocab-axis control (M3) in the same wave: 0/15 on both entries reads ratio-wall; a split reads axis effect. No bracket conclusion is claimable without the paired control.
- **Cheapest first build:** ARCH.CL768X09 body (~320s GPU on the wave-37/38 path) + one 40s pass to V14CL768X09. ARCH.CL768 body second (funds V9CL768, the mildest-pair rung).

### 3.7 NORM-ESTATE — the full norm-gain diagonal estate under conditioned forward install (merged M2)

- **Mechanism:** 39 gain vectors / 28,416 already-counted per-channel gains on the floor body (26x1024 block operator/ffn norms + 12x64 q/k layernorms + 1x1024 terminal); only the terminal has ever been solved (G arms, the estate's only repeatable +1). Corrected structural claim: the ffn_norm diagonal is input-side on both SwiGLU branches and therefore wholly outside the sealed w2-solve span — free capacity outside every objective the campaign has minimized.
- **Law hope:** L8 (diagonal arm lands three at 5 of 6 K, at ONE of 39 sites), L10 (conditioned capture worth 3 cases — each site solved against the child's own stream through the already-repaired prefix), L9 (repair must live in weights; decode-repair null), L11 (re-aims readers at retained energy; exchanges zero directions, so wave-21's basis-invariance condemnation cannot reach it). CT640 precedent quoted in the preregistration: this is conditioned-LOCAL target matching, not trajectory targeting.
- **Solve:** forward site order, re-capture after each site, terminal solved LAST (sealed G arm stays a nested special case). Objectives/arms, pre-registered as a grid straddling the wave-29 hazard (20,480 unknowns → filler; 20 → echo):
  - **Arm M (moment-match):** d_j = g_j·sqrt(E[n_p,j²]/E[n_c,j²]) — pure closed form, zero fitted parameters.
  - **Arm S (scalar per endpoint):** 27 unknowns, reader-Gram-weighted normal equations (Hadamard of two PSD Grams, SPD, one 1024x1024 Cholesky per endpoint; readers = [w1;w3] / [q;k;v] / in_proj / lm_head).
  - **Arm D (full diagonal):** 27,648 unknowns — ABOVE the wave-29 filler count, flagged as the hazard side; run in both plain per-coordinate stream-match form (separable, no inverse) and reader-weighted form; λ laddered 0.3/1.0 mirroring sealed G03/G10 for a like-for-like read. q/k layernorms (64-dim) get the plain separable solve.
  - Identity gate: child = parent returns g bit-exactly. update_scale published per site, fail-closed above 1.0.
- **Storage:** ZERO delta. 94,552,832 before and after; byte-identical shapes and config.
- **Export:** strongest available — no shape, config, or tokenizer change; nothing to rule on.
- **Kill gate (built children):** the full pre-registered grid (M, S, D-plain, D-reader at the λ ladder) on ARCH.V10CL896X09, all reading ≤ the sealed one-site G10 twin (2/15) with no case swap → L8 is a TERMINAL-SITE law, the diagonal estate closes with a cause, and the per-wave G-arm cell stops being budgeted. Any arm beating G10 → re-run free across the whole 0.42–0.60 canonical frontier before any further fold.
- **Cheapest first build:** Arms M and S (~2 min GPU each incl. re-captures) beside the sealed 1/15 / G10 2/15 pair. **Never quoted as a ratio result.**

### 3.8 ROWSET-LANE — lane-steered census, single cell (capability selection at the floor)

- **Mechanism:** retrieval died at exactly K=10,240 on X09 ("sholner") while alive at 12,288, and the rows lost between those rungs are precisely count-0 tie-break rows — the only rows steering can reorder. One cell: can a lane-style census re-admit the tokens whose loss killed the lane?
- **Law hope:** L6 (block choice sets WHICH capabilities survive; zero lane overlap at same ratio), L10 (conditioning at identical storage), L7 (body contributes the family, kept set the segmentation).
- **Solve:** key = (−evidence_count, −lane_count, parent_id) at K=10,240 on retained CL896X09; steering corpus = QA-with-explicit-facts documents, sealed by hash, register-matched, with the frozen prompts' specific facts, times, and entities EXPLICITLY FORBIDDEN (memorization guard per verdict). One pre-registered claim: retrieval reads; name what happens to uncertainty. λ arm dropped.
- **Storage:** 94,552,832 = 0.4116484014x, byte-matched to the sealed 1/15 generic twin.
- **Export:** standard; frontier-eligible per child.
- **Kill gate (built child):** retrieval does not return at the steered cell → census steering cannot select capabilities; the entire portfolio concept (§4, lever b) dies for 40 seconds of CPU.
- **Expansion (contingent):** uncertainty-steered X09 member; X13 members ONLY after wave 39's V10CL896X13 control seals in the ledger. Each member a separate standard checkpoint; no router, no shared runtime, claims per-child.
- **Cheapest first build:** the one cell above. CPU + one review.

### 3.9 SC-CORESET — ShortConv channel coreset (RESEARCH PROBE + adjudication ask)

- **Target reclassified per verdict:** research-floor + adjudication ask, never a frontier floor claim under the current contract (Lfm2ShortConv hardwires hidden_size; probe materializes via hf_lfm2_canonical.py; the vocab_size precedent is the adjudication route).
- **Mechanism:** first internal-channel fold of the conv branch (31.07% of the floor child, touched only by whole-block excision at 3 cases/block). Structure is exactly the w2 pattern: channelwise gating throughout, out_proj sole writer, kept paths verbatim.
- **Fixes applied (all three):** (a) pre-registered AGAINST wave14 `NOTES-MX-R.md` §2.4 — at identical storage the verbatim conv coreset LOST to evidence-weighted low-rank at 12/15 site-levels (C05: 3 vs 12; L07 cos 0.8429 vs 0.8718 at S4); therefore the matched-storage LRA twin is built as the head-to-head control, plus the S4-sibling attribution cell A3's spec required, so a death attributes between mechanism and level; (b) the predict-and-skip clause is STRUCK — the E[uuᵀ] spectrum pre-read publishes beside the sealed SwiGLU spectra either way and NEVER gates (WAVE23 FIX-1 wording); the class closes only by the built child; (c) target field now matches export status.
- **Law hope:** L5 (structural match to the fold that works), L6 (conv sites carry removable capacity; coreset is the graded version of excision), L4 (width closed — conv mass reachable only via its own channel axis).
- **Solve:** per site from conditioned capture: cache gated post-conv u_t; pivoted Cholesky on E[uuᵀ] (pivot order in receipt); in_proj rows (all three streams) and depthwise taps at pivot indices VERBATIM; out_proj-only prior-anchored ridge, λ laddered, update_scale ≤ 1.0; sequential forward install; identity gate at |S| = H.
- **Storage:** block at |S|=512: 2,100,736 (vs 4,199,424). CL896SC512 = 141,338,368 = 0.615340x. Deep rung (floor recipe, 7 conv blocks): 79,862,016 = 0.347699x; |S|=768 rung: 87,207,424 = 0.379666x.
- **Kill gate (built children):** ARCH.CL896SC512 AND its matched-storage LRA twin both ≤ 1/15 vs CL896's sealed 4/15 → the largest unfolded mass is buried by built children and the "31% untouched" argument ends permanently. Coreset ≥ 3/15 → conv sites fold like MLP sites; the deep rung is spent only on survival PLUS an A2-style adjudication from Cyril.
- **Cheapest first build:** the SC512 + LRA-twin pair on the living CL896 body, reusing the sealed conditioned capture (one GPU pass each) + spectrum pre-read (CPU, publishes, never gates).

---

## 4. The capability-portfolio question (standalone deliverable option)

**The question, put to the caller as its own decision:** at fixed ~0.41x all-params-counted, is the surviving capability set CHOOSABLE, and if so, is "N choosable specialists at the floor" a deliverable the campaign wants, or does everything fund the single-model floor?

**What is already proven:** one lever works — excision block choice (L6: X09 vs X13 at identical ratio share ZERO of 16 lanes). The floor's surviving pair (retrieval + uncertainty at V10CL896X09) is currently an ACCIDENT, not a design.

**The deliverable, if elected:** a portfolio of N separate standard Lfm2ForCausalLM checkpoints at byte-identical 0.4116484014x (94,552,832 params, wave-39-style parameter-count identity as a falsifiable build criterion), each pre-registered to hold ONE named capability. NO router, NO shared runtime, NO claim the portfolio is one model — every claim per-child at its own all-params-counted ratio. Contract-clean by construction (no MoE/conditional execution anywhere).

**The four candidate levers, in evidence order:**
1. Excision block choice — proven (L6).
2. Lane-steered census — unproven; §3.8 is its one-cell test.
3. Frequency-band choice (PLANE m/s arms at byte-identical storage) — contingent on a d=48 pulse (§3.4).
4. Norm-gauge arms (§3.7) — capability-band displacement, if any arm moves cases.

**Gate to declare the portfolio real:** two same-ratio children with pre-registered DISJOINT lanes, each reading its named lane where its twin does not, from at least TWO independent levers — one lever (blocks) already exhibits this; the open question is whether a second lever makes capability choice compositional. The cheapest decisive spend is §3.8's single cell.

---

## 5. Kill list

1. **Closure-debt knapsack census** — the priced constraint has never bound: every build receipt at every K ever built reads tokens_skipped_because_their_closure_did_not_fit: 0 and observed_tokens_admitted: 4,633, so the value function is degenerate at all feasible budgets and the fill ordering it "optimizes" is already an arbitrary tie-break; live fragment absorbed into ROWSET Arm A; re-proposable only if a future corpus makes the closure counter nonzero in a real build.
2. **Structural factorization family (Monarch, butterfly, tensor-train, displacement-rank, semiseparable, Nystrom, ID, sparse+low-rank)** — same reader-projected retained-energy error class L11 sealed basis-invariant across six objectives with a flat spectrum, AND none is stock-expressible; double kill, checkable in source.
3. **Margin/temperature emission restore** — killed at depth in wave 29: V19X13CL896 reads 0/15 with generative margin at 96% of parent, and stock Lfm2ForCausalLM has no lm_head bias to host it; nothing to correct.
4. **Massive-direction write restore** — wave 30: W768/W720 write 4.35%/4.05% of parent energy on the channel and read 3/15 and 0/15; the quantity does not track behavior.
5. **Lane-portfolio λ1.0 arm** — dropped per DEEP_RECORD 8.3's two-solve rule; re-proposable only as its own budgeted solve.
6. **S6/29-DIAGONAL as width-hosted** — stranded on the closed width axis with a no-op identity gate as written; content re-hosted into NORM-ESTATE (§3.7), the width version is dead.
7. **XA13-kills-the-class falsifier logic** — "substitution preserves strictly more information than deletion" is false (a bad operator pollutes the residual stream where a clean hole does not); replaced by the two-cell pair verdict in §3.3.
8. **SC-CORESET's spectrum-gating clause** — verbatim predict-and-skip; struck; the pre-read publishes and never gates.
9. **HD32-first opening of PLANE** — not killed as a rung (it is rung two) but killed as the OPENING: the estate's own plane-energy census predicts ~0.50 retention there, and re-sealing an axis at a predicted-dead rung when a milder rung exists at identical cost repeats the HD16 error.

---

## 6. Suggested wave partition (build order ≠ rank order; reviews are the real budget)

**Wave A — bracket anatomy + cheap gates (~8 reviews, GPU ≈ one CL body + one construction):**
ALIAS ARM-V @K=9,216; ROWSET Arms A+B @K=9,216; BISECT midpoint K=9,728 + off-corpus meter (child shared with MLP-QUANTUM per M3); ARCH.CL768X09 body + V14CL768X09; PLANE HD48 solo + plane-energy census; XCHANGE XA13 go-gate. This wave jointly answers what the death bracket IS (mapping vs rows vs K vs axis) via the §1 interpretation matrix — every cell lands at or beside a sealed control.

**Wave B — contingent expansions + capability tier:**
ROWSET Arm C + K=8,192 replication; ALIAS ARM-C + K=10,240 twin; BISECT remaining interior rungs per the sequential rule; ARCH.CL768 body + V9CL768, then V12/V10CL768X09 if the bracket entry pulses; PLANE HD32 + composed d=48 behind the ≥12/15 gate; XCHANGE XA09 value cell; NORM-ESTATE Arms M+S (then D grid); ROWSET-LANE single cell (§4 decision input); SC-CORESET SC512 + LRA twin pair.

Every class on this slate closes only by the built children named in its kill gate; every frontier-claimable child is a stock Lfm2ForCausalLM checkpoint with all parameters counted; the two research probes (SC-CORESET, and ALIAS if the tokenizer ruling lands probe-only) are labeled as such in their own spec blocks.