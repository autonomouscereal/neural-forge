# KDA / CSA / HCA / Local-Attention Fold Kernel Design

Updated: 2026-08-04
Status: design only.  No code in this document is implemented.  A
deep-implementer builds from this document; the staged proof plan defines the
first acceptable evidence.  This is the implementation contract for the next
kernel wave, written to the standard of
`docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md` and
`docs/CROSS_WIDTH_RESIDUAL_CROSSING_DESIGN_2026-08-01.md`.

## Problem

The chimera-omni target architecture mixes five sequence-mixer families --
`kda`, `local`, `csa`, `hca`, `gated_mla`
(`chimera-omni/architecture_manifest.json:9-15`, block dispatch at
`chimera-omni/src/chimera_omni/modeling/layers.py:462-473`) -- under a
depth-axis Block AttnRes residual scheme
(`chimera-omni/src/chimera_omni/modeling/chimera.py:101-132`).  Today's fold
kernels cover softmax GQA/RoPE attention
(`src/neural_forge/hf_causal_gqa_attention_fold.py:536`), LFM2 ShortConv
(v4/v5 CPCF lane), gated MLP, cross-width residual charts
(`src/neural_forge/hf_causal_cross_width_charts.py`), and the
attention<->shortconv kind translations
(`src/neural_forge/hf_causal_kind_translation.py`).  None of them can fold a
KDA delta recurrence, a compressed-summary attention branch, a windowed NoPE
attention, or a depth-attention residual trunk.  This document designs the
closed-form fold recipes, cross-width chart behavior, kind-translation
viability, IR mapping, and evidence contract for these families under the
fold-only contract (no gradients, sealed evidence, deterministic
ridge/POD/balanced-truncation solves; `docs/CURRENT_FOLDING_GOAL.md:14-19`).

## Ground truth: what the target families actually are

All statements below are grounded in the executable references, which govern
over any prose (including the K3 tech report's text extraction, whose
parenthesization of the update equation is ambiguous at
`k3_tech_report.txt:229-231`).

### KDA (Kimi Delta Attention) -- `layers.py:88-195`

Per head `h` with key dim `Dk` and value dim `Dv`
(`chimera-omni/src/chimera_omni/config.py:164-168`), state
`S_t in R^{Dk x Dv}` (`layers.py:160-167`):

```text
inputs:   q_t, k_t = L2Norm(SiLU(ConvK4(W_qk x_t)))               layers.py:136-140
          v_t      = SiLU(ConvK4(W_v x_t))                        layers.py:138
          alpha_t  = exp(g_min * sigmoid(exp(A_h) * (W_up W_down x_t + b_alpha)))
                                                                  layers.py:142-148
          beta_t   = sigmoid(W_beta x_t)  in (0,1), per head      layers.py:149
update:   S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
                                                                  layers.py:119-130
readout:  o_t = S_t^T q_t                                         layers.py:183
write:    y_t = W_out [ sigmoid(W_g x_t) (*) HeadRMSNorm(o_t) ]   layers.py:185-191
```

- The decay is **channel-wise diagonal in the key basis** (`alpha_t` in
  `(exp(g_min), 1)^Dk`, `g_min = -5` by default, retention floor
  `exp(-5) ~= 6.7e-3`; `config.py:35`, `k3_tech_report.txt:322-340`).
- The depthwise short conv on q/k/v has kernel 4, bias, and SiLU
  (`layers.py:60-74`, `config.py:34`) -- a *plain* FIR+bias+SiLU, not the
  B/C-gated LFM2 ShortConv.
- The decay projection `W_up W_down` is an explicit low-rank linear map
  (`kda_decay_rank`, `config.py:33`, `layers.py:108-109`).
- The production reference (`modeling_kimi_linear.py:477-541`) matches this
  block modulo kernel packaging: `A_log`/`dt_bias` (`:520-527`), per-head
  `b_proj` write strength (`:529`), full-rank or low-rank output gate
  (`:531-537`), `FusedRMSNormGated` sigmoid gate (`:539-540`), and the fused
  `chunk_kda` / `fused_recurrent_kda` paths (`:609-645`).  Whether the fla
  fused kernels are bit-compatible with the chimera slow reference is
  **untested and unknown**; the proof ladder tests it before any Kimi-Linear
  parent is folded.
- Chunkwise form: recurrent across chunks, parallel within
  (`k3_tech_report.txt:251-283`); the serving state is a fixed `Dk x Dv`
  matrix per head (`k3_tech_report.txt:1160`).

### CSA and HCA -- one module, two strides -- `layers.py:198-228`

`csa` is "compressed sparse attention for medium-range summaries" and `hca`
is "hierarchical compressed attention for long-range summaries"
(`chimera-omni/docs/ARCHITECTURE.md:48-49`).  **Both are the same
`SparseLatentAttention` module**, differing only in stride:
`csa_compress_stride = 64`, `hca_compress_stride = 1024`
(`config.py:60-61`, dispatch `layers.py:466-469`):

```text
local_t   = LocalCausalAttention(x)_t                             layers.py:216
S_b       = mean of x over block b of `stride` tokens             layers.py:208-213
summaries = [0, S_0, S_1, ...]        (zero row prepended)        layers.py:217-219
scores    = (x Wq^T)(summaries Wk^T)^T / sqrt(H)   single head, full width,
                                                   NoPE           layers.py:220-223
mask      = summary j visible to token t iff j <= floor(t/stride) layers.py:224-226
y_t       = local_t + W_out softmax(scores) (summaries Wv^T)      layers.py:227-228
```

Consequences the fold and translation analysis must respect:

1. The global path sees only **completed** blocks: token `t` in block `b`
   reaches summaries of blocks `0..b-1` (summary `j>=1` is the mean of block
   `j-1`; the zero summary is the only key for tokens in block 0).  The
   newest summarized token is up to `2*stride - 1` positions old.
2. The summary branch is **single-head at full hidden width** with plain
   linear q/k/v/out maps and no positional encoding.
3. **Coverage gap:** with `local_window = 256` (`config.py:59`) and HCA
   stride 1024, content between 256 and up to 2047 positions back can be
   invisible to an HCA layer (outside the window, inside an incomplete
   block).  This is an architecture property, not a fold defect; it bounds
   what a translation into HCA can be asked to carry.

### Local attention -- `layers.py:26-57`

Standard softmax GQA with a hard sliding window `(t - s) < window`
(`layers.py:52-54`), fp32 scores, **no RoPE and no q/k norm** -- the whole
architecture is NoPE ("recurrence carries order and recency",
`architecture_manifest.json:17`).  This is the existing GQA kernel operator
class minus RoPE plus a band mask.

### Gated MLA -- `layers.py:231-306` (adjacent, not this wave)

K3-style NoPE MLA: compressed query (optional) and kv latents with RMSNorm
(`layers.py:244-269`), shared extra key channels (`layers.py:289-294`),
full-rank sigmoid output gate (`layers.py:270`, `:306`;
`k3_tech_report.txt:342-366`).  Its fold is latent-factorized softmax
attention: SVD-exact latent truncation plus the existing coupled Q/K + V/O
solve in NoPE mode.  This document scopes gated-MLA only as far as the
NoPE-mode kernel extension (shared with `local`); the latent-path fold gets
its own short design when a gated-MLA parent exists.

### Block AttnRes -- the depth axis changes the residual contract

`DepthAttentionResidual` mixes a learned pseudo-query over RMSNorm-ed depth
sources with softmax weights (`layers.py:309-328`).  The trunk keeps a
`prefix_sum` plus `completed_blocks` snapshots appended every
`attnres_block_size` layers (`chimera.py:109-132`; the Kimi production
analogue is `_apply_attn_res` and its block-residual bookkeeping,
`modeling_kimi_linear.py:973-1046`, `:1075-1088`).  Two facts with fold
consequences:

1. **Branch inputs are not the running residual.**  `attention_input` and
   `feed_forward_input` are depth mixtures of `prefix_sum` and completed
   blocks (`chimera.py:116`, `:127`).  Evidence must seal these mixed rows
   as the operator-norm inputs; the `block_input`-based boundary roles of
   `hf_causal_cross_width_charts.py:57-59` do not describe this trunk.
2. **Depth sources from many depths are compared inside one softmax**, so
   any per-boundary chart disagreement becomes a score perturbation that no
   downstream writer absorbs (see the cross-width section).

## Section 1 -- Per-family fold coordinates and closed-form recipes

Shared conventions.  All fits operate on sealed operator-norm input rows and
target the branch residual write, with frozen fit/selection/validation
partitions, scale-normalized condition-capped ridge, and BF16 full
intermediate replay before any gate is scored -- the v4/v5 numerical policy
(`docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md`, section A.4;
`src/neural_forge/hf_lfm2_cpcf_short_conv_v5.py:35`, `:1052`).  The design
invariant carried over from the GQA kernel: **install the selection
machinery first, replay the actual installed operator, then fit the writer
endpoint-coupled against the residual delta**
(`hf_causal_gqa_attention_fold.py:536` onward: Q/K ridge, probability
replay, then V/O against the endpoint).  For KDA the "installed
probabilities" role is played by the *installed recurrence replay*.

### 1.1 KDA

**Structure to exploit.**  The recurrence is a linear time-varying (LTV)
system in the state: given the input-side signals `(k, v, alpha, beta)`, the
map from state to state is linear (`A_t = (I - beta_t k_t k_t^T)
Diag(alpha_t)`), the writes are rank-one, and the readout is linear in the
state.  Unrolling gives the exact effective-weight identity, the KDA
analogue of `attention_band_decomposition`
(`hf_causal_kind_translation.py:1577`):

```text
o_t = sum_{s<=t} w_h(t,s) v_s
w_h(t,s) = beta_s * q_t^T [ prod_{r=s+1..t} (I - beta_r k_r k_r^T) Diag(alpha_r) ] k_s
```

`w_h` is computable exactly from cached `q,k,alpha,beta` by forward
accumulation with **zero parent forwards** (O(T^2 Dk) per head; trivial at
evidence scale, T ~ 121 x 64 prompts as in the stage-2 runs).  A cached
`o_t` reconstruction check is the KDA evidence-consistency gate, mirroring
the FIR exactness check the LFM2 trace cache performs for ShortConv
(`ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md`, "Existing
machinery", citing `hf_lfm2_circuit_trace_cache.py:3201-3222`).

**Natural fold coordinates.**

1. **Head axis.**  Barycentric complete head contraction, exactly the GQA
   kernel pattern (`hf_causal_gqa_attention_fold.py:202-260`), scored on
   gated-output energy.  Honesty: merged-head recurrences do not commute and
   are never conjugated exactly; the merged head is *refit*, and the error
   is measured at the endpoint.
2. **Key/state-row axis (Dk).**  This is where the balanced-truncation lens
   applies -- with one hard constraint: `Diag(alpha_t)` is diagonal **in the
   key basis**.  An arbitrary rotation of the key axis destroys the
   diagonal-decay parameterization (a rotated decay is a full matrix the
   architecture cannot store), and L2 normalization of q/k is not preserved
   under non-orthogonal maps.  Therefore the v1 recipe uses balanced
   coordinates to *rank*, and channel-aligned selection to *cut*:
   - Build the empirical controllability sketch from cached state-writing
     rows `sqrt(beta_s) k_s` and cached chunk-boundary state snapshots, and
     the observability sketch from cached `q_t` rows weighted by the
     downstream gate/norm/writer sensitivity, via
     `build_balanced_krylov_basis` (`src/neural_forge/balanced_krylov.py:315`,
     snapshot balanced POD with deterministic SVD fallback,
     `balanced_krylov.py:1-10`).
   - Take the **channel-aligned pivot selection** from the balanced modes
     (`canonical_pivot_indices`, `balanced_krylov.py:119-120`), the same
     channel-subset discipline as the deployed ShortConv channel QDEIM
     (`src/neural_forge/hf_lfm2_short_conv_channel_qdeim.py`).  Selection
     keeps `Diag(alpha)` exactly diagonal and keeps the conv taps carryable.
   - The rejected alternative (full balanced rotation plus diagonal decay
     refit) is recorded under Trade-offs.
3. **Value axis (Dv).**  `HeadRMSNorm` carries a per-(head, channel) gain
   (`layers.py:77-85`) and the output gate is channel-wise, so rotations
   conjugate the gain into a matrix the module cannot store.  Same verdict:
   channel-aligned selection ranked by observability through `W_out`
   columns, with the norm gain, gate columns, and conv taps following the
   selection.
4. **Decay rank.**  `W_up W_down` is rank-limited linear; child
   `kda_decay_rank` truncation is an **exact SVD** of the composed map
   restricted to selected channels, no solve required.

**Closed-form fold recipe (KDA -> KDA, per layer, ordered):**

1. Unroll `w_h(t,s)` and verify cached `o_t` reconstruction (fail closed on
   tolerance breach; this seals evidence self-consistency).
2. Barycentric head assignment on gated-output energy.
3. Key/value channel selection per child head (balanced ranking, pivot
   selection, as above).  Carry conv taps and biases for selected channels
   (depthwise conv rows are channel-aligned: exact carry, no solve).
4. **Q/K refit:** ridge from child norm rows to selected-channel
   pre-normalization targets; replay the child conv/SiLU and L2
   normalization exactly rather than fitting through them (the pre-RoPE
   target discipline of the GQA kernel, transplanted).
5. **Decay refit, closed form by inverse chain:** the parent log-decay
   target `g* = g_min * sigmoid(exp(A_h) z)` is cached; compute
   `u = inverse_sigmoid(g*/g_min)` (finite because `g*` is strictly
   interior; the clamp policy is explicit and sealed), pick child `A_h` to
   match the per-head scale of `u`, then ridge-fit the low-rank
   `W_up W_down` and select bias rows.  Deterministic linear algebra
   throughout.
6. **Beta refit:** inverse-sigmoid targets from parent `beta`, per-head
   ridge on `W_beta`.  Merged heads with high beta dispersion are a
   measured defect (threshold chosen on the selection partition; fail
   closed above it).
7. **Installed-recurrence replay:** run the child recurrence on sealed,
   sequentially recaptured child rows in FP64 -- the analogue of replaying
   installed softmax probabilities.
8. **Gate + writer solve:** inverse-sigmoid ridge for `W_g` against the
   parent gate pattern, then endpoint-coupled ridge for `W_out` from
   replayed gated-normed child outputs to the branch residual delta (or the
   chart-transported target when crossing width).
9. **BF16 replay of every intermediate** (conv, normalize, recurrence,
   gate, writer), then acceptance diagnostics.  Long-horizon state-drift
   gate: state error at successive chunk boundaries must not grow
   superlinearly on validation rows (growth-rate policy constant);
   `linear_rollout` / `rollout_error_report` are the harness
   (`balanced_krylov.py:775`, `:845`).

**What is exact and what is measured.**  Conv-tap carry, decay-rank SVD, and
the unroll identity are exact.  Channel selection, head merging, and every
inverse-sigmoid ridge are approximations whose error is measured at the
endpoint on validation rows.  Because the system is LTV, Hankel-tail bounds
are *empirical* (snapshot POD), not a priori guarantees -- stated, not
papered over.

### 1.2 CSA / HCA

**Structure.**  Two additive sub-branches share one residual write
(`layers.py:228`).  The summary branch is *plain single-head NoPE softmax
attention over a fixed linear downsampling* of the sequence: mean pooling is
a linear operator `P_stride`, keys/values are linear images of pooled rows,
and everything the coupled GQA kernel does (score ridge, probability replay,
endpoint V/O) applies with the key/value row set replaced by pooled rows and
the triangular mask replaced by the block-causal mask
(`layers.py:224-226`).

**Natural fold coordinates.**  The summary branch has no head or internal
width to contract -- q/k/v/out are `H x H` (`layers.py:203-206`); its only
coordinates are the residual chart itself.  The local sub-branch has the
standard GQA head coordinates.  Stride is a **contract constant**, not a
fold coordinate: changing stride changes which keys exist and is a kind
translation (section 3), never a silent fold parameter.

**Closed-form fold recipe (per layer, ordered):**

1. Split the cached branch delta exactly into `local_delta` and
   `summary_delta` (both captured per sub-branch; their sum must
   reconstruct the sealed write -- fail closed otherwise).
2. Fit the local sub-branch with the extended GQA kernel (NoPE + window
   flags, section 1.3) against `local_delta`.
3. Fit the summary sub-branch: ridge `Wq`/`Wk` from child rows and child
   *pooled* rows to desired score targets (parent scores are cached; the
   zero summary row is preserved exactly because linear maps send zero to
   zero); replay installed probabilities under the block-causal mask; fit
   `Wv`/`W_out` endpoint-coupled against `summary_delta`.
4. **Bounded alternation (2 rounds):** refit summary V/out against
   `delta - replayed_local`, then local V/O against
   `delta - replayed_summary`.  Each sub-step is an exact ridge given the
   other branch, so the fit objective is monotone nonincreasing on fit
   rows; rounds are fixed and chosen on the selection partition.
5. BF16 replay and diagnostics.  Per-sub-branch retained energy is reported
   separately so a summary-branch collapse cannot hide inside a good local
   fit.

### 1.3 Local / windowed attention

**Do not write a new kernel.**  Extend
`compile_hf_causal_gqa_attention_fold` with two policy fields:

- `attention_window: int | None` -- threads a band constraint into the mask
  construction of `causal_attention_probabilities`
  (`hf_causal_gqa_attention_fold.py:310-346`), exactly where the causal
  tril mask is built (`:324-327`).
- `position_encoding: "rope" | "nope"` -- `nope` makes `apply_rope` /
  `invert_rope` identity passthroughs (`:295-302`), skips RoPE pair
  selection and contraction (`:389-408`), and drops the even-head-dim
  requirement (`:397-400`), which exists only for rotation pairing.

Everything else -- barycentric contraction, Q/K ridge to desired
pre-rotation targets, installed-probability replay, endpoint-coupled V/O,
sealed reports -- is unchanged.  Chimera `local` folds with
`attention_window = cfg.local_window`, `position_encoding = "nope"`; the
gated-MLA NoPE score path reuses the same flag later.

**Lost freedom to name now:** under NoPE there is no RoPE cosine-ladder
mechanism, so the synthesized position-selective desired-evidence trick of
the committed Direction B (`_rope_band_preimage`,
`hf_causal_kind_translation.py:882`) is unavailable for NoPE children.
Position selectivity in chimera children must come from window masks, block
masks, or KDA decay -- never from score synthesis.  This single fact drives
several verdicts in section 3.

## Section 2 -- Cross-width behavior under residual boundary charts

Chart contract recap (`src/neural_forge/hf_causal_cross_width_charts.py:1-33`):
one orthonormal encoder `E_b in R^{Hp x Hc}` per residual-write boundary from
balanced controllability/observability POD, one cycle-anchored ridge decoder
`D_b in R^{Hc x Hp}` (`:982-1005`), four measured defects with sealed policy
bounds (cycle, incoming transition, absorption demand vs writer rank, norm
commutation; `:1008-1098`), contraction-only (`:1147-1151`), and the
sequential branch target `Y = h_next E_next - hhat_b_actual` with live child
recapture (`:1432-1520`, `:1569-1660`).  Consumer weights fold as
right-multiplication by the chart (`fold_consumer_weight`, `:541`).

Per family:

- **KDA.**  The recurrent state `S` lives in head-internal `(Dk, Dv)`
  coordinates -- it is **chart-invariant**.  Charts touch only the
  residual-facing maps: consumers `W_q, W_k, W_v, W_down (decay), W_beta,
  W_g` and producer `W_out`.  Because every consumer fit is a ridge from
  child rows of any width (the width-agnostic property the chart module
  relies on, `hf_causal_cross_width_charts.py:3-8`), KDA cross-width
  folding is target construction plus the section 1.1 recipe; no new
  solver.  The internal conv is downstream of the projection and needs no
  chart.
- **CSA/HCA.**  All four summary maps are residual-width square: `Wq` is a
  consumer at the input boundary; `Wk`/`Wv` consume **pooled** rows;
  `W_out` is a producer at the output boundary.  Mean pooling commutes with
  any linear chart -- mean(h) E = mean(h E) -- so the *same* boundary chart
  serves raw and pooled rows exactly, and the zero summary maps to zero.
  This commutation is a genuinely clean property: no pooled-row chart
  variant is needed.  The local sub-branch behaves as the existing GQA
  branch under charts.
- **Local.**  Identical to the deployed GQA cross-width behavior: Q/K/V
  ridge from child-width rows to head targets; RoPE machinery absent
  (NoPE), so the head-dim pair-contraction constraint disappears; the
  window is width-independent.
- **Block AttnRes -- the hard constraint.**  `DepthAttentionResidual`
  compares depth sources from many different depths inside one softmax
  (`layers.py:323-327`).  With per-boundary charts, source `d` arrives in
  chart `E_d` while the pseudo-query and norm gains are single vectors;
  chart transitions between depth sources become score perturbations that
  **no downstream writer absorbs** -- the absorption-demand machinery
  (`hf_causal_cross_width_charts.py:1079-1097`) has no writer to charge
  here.  v1 rule, fail-closed: chimera children with
  `residual_mode = "block_attnres"` use **one global chart** (`E_b == E`
  for all boundaries; transition defects identically zero; only cycle and
  norm-commutation defects remain live).  Per-boundary charts for AttnRes
  trunks require extending the transition-defect algebra to the depth
  mixer and are explicitly out of scope.  The AttnRes module itself folds
  as a consumer: transported anchor `query_c = E^T query_p`, refit by
  ridge on sealed depth-source scores; RMSNorm gains fold like other norm
  gains.  New boundary roles are required because branch inputs are depth
  mixtures (`chimera.py:116`, `:127`): `attnres.attention_in`,
  `attnres.mlp_in`, plus depth-source snapshot boundaries at each
  `attnres_block_size` interval.

## Section 3 -- Kind-translation viability

Discipline as in the committed design: state exactly which mass is
representable, measure the rest, route it to a carrier, fail closed without
one.  Remainder sealing, carrier targets, and checkpoint gates reuse the
existing machinery unchanged (`seal_kind_translation_remainder`,
`carrier_endpoint_target_with_remainder`,
`evaluate_carrier_routing_checkpoint`;
`hf_causal_kind_translation.py:2704`, `:2766`, `:2786`).

### Verdict matrix

| source -> target | verdict | representable mass | remainder / obstruction |
|---|---|---|---|
| shortconv -> KDA | honest, exact-in-limit | frozen-gate FIR taps as pole mixtures | gate covariance rho_2; pole tail beyond lag L |
| shortconv -> local (NoPE) | fail-closed by design | lag-0 self + window-uniform patterns | no NoPE position-selection mechanism |
| KDA -> global softmax (GQA/MLA) | measured-partial, primary lane | signed unrolled weights via V/O | per-row mass normalization variance |
| KDA -> local | conditional per layer | in-window unrolled mass | slow-decay heads (alpha near 1) |
| KDA -> shortconv | measured-partial, hardest | in-band separable mass | state memory beyond L; stage-2 rho_far precedent |
| softmax-GQA -> KDA | measured-partial, the new balanced-truncation lane | exponential-mixture lag profiles + column-mass writes | erase-term coupling; query-specific selection |
| softmax-GQA -> CSA/HCA | honest, promising | window mass + far block-mean mass | far within-block deviation; HCA coverage gap |
| CSA/HCA -> softmax-GQA | exact-in-limit | both sub-branches; heads are additive | softmax saturation leak; block-constant score fit |
| local <-> global GQA | restriction / relaxation | all (local to RoPE-global); band split (global to local) | NoPE child cannot synthesize a window cutoff; far-mass routing |
| HCA -> CSA (down-stride) | exact-in-limit | any coarse pattern refines | none structural |
| CSA -> HCA (up-stride) | measured-partial | re-pooled block means | fine-block structure; coverage gap |
| KDA <-> CSA/HCA | not designed this wave | compose via a GQA intermediate | two-step remainder accumulation |

### The load-bearing analyses

**shortconv -> KDA** (replaces the strained shortconv -> attention route
for chimera children).  Freeze/linearize the LFM2 gates exactly as in the
committed Direction B (`linearize_short_conv_branch`,
`hf_causal_kind_translation.py:705`; `rho_2` measured and routed).  A KDA
head with *constant* key -- achievable exactly by zeroing `W_k` so
`k = L2Norm(SiLU(conv_bias))`, a constant unit vector (the conv has a bias,
`layers.py:64-70`) -- collapses the per-head recurrence to a single-pole
scalar IIR per value channel: `m_t = a m_prev + beta v_t` with effective
pole `a = (1 - beta) * sum_i k_i^2 alpha_i`.  One pole per head.  An L-tap
FIR is matched at lags `0..L-1` by L distinct poles through a Vandermonde
solve (closed form; conditioning reported), requiring `num_heads >= L`
(L = 3 for LFM2; chimera default heads = 4,
`architecture_manifest.json:36` -- holds, else fail closed).  The tail
beyond lag L-1 is `sum_i c_i a_i^lag`, minimized by poles near the
retention floor `exp(-5)`; tail mass is *measured and reported*, never
assumed zero.  This mirrors the one-head-per-lag construction of Direction
B with poles instead of RoPE cosines -- and unlike Direction B it works in
a NoPE architecture.

**softmax-GQA -> KDA** (the delta state under the balanced-truncation
lens).  Two-stage closed form:

1. **Lag-profile stage.**  The parent mean lag profile
   `p_bar(l) = mean_t P(t, t-l)` (from `attention_band_decomposition`
   diagnostics, `hf_causal_kind_translation.py:1577`) is fitted by a sum of
   `num_heads` exponentials -- balanced truncation of the 1-D Hankel matrix
   built from `p_bar`, Ho-Kalman/Prony in the deterministic SVD form
   `build_balanced_krylov_basis` already implements
   (`hankel_singular_values` sealed in its report,
   `balanced_krylov.py:101-106`).  Poles map to per-head baseline decays
   via the inverse chain of recipe step 5; the Hankel singular-value tail
   is the sealed a priori bound *for the mean profile only*.
2. **Content stage.**  Desired write mass per key position -- column mass
   `m_s = sum_t P(t,s)` -- sets inverse-sigmoid `beta` targets; `W_k` is
   ridge-fitted to separate high- from low-column-mass content; `q`, gate,
   and writer follow recipe steps 4-8.

What is *not* fitted: the erase-term `(I - beta k k^T)` coupling and
query-specific (not merely content-specific) selection.  Both land in the
measured endpoint remainder and route downstream.  Expectation to test,
not assert: because KDA memory is exponential rather than hard-L, the
routed remainder should sit far below the stage-2 attention-to-conv result
(`rho_far` 0.52-0.70 on real LFM2,
`reports/kernel-stage2-20260801/kernel-stage2-report.md:46-49`).

**softmax-GQA -> CSA/HCA** (the carrier the conv translation lacked).
Three-way exact split of the cached delta, generalizing the Direction A
band split:

```text
Delta = Delta_window + Delta_far_blockmean + Delta_far_deviation
Delta_far_blockmean_t = sum over heads, blocks of
    Pbar_h(t,b) * Vbar_h_b * W_o[h]^T
where Pbar sums probabilities over each completed block and Vbar is the
block-mean value row
```

`Delta_window` goes to the local sub-branch (existing kernel).
`Delta_far_blockmean` is exactly the operator class of the summary branch:
block-constant weights times block-mean values, which is what mean pooling
computes.  `Delta_far_deviation` -- the within-block covariance of
probabilities and values -- is **provably outside the class** and routes to
a carrier.  For HCA, the coverage-gap mass (outside the window, inside the
incomplete block) is additionally unrepresentable by construction and is
reported as its own energy fraction.  All fractions are computable exactly
from cached probabilities and values with zero parent forwards.

**KDA -> softmax attention.**  The unrolled `w_h(t,s)` is exact from
evidence.  Softmax rows are positive and normalized; `w_h` rows are signed
and unnormalized.  Signs and magnitudes move into the V/O solve (the
committed design precedent for sign placement), but *per-row total mass*
variation has no home in a fixed linear V/O -- its variance is the measured
obstruction, dual to the softmax-denominator coupling of Direction A item
3.  Fit path: low-rank factorization of the shifted log-magnitude weight
matrix as desired scores, synthesized-evidence entry into the unchanged
GQA kernel (`attention_evidence_from_projections`,
`hf_causal_kind_translation.py:1677`); the kernel replay scores the truth.

## Section 4 -- IR mapping, evidence capture, fail-closed conditions

### CircuitKind assignments (`src/neural_forge/dynamic_circuit_ir.py:42-55`)

| chimera block kind | CircuitKind | ports / edges | key attributes |
|---|---|---|---|
| kda | `RECURRENT_MIXER` (`:50`) | `RECURRENT_STATE` port (`:77`), `STATE` edge (`:63`) | heads, key_head_dim, value_head_dim, decay_rank, conv_kernel_size, gate_lower_bound |
| local | `ATTENTION` (`:48`) | standard | window, position_encoding = nope, query/kv heads |
| csa / hca | `ATTENTION`, one composite node | standard | summary_stride, local_window, position_encoding = nope |
| gated_mla | `ATTENTION` | standard | kv_latent, q_latent, extra_head_dim, output_gate = full_rank, nope |
| LFM2 shortconv | `SHORT_CONV` (`:49`) | existing | existing |

CSA/HCA is deliberately **one** operator node, not a local node plus a
summary node: the family-neutral branch contract admits exactly one norm,
one operator, one ordered residual write per branch
(`src/neural_forge/dynamic_fold_planner.py:773-786`), and the two
sub-branches share one residual write.  KDA's internal short convs are
internal structure of the `RECURRENT_MIXER` node (attributes), not separate
`SHORT_CONV` nodes, for the same reason.  The planner gains new arms in the
branch contract keyed on kind and attributes, and new method ids:

```text
causal_kda_recurrent_layer            causal_summary_attention_layer
causal_windowed_attention_layer       (policy variant of the GQA method)
causal_short_conv_to_kda_layer        causal_gqa_to_kda_layer
causal_kda_to_gqa_layer               causal_gqa_to_summary_attention_layer
```

### Evidence capture (mirrors `HFCausalLayerEvidence` field style,
`src/neural_forge/hf_causal_circuit_evidence.py:214-233`; one parent
forward, sealed, `:1-9`)

New capture dataclasses, per layer, all rank-3 `[batch, seq, .]` unless
noted:

**`ChimeraKDALayerEvidence`** -- `block_input` (depth-mixed attention
input), `attention_norm_output`, `query_rows` / `key_rows` / `value_rows`
(post-conv, post-SiLU, post-normalize, per head), `pre_conv_query` /
`pre_conv_key` / `pre_conv_value` (so conv taps can be verified),
`retention_rows` (alpha), `decay_logit_rows`, `beta_rows`,
`recurrent_output_rows` (o_t pre-norm), `state_snapshots` (rank-4
`[batch, heads, Dk, Dv]` at fixed chunk boundaries plus final;
`layers.py:152-191` exposes `forward_with_state`), `gate_rows`,
`attention_delta`, `block_output`.

**`ChimeraSummaryAttentionLayerEvidence`** -- `block_input`,
`attention_norm_output`, `local_delta`, `summary_delta` (their sum must
equal the sealed branch write), local sub-branch Q/K/V rows and per-head
probabilities, `raw_summaries`, `summary_scores`,
`summary_probabilities`, `block_ids`, `summary_value_rows`.

**`ChimeraLocalAttentionLayerEvidence`** -- the existing GQA field set plus
`window` and the applied band mask.

**`ChimeraDepthResidualEvidence`** (trunk-level) -- depth-source stacks and
per-application softmax weights (the model already exposes both as aux:
`attnres_depth_sources`, `attnres_output_weights`, `chimera.py:183-186`),
`prefix_sum` snapshots at every `attnres_block_size` boundary, and the
depth-mixed `attention_input` / `feed_forward_input` rows per layer.

Capture prerequisite: `build_decoder_layout` /
`capture_hf_causal_circuit_evidence` currently reject anything but
llama/qwen3-shaped layouts (`hf_causal_circuit_evidence.py:152-155`), which
is exactly what blocked the LFM2 cross-width stage 2
(`reports/kernel-stage2-20260801/kernel-stage2-report.md:14-18`).  Chimera
gets its own capturer; it does not squeeze into the llama record.

### Fail-closed conditions (each aborts before any child mutation, under
the snapshot-transaction pattern of
`src/neural_forge/dynamic_causal_operator_promotion.py`)

1. KDA unroll fails to reconstruct cached `o_t` within tolerance, or a
   state-snapshot replay diverges (evidence inconsistency).
2. Inverse-sigmoid targets saturated beyond the sealed clamp policy
   (alpha at a bound, beta at 0/1, gate at 0/1) on more than the policy
   fraction of rows.
3. Merged-head beta or decay dispersion above the selection-partition
   threshold.
4. Vandermonde / pole-fit conditioning above cap, or `num_heads` below the
   required pole/lag count.
5. CSA/HCA: local+summary split does not reconstruct the sealed delta;
   summary probability replay places less than the policy fraction of mass
   on assigned blocks; stride mismatch without an explicit translation
   method.
6. AttnRes trunks: any plan requesting per-boundary charts (v1 admits only
   the global chart); depth-source count mismatch between evidence and
   child.
7. Translation-specific: NoPE lag-selectivity requests (shortconv ->
   NoPE-local); `rho`-style remainder above threshold with no downstream
   carrier (for chimera children the guaranteed global carrier is the
   final gated-MLA layer when `force_final_gated_mla` is set,
   `config.py:152-155` -- plans without any global layer after a lossy
   conversion fail closed).
8. The standing set: partition overlap, non-finite tensors, condition-cap
   exhaustion, validation endpoint at or above the no-branch baseline
   after BF16 replay.

### Persistence

Sealed canonical-JSON reports with `report_sha256` self-seals and
content-addressed tensor fingerprints, repo convention unchanged.  New
schema identifiers:

```text
neural-forge-chimera-kda-fold-v1
neural-forge-chimera-summary-attention-fold-v1
neural-forge-causal-windowed-attention-fold-v1   (GQA report + window/nope fields)
neural-forge-causal-short-conv-to-kda-fold-v1
neural-forge-causal-gqa-to-kda-fold-v1
neural-forge-causal-kda-to-gqa-fold-v1
neural-forge-causal-gqa-to-summary-attention-fold-v1
neural-forge-chimera-depth-residual-chart-note-v1  (global-chart attestation)
```

Remainder bundles reuse `neural-forge-causal-kind-translation-remainder-v1`
unchanged (`hf_causal_kind_translation.py:127-133`).

## Section 5 -- Implementation plan

### Modules and signatures

All paths under
`C:/Users/cereal/Documents/ai training/neural-forge/src/neural_forge/`.

1. **`hf_chimera_circuit_evidence.py`** (new)
   - `capture_chimera_circuit_evidence(model, model_inputs, *, chunk_size, fit_fraction, selection_fraction) -> ChimeraCircuitEvidence`
   - dataclasses of section 4; one forward via `forward_with_state` and aux
     depth outputs; seal and verify in the `hf_causal_circuit_evidence.py`
     style.
2. **`hf_causal_gqa_attention_fold.py`** (extend, no fork)
   - policy fields `attention_window: int | None = None` and
     `position_encoding: str = "rope"`; mask threading in
     `causal_attention_probabilities`; NoPE passthrough for
     `apply_rope`/`invert_rope`; the report gains the two fields.
3. **`hf_causal_kda_fold.py`** (new)
   - `kda_effective_weights(evidence, layer_index) -> KDAWeightUnroll`
     (exact unroll plus reconstruction check)
   - `select_kda_channels(unroll, evidence, *, policy) -> KDAChannelSelection`
     (balanced ranking via `build_balanced_krylov_basis`, pivot selection)
   - `compile_hf_causal_kda_fold(evidence, layer_index, child_geometry, *, policy, chart=None, recapture=None) -> HFCausalKDAFold`
     (recipe steps 2-9; endpoint-coupled writer; sealed report)
   - `replay_kda_branch(fold, rows, *, dtype)` (FP64/BF16 replay)
4. **`hf_causal_summary_attention_fold.py`** (new)
   - `split_summary_branch_delta(evidence, layer_index) -> SummarySplit`
   - `compile_hf_causal_summary_attention_fold(evidence, layer_index, child_geometry, *, policy, chart=None, recapture=None) -> HFCausalSummaryAttentionFold`
     (calls the extended GQA kernel for the local sub-branch; bounded
     2-round alternation)
5. **`hf_causal_kda_kind_translation.py`** (new)
   - `compile_short_conv_to_kda_translation(linearization, child_geometry, *, policy)`
     (constant-key pole construction; Vandermonde solve; tail diagnostics)
   - `attention_lag_pole_fit(band, *, heads, policy) -> LagPoleFit`
     (Hankel/Prony stage via `build_balanced_krylov_basis`)
   - `compile_gqa_to_kda_translation(band, evidence, child_geometry, *, policy)`
   - `compile_kda_to_gqa_translation(unroll, child_geometry, *, policy)`
     (feeds `attention_evidence_from_projections` and the unchanged kernel)
   - `compile_gqa_to_summary_attention_translation(decomposition, child_geometry, *, policy)`
     (three-way split; remainder sealed via
     `seal_kind_translation_remainder`)
6. **`hf_causal_cross_width_charts.py`** (extend)
   - AttnRes boundary roles; a `global_chart_only` policy flag that
     compiles one shared encoder/decoder pair and stamps every boundary
     with it (transition defects zero by construction, attested in the
     report).
7. **`dynamic_fold_planner.py` / `dynamic_causal_operator_promotion.py` /
   `dynamic_fold_registry.py`** (extend)
   - branch-contract arms for `RECURRENT_MIXER` and attention attributes;
     the method ids of section 4; `FoldOperator` classes following
     `_CausalBranchOperator` (bind, compile, snapshot, install,
     transaction) exactly.

### Test strategy and the toy-to-real proof ladder

Stage 0 -- unit toys (new `tests/test_hf_causal_kda_fold.py`,
`tests/test_hf_causal_summary_attention_fold.py`,
`tests/test_hf_causal_kda_kind_translation.py`):

- constant-key KDA head equals the closed-form scalar IIR; Vandermonde
  pole match reproduces a random L-tap FIR at lags 0..L-1 with the
  predicted tail; unroll identity equals the slow recurrence to 1e-12.
- CSA toy: block-constant synthetic attention folds into the summary
  branch with near-zero validation error; adversarial within-block
  deviation shows up entirely in `Delta_far_deviation` and the
  fail-closed path fires.
- windowed/NoPE kernel flags: the window mask restricts probabilities
  exactly; the NoPE path matches expectations on synthetic evidence with
  no rotation applied.
- AttnRes: global-chart fold of a 2-block toy trunk reproduces depth
  weights; a deliberate per-boundary chart request fails closed.

Stage 1 -- tiny chimera, same width, kind-preserving.  The reference
config (H=128, 4 layers, pattern kda/local/csa/hca,
`architecture_manifest.json:33-47`) with random weights: capture, fold
each branch into an identical-geometry child, expect endpoint error at
numerical noise.  This proves capture plus replay identity per family
before any compression claim.

Stage 2 -- tiny chimera, cross-width (H=128 to 96) under the global
chart, plus head/channel contraction (heads 4 to 3, Dk cut by pivot
selection).  Full defect table and per-branch receipts in the stage-2
report style (`reports/kernel-stage2-20260801/kernel-stage2-report.md`).

Stage 3 -- real LFM2 shortconv to KDA on the **same sealed 64-prompt
cache and the same layers (L1, L3)** the committed stage 2 used, so the
verdict is directly comparable to shortconv to attention (val endpoint
RMSE 0.769-0.825, `kernel-stage2-report.md:31-36`).  Beating or matching
those numbers with smaller routed remainder is the acceptance bar for the
KDA solve machinery on real evidence.

Stage 4 -- real softmax attention to KDA on SmolLM2-135M evidence (llama
layout is supported, `hf_causal_circuit_evidence.py:152-155`):
lag-profile pole fits, remainder energies, direct comparison against the
stage-2 attention-to-conv `rho_far` 0.52-0.70 baseline.  Also the
attention-to-CSA three-way split measured on the same layers.

Stage 5 -- trained chimera parent (from the chimera repo tiny-proof
training path) folded cross-width; then, only if the fla-core CPU replay
equivalence proof passes, a Kimi-Linear parent
(`modeling_kimi_linear.py:609-645`; `transpose_state_layout` and
in-kernel L2 norm and gate options make bit-equivalence a real question,
flagged unknown).

Every stage publishes sealed receipts, keeps validation untouched until
its operator is frozen, and honors the ban on regenerating sealed
evidence (`docs/CURRENT_FOLDING_GOAL.md`, evidence rules; on-policy
extension rounds only under
`docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md`).

### Measured-risk section -- what stage 2 teaches about window and state limits

- **Hard windows starve.**  Real LFM2 attention layers held 52-70% of
  their delta energy beyond lag 2 (`rho_far`,
  `kernel-stage2-report.md:46-49`).  Any hard-window target class (FIR,
  small local windows) will keep failing on such layers.  KDA
  exponential state and the CSA summary branch are precisely the two
  escape hatches this wave adds; the design predicts, and stages 3-4
  must measure, materially lower routed remainder.
- **Compiled is not converged.**  Direction B compiled everywhere yet
  validation endpoint RMSE stayed near 0.77-0.83
  (`kernel-stage2-report.md:31-36`).  Expect the same regime here;
  acceptance is always relative to the explicit baselines (zero,
  no-branch, exact-part-only), never absolute.
- **Rows versus columns.**  4041 fit rows against 3072 free columns put
  the stage-2 separable floor near interpolation
  (`kernel-stage2-report.md:114-119`).  KDA per-head solves add
  `Dk + Dv + decay_rank` column blocks per head; row budgets must be
  planned (and extended on-policy) before the real-model stages, and
  fit-versus-validation gaps read on validation only.
- **Conditioning owns the ridge.**  Stage 2 saturated the 1e4 condition
  cap on every conv sub-solve (`kernel-stage2-report.md:110-113`).  The
  inverse-sigmoid decay, beta, and gate fits operate near saturation
  regions where the sigmoid Jacobian collapses; the clamp policy plus
  condition-capped equilibrated ridge
  (`hf_lfm2_cpcf_short_conv_v5.py:1052`) is mandatory, and cap
  saturation is reported per sub-solve.
- **Recurrence replay is the compute hot spot.**  Order T x Dk x Dv per
  head per candidate, heavier than FIR replay; chunk-boundary snapshots
  amortize long-horizon checks.  Real-model stages run on the AI server
  CPU budget alongside CPCF runs, nice-d, exactly like stage 2.

## Trade-offs

- **Channel-aligned selection for KDA state reduction (chosen)** versus
  full balanced rotation with diagonal decay refit (rejected): rotation
  breaks the diagonal decay, the L2 normalization, and conv-tap carry all
  at once; three stacked refits buy a marginally better subspace at the
  price of unmeasured coupling error.  Selection keeps three exact carries
  and uses the balanced machinery for what it is provably good at here --
  ranking.
- **Reuse the GQA kernel with window/NoPE flags (chosen)** versus a
  bespoke windowed-NoPE compiler (rejected): the flags touch about four
  code sites; a fork duplicates roughly 850 audited lines.
- **One composite IR node for CSA/HCA (chosen)** versus two operator
  nodes (rejected): two nodes break the one-operator-per-branch contract
  (`dynamic_fold_planner.py:773-786`) and falsely suggest two independent
  residual writes.
- **Global chart for AttnRes children (chosen)** versus per-boundary
  charts with a depth-mixer transition theory (rejected for v1): the
  theory does not exist yet; shipping without it would inject
  unabsorbable score perturbations -- the A0025 lesson replayed on the
  depth axis.
- **Sequential-then-alternate CSA solve (chosen)** versus one joint
  local-plus-summary solve (rejected): the joint problem is bilinear
  across sub-branches with no closed form; bounded alternation of exact
  ridges is the established v4/v5 pattern.
- **shortconv to KDA as the chimera-lane translation (chosen)** versus
  forcing shortconv to NoPE-local (rejected): the NoPE class provably
  lacks lag selectivity; designing around that would rebuild the v3
  noise-fitting failure.

## Risks, approvals, and what this does not claim

- **Read/write surface.**  chimera-omni and `modeling_kimi_linear.py` are
  read-only inputs.  All new code lands in the neural-forge modules and
  tests listed above; no production service, credential, or vault entry
  is touched; installation of any fold stays transactional behind the
  existing promotion snapshot pattern.  Real-model stages consume
  existing sealed caches (IO-heavy; schedule on the AI server next to
  CPCF runs) and must not regenerate sealed evidence.
- **Content selection remains the point of attention.**  Every
  translation here can score well on RMSE while smoothing away *which*
  token was selected; manual whole-response review remains the verdict
  (`docs/CURRENT_FOLDING_GOAL.md`, review contract).  This is the top
  risk, unchanged from the committed design.
- **Unknowns, stated plainly.**  (1) fla `chunk_kda` versus the chimera
  slow recurrence: bit-equivalence untested; gate order inside the fused
  kernel may differ.  (2) The K3 report update-equation parenthesization
  is ambiguous in the text extraction (`k3_tech_report.txt:229-231`); the
  chimera executable form (`layers.py:119-130`) governs this contract.
  (3) No trained chimera parent of nontrivial scale exists yet; stages
  1-2 prove machinery, not intelligence.  (4) Whether KDA exponential
  memory actually captures real softmax far mass is the central empirical
  question of stage 4 -- this design gives it a measurement, not an
  answer.
- **No whole-model intelligence claim.**  No child produced by this
  design is accepted on metrics alone, and no fold-only rule is relaxed:
  no gradients, no optimizer, no parent runtime sidecar, no unsealed
  resampling, no weakened validation gate.
