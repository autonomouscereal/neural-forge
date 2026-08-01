# Attention <-> ShortConv Operator-Kind Translation Design

Updated: 2026-08-01
Status: design only.  No code in this document is implemented.  A
deep-implementer builds from this document; the staged proof plan defines the
first acceptable evidence.

## Problem

The family-neutral per-branch fold operators cover attention-to-attention
(`src/neural_forge/hf_causal_gqa_attention_fold.py`) and gated-MLP-to-gated-MLP
(`src/neural_forge/hf_closed_form_causal_circuit_fold.py:1186`), but no
operator translates between operator KINDS.  The planner's family-neutral
layer contract only splits layers into ATTENTION and GATED_MLP branches
(`src/neural_forge/dynamic_fold_planner.py:764-771`) and otherwise emits the
`cross_family_closed_loop_layer_operator` gap
(`src/neural_forge/dynamic_fold_planner.py:2592-2599`).  Consequently an
all-attention parent (Qwen3-style) cannot fold into an LFM2-shaped child whose
layers are mostly gated ShortConv, and an LFM2 parent cannot fold into an
all-attention child.  This document designs both closed-form translations:
attention-to-ShortConv and ShortConv-to-attention.

The local LFM2.5-230M parent has 14 layers ordered
`conv, conv, attn, conv, attn, conv, attn, conv, attn, conv, attn, conv,
attn, conv` — 8 ShortConv + 6 attention
(`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:55-59`).  Other LFM2
sizes use different mixes (e.g. 10 conv + 6 attention); nothing in this design
depends on a specific count or on FIR length 3 specifically — only on
`fir_length = L >= 1`.

## Existing machinery this design builds on (read before implementing)

- **ShortConv semantics** are contract-pinned: `(B_t, C_t, x_t) =
  chunk_3(in_proj(h_t))`, `r_t = B_t * x_t`,
  `v_t = conv_bias + sum_lag k_lag * r_(t-lag)`,
  `delta_t = out_proj(C_t * v_t)`, with `k_lag = conv.weight[:, 0, L-1-lag]`
  (`src/neural_forge/hf_lfm2_canonical.py:100-105`, executable slow-path
  reference at `hf_lfm2_canonical.py:726-761`; `in_proj` is `[3H, H]`,
  `out_proj` is `[H, H]`, `hf_lfm2_canonical.py:820-825`).  Heterogeneous
  internal channel width and factorized output writers are already
  materializable (`hf_lfm2_canonical.py:111-156`).
- **Attention branch kernel**: `compile_hf_causal_gqa_attention_fold`
  (`src/neural_forge/hf_causal_gqa_attention_fold.py:470-722`) fits Q/K by
  ridge to desired pre-RoPE targets (`:605-613`), replays the *installed*
  softmax probabilities, then fits V/O against the residual endpoint using
  those actual probabilities (`:646-657`).  Its replay path (`:422-467`) and
  `causal_attention_probabilities` (`:302-338`) are the exact target-side
  semantics.  Head contraction is barycentric and complete (`:113-127`,
  `:222-250`); RoPE pair contraction rejects expansion and odd dims
  (`:377-380`).
- **Cached evidence.**  Family-neutral parents: `HFCausalLayerEvidence` seals
  per-layer `block_input`, `attention_norm_output`, Q/K/V projection outputs,
  `attention_delta`, `mlp_delta`, `block_output`
  (`src/neural_forge/hf_causal_circuit_evidence.py:215-233`), from exactly one
  parent forward (`hf_causal_circuit_evidence.py:1-9`).  LFM2 parents: the
  circuit trace cache seals `residual.pre`, `operator.input`,
  `residual.post_operator`, `mlp.*`, `residual.post_layer`
  (`src/neural_forge/hf_lfm2_circuit_trace_cache.py:2811-2820`); attention
  internals including per-head `attention.probabilities`, `attention.v`,
  key/value histories and `qk_scores`
  (`hf_lfm2_circuit_trace_cache.py:2966-3039`); and ShortConv internals
  `short_conv.b/c/x/bx`, context, out-projection input, lag histories,
  `pre_bx_state`, the FIR kernel, and an exactness check that the cached
  context reconstructs from the FIR
  (`hf_lfm2_circuit_trace_cache.py:3098-3106`, `:3201-3222`; conv cache
  states at `:2586-2599`, `:3459-3501`).
- **ShortConv solve machinery.**  The v4/v5 CPCF lane already implements
  composed-endpoint short-conv fitting with source priors, signed channel
  directions, deterministic variable-projection updates, condition-capped
  equilibrated ridge, and target-BF16 replay before promotion
  (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:227-279`,
  `src/neural_forge/hf_lfm2_cpcf_short_conv_v5.py:1-15`, `:1577-1621`).  The
  attention-to-conv solve below is a retargeting of that machinery, not a new
  solver family.
- **The Volterra/CP diagnosis.**  Each parent short-conv channel contributes
  one lagged bilinear rank-one term
  `C_s(current) * K_s(lag) * B_s(lagged) * X_s(lagged) * output_direction_s`;
  channelwise averaging does not commute with the composed operator, and the
  correct object to fit is the composed input-to-residual operator
  (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:186-218`).  This
  design treats *both* translation directions as fits of that composed
  operator.

## Notation

All per-layer fits operate on the operator-norm input rows (RMSNorm output),
written `n_t in R^H`, and target the branch residual write `Delta_t in R^H`.
Row-vector convention.  `L` is the FIR length.  `P^h_{t,s}` is the cached
softmax probability of head `h`, query position `t`, key position `s`;
`V^h_s = W_v^h n_s` the cached per-head value rows.

ShortConv branch, channelwise:

```text
B_t = n_t W_B^T + b_B      C_t = n_t W_C^T + b_C      x_t = n_t W_x^T + b_x
r_t = B_t (*) x_t
v_t = b_conv + sum_{l=0}^{L-1} k_l (*) r_{t-l}
Delta_t = (C_t (*) v_t) W_out^T + b_out
```

`(*)` is elementwise.  Both gates make the branch *bilinear* in `n`: `r` is
quadratic in `n` (B times x), and `C (*) v` multiplies two `n`-dependent
quantities again.  Attention is likewise bilinear-through-softmax in `n`
(scores quadratic, values linear).  The translations below are exact on the
parts of the operators whose bilinear structure matches, and measured plus
rerouted on the parts that provably cannot match.

## Direction A: attention -> ShortConv (the hard direction)

### A.1 Exact lag decomposition of the cached attention branch

From cached probabilities and values, split the parent attention delta exactly
into a position-local band and a far remainder, with no parent forward:

```text
Delta_t = Delta_band_t + Delta_far_t
Delta_band_t = sum_h sum_{l=0}^{L-1} P^h_{t,t-l} V^h_{t-l} W_o[h]^T
Delta_far_t  = sum_h sum_{s <= t-L}  P^h_{t,s}  V^h_s    W_o[h]^T
```

Both terms are computable row-exactly from `attention.probabilities`,
`attention.v` / `attention.value_history`, and the parent `W_o`
(`hf_lfm2_circuit_trace_cache.py:3034-3039`; for family-neutral evidence, by
replaying `causal_attention_probabilities` from the sealed Q/K/V projection
outputs, deterministic given rows, masks and positions —
`hf_causal_gqa_attention_fold.py:302-338`).

Diagnostics defined here and used everywhere below:

```text
band_mass_t = (1/H_heads) sum_h sum_{l<L} P^h_{t,t-l}          (per row)
rho_band    = ||Delta_band||_F^2 / ||Delta||_F^2               (fit rows)
rho_far     = ||Delta_far||_F^2  / ||Delta||_F^2
```

### A.2 What the ShortConv class can represent

For channel `c` the conv contributes `sum_l k_{l,c} C_{t,c} B_{t-l,c}
x_{t-l,c}` into `W_out` column `c`.  Setting the x-stream to carry a value
image of the residual (`x = n W_x^T`), the conv realizes a **lag-separable,
per-channel rank-one factorization of the local attention kernel**:

```text
attention (in band):  weight(t, t-l)   = P^h_{t,t-l}     (joint in q_t, k_{t-l})
conv:                 weight_c(t, t-l) = k_{l,c} * C_{t,c} * B_{t-l,c}
```

i.e. content dependence must factor as (function of the query row) times
(function of the key row), per channel, with a fixed lag profile.  This is a
CP/rank-one approximation of `exp(q_t . k_s)` restricted to the band — the
same structured-operator view the 100x failure record derived for the conv
fold (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:205-218`).

### A.3 What is provably lost, and where it must go

1. **All mass beyond lag L-1** (`Delta_far`).  A depthwise FIR is
   position-local by construction; no gate recovers positions outside the
   window.  Not fittable, and must not be fitted.
2. **Non-separable in-band content coupling** — the residual of the best
   rank-one-per-channel factorization of `P^h_{t,t-l}`.  Bounded and measured,
   never pretended away.
3. **Softmax normalization coupling.**  Attention band weights co-vary with
   off-band score mass (the softmax denominator).  The conv has no
   normalization; fitted `k`/`B`/`C` absorb only the average denominator.
4. **Additive constants** when the child conv is materialized without biases
   (`conv_bias` metadata, `hf_lfm2_canonical.py:817-819`): constants must ride
   the residual stream and be absorbed by the next fitted writer, per the
   affine-lift intercept-transport rule
   (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:256-258`).

**Remainder routing (mandatory).**  Child layers compile front-to-back.  For a
converted layer `i`, after its conv fit is frozen, compute on sealed rows

```text
R_i,t = Delta_i,t - DeltaHat_i,t     (replayed fitted conv, FP64 then BF16)
```

which contains `Delta_far` plus the in-band fit residual.  `R_i` is added to
the *endpoint target* of the nearest downstream child layer `j` that can
represent global mixing (the next surviving attention layer):

```text
Y_j = Delta_j + sum_{i converted, carrier(i)=j} R_i
```

This preserves the cumulative residual sum at every surviving-attention
checkpoint boundary and at the final-norm input — the quantities the
downstream stack actually consumes.  It is the forward analytic sweep of the
active mechanism (`docs/CURRENT_FOLDING_GOAL.md:152-158`) applied at KIND
granularity.  Two honesty rules:

- The carrier computes its addition from *its own* residual state, which now
  differs from the parent state; carrier fits therefore use sequentially
  recaptured child inputs, not parent-coordinate inputs (the A0025 lesson:
  small local errors, endpoint error up to 14.37 when components were fitted
  in incompatible evolving coordinates —
  `reports/92b2e091-a0025-live-prefix/CAUSAL_GAP_AUDIT.md:26-34`,
  `docs/CURRENT_FOLDING_GOAL.md:78-81`).
- Routing is judged at the checkpoint: acceptance compares checkpoint
  cumulative-residual validation error of (carrier fitted with remainder)
  against (carrier fitted without remainder).  If routing makes the checkpoint
  worse, conversion of layer `i` fails closed; the remainder is not silently
  dropped.

### A.4 The solve

Variables (child conv layer, external width `H`, internal channel width `D`,
FIR length `L`): `W_B, W_C, W_x in R^{D x H}`, `k in R^{D x L}`,
`W_out in R^{H x D}` (or the factorized writer,
`hf_lfm2_canonical.py:121-123`), optional biases.

Evidence rows: sealed operator-norm inputs `n_t` (child-side, sequentially
recaptured), target `Y_t = Delta_band_t` (parent residual coordinates; under
the companion cross-width design, the chart-transformed target).  Partitions:
the frozen `selection` / `fit` / `validation` splits, all three admitted,
selection used only for policy choice
(`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:236-239`).

Stages (all closed form; this is the v4/v5 recipe retargeted):

1. **Initialization from cached attention structure.**
   - `k_l(0) = mean_{t,h} P^h_{t,t-l}`, broadcast over channels (mean band
     lag profile).
   - `W_x(0)`: signed channel directions maximizing covariance between the
     lagged input stream and `Y`, via the deterministic PLS /
     balanced-circuit basis mandated for v4
     (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:244-249`;
     `src/neural_forge/dynamic_balanced_circuit_basis.py:1-22`).  Positive
     averaging is banned (failure record item 8, `:196-199`).
   - `B(0) = 1`, `C(0) = 1` (i.e. `W_B, W_C = 0` with unit bias when biases
     exist; otherwise initialized by the first alternation pass).
2. **Bounded deterministic variable projection** (fixed count, e.g. 3 rounds;
   each sub-step is an exact weighted ridge given the others, so the fit
   objective is monotone nonincreasing on fit rows):
   - `(W_out | rest)`: ridge from replayed features `C (*) v` to `Y`.
   - `(k | rest)`: per-channel lag ridge on the composed endpoint using the
     exact per-channel normal-equation accumulation — never expand
     per-channel Jacobians; use the sufficient-statistic form
     `G = (||w||^2/O)(J^T J / n)`, `q = J^T (R w)/(n O)`
     (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:259-265`).
   - `(W_B | rest)`: given `x`, `r_{t-l} = B_{t-l} (*) x_{t-l}` is linear in
     `B`; ridge from `n` rows.
   - `(W_C | rest)`: given `v`, `Y_t ~ (C_t (*) v_t) W_out^T` is linear in
     `C`; ridge from `n` rows.
   - `(W_x | rest)`: symmetric to `W_B`.
3. **Regularization and conditioning.**  Scale-normalized ridge
   `lambda = alpha * trace(X^T X)/d` with an unpenalized intercept
   (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:241-247`),
   condition caps from the deployed candidate set `(1e3, 1e4)`
   (`src/neural_forge/hf_lfm2_cpcf_short_conv_v5.py:35`), and the equilibrated
   ridge system (`hf_lfm2_cpcf_short_conv_v5.py:1052`).  Expected
   conditioning: `n_t` rows are RMS-normalized, so Gram condition is set by
   residual-stream anisotropy, not scale; the worst-conditioned alternation
   step is `(W_C | rest)` when `v` variance collapses — detect via the
   condition cap and fall back to frozen `C = C(0)`.  `alpha` and the round
   count are chosen on the *selection* partition only.
4. **BF16 replay before any gate is scored**: materialize every candidate
   tensor in the target dtype and replay B/C/x, FIR, and writer intermediates
   in BF16; casting only the final FP64 prediction is not a target-dtype
   evaluation (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:267-270`).

### A.5 Evidence requirements (Direction A)

Per converted parent attention layer: sealed operator-norm input rows;
per-head probabilities and value rows (or Q/K/V projection outputs sufficient
to replay them deterministically); the branch residual delta; masks and
position ids; the three frozen partitions; the parent `W_o`.  All are present
in the two existing capture formats cited above.  No new capture kind is
required for same-width folding; cross-width additionally requires the chart
set of `docs/CROSS_WIDTH_RESIDUAL_CROSSING_DESIGN_2026-08-01.md`.

## Direction B: ShortConv -> attention (the easier direction)

### B.1 Exact statement

A kernel-L depthwise FIR **with frozen gates** is exactly a banded linear
attention.  Freeze `B_t = Bbar`, `C_t = Cbar` (evidence means over sealed
`short_conv.b` / `short_conv.c` rows,
`hf_lfm2_circuit_trace_cache.py:3098-3100`).  Then

```text
Delta_t = sum_{l=0}^{L-1} n_{t-l} M_l^T + const
M_l = W_out diag(Cbar (*) k_l (*) Bbar) W_x        (exact, closed form)
```

an attention operator with hard banded weights `A(t, t-l) = [l < L]` and
per-lag value maps `M_l`.  This identity is exact whenever the gates are
constant on the evidence support; the gate-collapse test is `Var[B]`,
`Var[C]` on the cached rows against an explicit policy threshold.

### B.2 First-order gate linearization (general case)

When gates are not constant, expand bilinearly around evidence means
(`xbar = E[x]`, `vbar = E[v]`):

```text
C_t (*) v_t = Cbar (*) v_t + (C_t - Cbar) (*) vbar + eps_C,t
B_s (*) x_s = Bbar (*) x_s + (B_s - Bbar) (*) xbar + eps_B,s
```

giving three linear-in-`n` streams the attention class *can* represent:

```text
lag-l value map:  diag(Cbar (*) k_l)(diag(Bbar) W_x + diag(xbar) W_B)  (key at t-l)
self gate map:    diag(vbar) W_C                                       (key at t)
second order:     eps_t = W_out [Cbar (*) (sum_l k_l (*) eps_B,t-l) + eps_C,t]
```

`eps` is the covariance remainder — **provably not representable** by
linear-value attention (attention values are linear in the key row; `eps` is
quadratic).  Measure `rho_2 = ||eps||^2 / ||Delta||^2` on fit rows; route
`eps` downstream exactly as in A.3.  Because `eps_t` depends only on positions
`t-L+1..t`, the routing carrier may be the same layer's MLP branch fit (its
endpoint target simply includes `eps`); the carrier choice is scored on the
selection partition.

### B.3 Compilation into the softmax GQA/RoPE class — reuse the existing kernel

Do not write a new attention compiler.  `compile_hf_causal_gqa_attention_fold`
already accepts *desired* source Q/K/V sequences plus a residual endpoint and
performs preimage, ridge, probability replay, and endpoint-coupled V/O
(`hf_causal_gqa_attention_fold.py:470-722`).  The conv-to-attention operator
therefore **synthesizes desired attention evidence** and feeds the unchanged
kernel:

1. **Desired probabilities.**  Assign one target head per lag `l < L`
   (requires target `query_heads >= L`; with `L = 3` and typical child heads
   `>= 4` this holds; fail closed otherwise).  Desired band profile for head
   `l`: probability 1 at key position `t-l`, 0 elsewhere.  Signs and
   magnitudes of `M_l` live entirely in the V/O solve, so probability targets
   are nonnegative by construction.
2. **Desired pre-RoPE Q/K.**  Construct relative-position scores from the
   RoPE frequency ladder: with constant pre-RoPE vectors `q0, k0` supported on
   `m` rotation planes with amplitudes `a_j`, the post-RoPE score is a cosine
   series `g(t-s) = sum_j a_j cos(omega_j (t-s) - phi_j)`.  Solve the small
   deterministic LS problem for `a_j, phi_j` such that `g(l) = gamma` and
   `g(m) <= 0` for `m != l` on the window `[0, W]` (`W` = max sealed sequence
   length; `head_dim/2` frequencies available, e.g. 32 for `d = 64`).  Emit
   desired q/k *sequences* `q_t = q0`, `k_s = k0` (constant), inverse-rotated
   with `invert_rope` (`hf_causal_gqa_attention_fold.py:292-294`).  The kernel
   then ridge-fits `W_q, W_k` from actual child rows (`:605-613`); child rows
   are not constant, so the fit is approximate, and that approximation is
   *scored by the kernel itself*, which replays the actual installed
   probabilities before the V/O solve (`:634-657`).
3. **Desired values.**  Per lag head `l`, the linearized value stream of B.2:
   `V(l)_s = n_s (diag(Bbar (*) Cbar (*) k_l) W_x + diag(xbar (*) Cbar (*) k_l) W_B)^T`,
   plus one self head carrying `n_t (diag(vbar) W_C)^T`.
4. **Endpoint.**  `target_residual_delta_sequences = Delta_t` (the cached
   conv branch delta, `residual.post_operator - residual.pre`,
   `hf_lfm2_circuit_trace_cache.py:2811-2814`), so the coupled V/O ridge
   recovers first-order in-band misweighting automatically.

**When this is exact.**  (i) gates constant on support (B.1), and (ii) softmax
saturation: finite scores never give exactly one-hot probabilities, so banded
weights carry leak `eps_leak(gamma) <= t * exp(-gamma_margin)`; `gamma` is
capped by Q/K fit fidelity on non-constant rows.  The construction is exact in
the linear-attention operator sense and exact-in-the-limit for the compiled
softmax head; the residual of both approximations is measured at the endpoint
by the kernel's own fit/validation diagnostics
(`hf_causal_gqa_attention_fold.py:685-702`).

## Acceptance diagnostics (both directions)

Recorded in every sealed report; enforced thresholds are policy constants
chosen on the selection partition, never on validation:

- `rho_band`, `rho_far`, the per-row `band_mass` distribution (A);
  `Var[B]`, `Var[C]`, `rho_2` (B); rank-one band-factorization residual (A.2).
- Endpoint fit and validation relative RMSE and retained energy versus three
  explicit baselines: **zero**, **identity/no-branch** (the baseline v3
  omitted, `research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:128-133`),
  and **frozen-gate / linear-FIR-only** (the exact-part-only candidate).
- Checkpoint cumulative-residual validation error at every carrier boundary,
  routing-on versus routing-off.
- FP64 versus BF16 endpoint delta after full-intermediate BF16 replay.
- Condition numbers, effective ranks, coefficient norms, signed-cancellation
  measures, per-case metrics — the full v4 diagnostic receipt
  (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:229-235`), sealed
  even on rejection.

## Fail-closed conditions

Each aborts the layer conversion before any child mutation (the snapshot
transaction pattern of
`src/neural_forge/dynamic_causal_operator_promotion.py:298-362` applies
unchanged):

1. Direction A with `rho_far` above threshold and **no downstream carrier**
   (no surviving child attention layer after the converted layer).  In
   particular: an all-conv child of an all-attention parent fails closed at
   the first layer whose far mass exceeds the bound.  This is a truth about
   the operator classes, not a solver defect.
2. Checkpoint validation error with routing worse than without (rule A.3).
3. Direction B with target `query_heads < L`, non-default RoPE
   (`_require_rope_transport`,
   `dynamic_causal_operator_promotion.py:199-217`), or Q/K probability replay
   failing to place at least the policy fraction of mass on the assigned lags
   on validation rows.
4. Validation endpoint relative RMSE at or above the no-branch baseline after
   BF16 replay (the v3 lesson: fit passed while validation magnitude blew up,
   `research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:112-133`).
5. Any partition overlap, unauthenticated evidence, non-finite tensor,
   condition-cap exhaustion across all ridge candidates, or gate-collapse /
   linearization threshold violated with no routing carrier.

## Persistence

No database is involved.  All artifacts follow the repo convention: sealed
canonical-JSON reports with `report_sha256` self-seals and content-addressed
tensor fingerprints (as in `hf_causal_gqa_attention_fold.py:93-110`,
`hf_lfm2_cpcf_short_conv_v5.py:94-113`).  New schema identifiers:

```text
neural-forge-causal-attention-to-short-conv-fold-v1   (per-layer fold report)
neural-forge-causal-short-conv-to-attention-fold-v1   (per-layer fold report)
neural-forge-causal-kind-translation-remainder-v1     (sealed routed-remainder
    bundle: source layer, carrier layer, rho_band/rho_far/rho_2, remainder
    tensor sha256, checkpoint boundary id, routing-on/off checkpoint metrics)
```

The remainder bundle is a first-class sealed artifact because the carrier's
fit consumes it; its lineage binds the converted layer's report sha, the
evidence cache key, and the carrier action id.

## Planner and executor integration

- Two new method ids beside the existing pair
  (`dynamic_fold_planner.py:30-31`):
  `causal_attention_to_short_conv_layer` and
  `causal_short_conv_to_attention_layer`.
- `_family_neutral_causal_layer_contract` gains a KIND-crossing arm: when the
  source operator branch is ATTENTION and the target branch is SHORT_CONV (or
  vice versa) and every other contract clause holds, emit the translation
  method instead of returning `None` (`dynamic_fold_planner.py:746-807`); the
  LFM2 layer contract already tags SHORT_CONV nodes
  (`dynamic_fold_planner.py:814-831`).
- Two new `FoldOperator` classes in the promotion module following
  `_CausalBranchOperator` exactly (bind, compile, snapshot, install,
  transaction; `dynamic_causal_operator_promotion.py:365-602`), registered in
  `dynamic_fold_registry` like the existing branch operators.
- Direction B's compile step calls `compile_hf_causal_gqa_attention_fold`
  unchanged; Direction A's compile step calls a new
  `compile_hf_causal_attention_to_short_conv_fold` built on the v5 ridge/ALS
  primitives.

## Staged proof plan

1. **Toy exactness (unit tests, synthetic tensors).**
   a. Constant-gate conv to attention: assert endpoint error tracks the
      derived `eps_leak(gamma)` bound; signed FIR taps; heterogeneous
      channels.
   b. Banded synthetic attention (all mass in lags `0..L-1`, separable
      scores) to conv: assert near-zero validation endpoint error; then
      adversarial non-separable band and off-band mass: assert the measured
      `rho` values and that the fail-closed paths fire.
2. **Single-layer real LFM2, same width.**  Convert one real LFM2.5-230M conv
   layer into an attention branch inside an otherwise kind-preserving child;
   and one real attention layer into a conv branch with remainder routed to
   the next real attention layer.  Sealed cache only; report every acceptance
   diagnostic; no whole-model claim.
3. **Whole-model kind-swap control.**  LFM2 to LFM2-shaped child with exactly
   one layer kind-swapped, sequential recapture, full manual review against
   the kind-preserving control child.
4. **Cross-family.**  Qwen3-style all-attention parent (via
   `hf_causal_circuit_evidence`) to an LFM2-shaped child at *equal* hidden
   width: surviving attention layers use the existing branch operator; the
   rest use Direction A with routing.  Cross-width variants depend on
   `docs/CROSS_WIDTH_RESIDUAL_CROSSING_DESIGN_2026-08-01.md` and are sequenced
   after it.

Every stage publishes sealed receipts, keeps validation untouched until its
operator is frozen, and honors the ban on re-generating sealed evidence
(`docs/CURRENT_FOLDING_GOAL.md:120-127`); on-policy extension rounds may
supply additional sealed rows under the existing contract
(`docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md`, consumption rule).

## Trade-offs

- **Reuse the coupled GQA kernel for conv-to-attention (chosen)** versus a
  bespoke banded-attention compiler (rejected): the kernel already implements
  preimage, ridge, probability replay, and endpoint-coupled V/O with sealed
  reports; a bespoke compiler would duplicate roughly 700 audited lines for a
  marginally tighter band construction.
- **Endpoint remainder routing (chosen)** versus pretending the conv can
  absorb far mass with larger gates (rejected): the far term is outside the
  operator class; fitting it is exactly the noise-fitting failure mode v3
  died on.
- **First-order gate linearization with measured `rho_2` (chosen)** versus
  exact bilinear matching via architecture changes (rejected): squared-value
  or gated-attention children would represent `eps` exactly but are not
  materializable in the current blank targets and would fork the child
  runtime.
- **Bounded deterministic ALS (chosen)** versus a joint global CP solve
  (rejected): no closed form exists for the joint problem; bounded variable
  projection is the already-specified v4 policy and keeps cold/resumed byte
  identity achievable.
- **Fixed per-lag head assignment (chosen)** versus learned soft head
  allocation (rejected): deterministic, auditable, and sufficient because
  magnitudes live in the V/O ridge.

## Risks

- **Content selection is the point of attention.**  A layer can show high
  `rho_band` yet owe its usefulness to *which* in-band token it picks; the
  separable factorization can smooth that away while every RMSE metric looks
  acceptable.  Manual whole-response review remains the verdict
  (`docs/CURRENT_FOLDING_GOAL.md:22-27`).  This is the key risk of this
  design.
- Carrier layers may be unable to predict routed remainders from their own
  states; the checkpoint gate catches this, but the result is then a
  fail-closed conversion — reduced topology freedom, not silent quality loss.
- Direction B's Q/K constant-vector fit rides on the residual stream's DC
  component; models without a strong sink direction will leak band mass.  The
  kernel's replayed-probability diagnostics expose this per layer.
- Compute: replaying cached probabilities and band splits over the ~90 GB
  sealed caches is IO-heavy; stage 2+ runs belong on the AI server, scheduled
  like existing CPCF runs.  No production service, credential, or parent
  snapshot is modified by any stage; publication remains transactional.

## What this does NOT claim

- It does not claim attention-to-conv translation is lossless.  `Delta_far`
  and the non-separable band residual are provably outside the ShortConv
  class; the design measures them and routes them, and fails closed when
  routing has no carrier.
- It does not claim conv-to-attention is exact for live gates; exactness
  holds only under gate collapse and softmax saturation, both measured.
- It does not claim any whole-model intelligence result.  No child produced
  by this design is accepted on metrics alone; the manual-review contract of
  `docs/CURRENT_FOLDING_GOAL.md` governs.
- It does not relax any fold-only rule: no gradients, no optimizer, no parent
  runtime sidecar, no unsealed parent resampling, no weakened validation
  gate.
- It does not cover cross-width crossings; those belong to
  `docs/CROSS_WIDTH_RESIDUAL_CROSSING_DESIGN_2026-08-01.md`.
