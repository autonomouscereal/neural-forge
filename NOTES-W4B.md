# NOTES-W4B — attention <-> ShortConv operator-KIND translation, stage 1

Author: wave W4B (kind-translation kernel, toy scale, both directions)
Date: 2026-07-31
Branch: `codex/lrc-blank-compression` (HEAD at start: `c69d5a64`)
Design: `docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md`

Files owned and changed by this wave:

- `src/neural_forge/hf_causal_kind_translation.py` (new, 2863 lines)
- `tests/test_hf_causal_kind_translation.py` (new, 907 lines, 26 tests)
- `NOTES-W4B.md` (this file)

Deliberately untouched: `hf_causal_gqa_attention_fold.py` (consumed through its
public `compile_hf_causal_gqa_attention_fold` / `HFCausalGQATopology` /
`causal_attention_probabilities` / `apply_rope` API only),
`hf_causal_circuit_evidence.py`, `hf_lfm2_canonical.py` (consumed:
`Lfm2ShortConvParameters`, `lfm2_short_conv_reference`),
`dynamic_causal_operator_promotion.py`, `dynamic_fold_planner.py`,
`dynamic_fold_registry.py`. `git status` shows this wave adding exactly two
untracked files; the `M` entries in `src/` at the time of writing are a
concurrent wave's, not this one's.

This is **stage 1 of the design's proof plan only** — toy exactness on synthetic
tensors, both directions. No real LFM2 layer, no whole model, no planner or
executor wiring, no claim about intelligence.

---

## 1. What the module implements

### 1.1 ShortConv semantics (shared)

`ShortConvBranchParameters` holds one branch in the exact LFM2 physical layout
(`in_proj [3D,H]`, `conv [D,1,L]`, `out_proj [H,D]`, optional biases) and
evaluates the contract-pinned semantics from `hf_lfm2_canonical.py:100-105`:

```text
(B_t, C_t, x_t) = chunk_3(in_proj(h_t))    r_t = B_t * x_t
v_t = conv_bias + sum_lag k_lag * r_(t-lag)    delta_t = out_proj(C_t * v_t)
k[:, lag] = conv.weight[:, 0, L-1-lag]
```

`forward` is **bit-identical** to `lfm2_short_conv_reference` for square
(`D == H`) branches with and without biases — `torch.equal`, not `allclose`
(`tests/test_hf_causal_kind_translation.py::test_short_conv_forward_reproduces_the_canonical_reference`).
Heterogeneous `D != H` is supported by the same code path and exercised.

### 1.2 Direction B: the banded-linear-attention identity

`linearize_short_conv_branch` expands the branch bilinearly around evidence
means over the fit rows. The exact algebra implemented (derived, not copied,
because the design states only the collapsed form):

```text
r_lin_s = n_s A^T + a          A = diag(Bbar) W_x + diag(xbar) W_B
                               a = Bbar*b_x + b_B*xbar - Bbar*xbar
Delta_t = sum_{l <= min(L-1,t)} n_{t-l} M_l^T + n_t M_self^T + c_t + eps_t
M_l     = W_out diag(Cbar * k_l) A
M_self  = W_out diag(vbar) W_C
c_t     = (Cbar*(b_conv + sum_{l<=min(L-1,t)} k_l*a) - Cbar*vbar + b_C*vbar) W_out^T + b_out
eps_t   = (Cbar * sum_l k_l*eps_B,t-l + eps_C,t) W_out^T
eps_B,s = (B_s-Bbar)*(x_s-xbar)      eps_C,t = (C_t-Cbar)*(v_t-vbar)
```

`eps_t` matches the design's B.2 remainder exactly. Two things the design's
formulas leave implicit and this implementation gets right:

* **Boundary truncation is preserved.** The lag sum runs over `l <= t`, so the
  identity is exact at the start of a sequence too, and `c_t` is position
  dependent for the first `L-1` rows only (reported as
  `boundary_constant_rows`).
* **Under gate collapse the identity is exact including the constant**:
  `M_self == 0` identically and `eps == 0`.

Measured: `max|banded - branch| = 1.9e-15` against a branch whose delta has
`max|Delta| = 5.53`, `rho_2 = 4.9e-32`.

### 1.3 Direction B: compilation into the softmax GQA/RoPE class

`compile_short_conv_to_attention_translation` **synthesizes evidence** for the
existing kernel and calls it unchanged:

* **Streams.** `L` lag streams plus one self-gate stream (included when
  `||M_self||_F` clears a relative floor, or forced on/off by policy).
* **Head assignment.** Streams are assigned to *key/value* head groups by a
  balanced partition; every query head of a group inherits that group's lag.
* **Desired probabilities** are produced implicitly: each stream gets constant
  pre-RoPE `q0`/`k0` whose post-RoPE score is the cosine series
  `g(d) = sum_j A_j cos(w_j d) + B_j sin(w_j d)` with `(A_j, B_j) = (k0_j,
  k0_{j+half})` and `q0 = (1,...,1,0,...,0)`. The kernel then RoPEs them,
  softmaxes, and uses the result as its own desired probability target.
  The series identity is verified numerically inside the builder against
  `apply_rope`, and a deviation above `1e-6` relative fails closed.
* **Desired values** are `n_s G_s^T` where `M_s = O_s G_s` is a
  sign-canonicalized truncated SVD at the stream's available rank
  (`|K_s| * head_dim`), split `sqrt(sigma)` between the two factors for
  conditioning. `O_s` is *not* installed — the kernel's own endpoint-coupled
  ridge recovers the writer — but the truncation error is sealed.
* **Endpoint** is the cached branch delta, so the coupled V/O ridge absorbs
  first-order in-band misweighting, exactly as B.3 step 4 requires.
* **BF16** replay of every intermediate (projection, head norm, RoPE, softmax,
  context, writer) with all tensors materialized in BF16, then compared to FP64.

### 1.4 Direction A: exact lag split

`attention_band_decomposition` is row-exact from cached probabilities, cached
value rows and the parent `W_o`, with no parent forward:

```text
Delta_band_t = sum_h sum_{l<L} P^h_{t,t-l} V^h_{t-l} W_o[h]^T
Delta_far_t  = sum_h sum_{s<=t-L} P^h_{t,s} V^h_s W_o[h]^T
band_mass_t  = (1/H) sum_h sum_{l<L} P^h_{t,t-l}
rho_band = ||Delta_band||^2/||Delta||^2   rho_far = ||Delta_far||^2/||Delta||^2
```

Reconstruction is checked against the sealed delta (`1.7e-16` relative on the
toy) and a mismatch above `band_reconstruction_tolerance` fails closed.
`attention_evidence_from_projections` replays probabilities from sealed Q/K/V
projection outputs for family-neutral parents, using
`causal_attention_probabilities` from the read-only kernel.

Note `rho_band + rho_far != 1` in general: the two components are not
orthogonal. Both are reported as energy ratios against the same denominator,
which is what the design defines.

### 1.5 Direction A: the solve

Bounded deterministic variable projection against the composed endpoint
`Y = Delta_band`, in the design's sub-step order — writer, lag kernel, `W_B`,
`W_C`, `W_x` — with every sub-step an exact condition-capped ridge given the
others, and per-channel Gauss-Seidel with rank-one residual refresh for the four
gated sub-steps.

The `(k | rest)` and gate steps use the mandated sufficient-statistic form
(`G = ||w||^2 (J^T J)/n`, `q = J^T(R w)/n`); no per-channel Jacobian is ever
expanded. Per-channel designs are:

| step | design columns | coefficient |
|---|---|---|
| `k` | `[C_{t,c} r_{t-l,c}]_l` + `[C_{t,c}]` | `k_{.,c}`, `conv_bias_c` |
| `W_B` | `sum_l C_{t,c} k_{l,c} x_{t-l,c} n_{t-l}` + scalar | `W_B[c]`, `b_B,c` |
| `W_x` | `sum_l C_{t,c} k_{l,c} B_{t-l,c} n_{t-l}` + scalar | `W_x[c]`, `b_x,c` |
| `W_C` | `v_{t,c} n_t` + `v_{t,c}` | `W_C[c]`, `b_C,c` |

Regularization follows the v5 recipe exactly: column equilibration, eigen-solve,
ridge `= max(alpha*trace(G)/d, (lmax - kappa*lmin)/(kappa-1))` over the deployed
cap set `(1e3, 1e4)`, identifiability thresholding, unpenalized intercept for
the writer.

The policy ladder — initialization x rounds x alpha x condition cap, 24
candidates by default — is scored **only on the selection partition**;
validation is never read until the operator is frozen.

---

## 2. Deviations from the design, and why

The module docstring carries nine of these verbatim; they are repeated here with
the numbers that motivated them.

1. **`key_value_heads >= stream count`, not `query_heads >= L`** (design B.3
   step 1). A GQA group shares one key sequence, so every query head in the
   group sees the same relative-position profile and therefore the same lag.
   Assigning lags per *query* head would silently collapse distinct lags. Fails
   closed with an explicit message.

2. **Per-stream value rank is bounded by the head geometry** — a stream owning
   `m` KV heads carries a rank-`m*head_dim` image of its `H x H` map. The design
   does not mention this. Total representable rank across all streams is exactly
   `key_value_heads * head_dim`. Sealed as
   `stream_rank_truncation_relative_error` per stream (`<= 9.6e-16` on the toy,
   where `H = 8 <= head_dim = 8`).

3. **Desired Q/K are emitted pre-RoPE.** The design says to inverse-rotate with
   `invert_rope` first; the kernel RoPEs the source Q/K it is given and then
   inverse-rotates with the target theta (`:621-666`), so with a synthesized
   source topology equal to the target the constants round-trip exactly with one
   fewer rotation.

4. **The additive constant is not representable** by the bias-free attention
   class and can only ride the child rows' DC through the fitted `W_o`. Measured
   as `linearized_constant_relative_energy` (`0.0` on the toy branch, whose
   constant cancels).

5. **The v5 ridge frontier is reproduced, not imported.**
   `condition_capped_equilibrated_prior_residual_ridge_frontier`
   (`hf_lfm2_cpcf_short_conv_v5.py:1577`) requires authenticated
   `Lfm2CpcfV5RidgeCase` bundles and a *prior-residual* parameterization keyed to
   a transported source operator of the same kind; a kind translation has no such
   operator. The numerical policy is identical; the case-authentication and
   trust-radius layers are not reproduced.

6. **Monotonicity is guarded, not assumed.** Design A.4 stage 2 claims the fit
   objective is monotone nonincreasing because each sub-step is an exact ridge.
   That holds for the *regularized* objective only. Each sub-step is therefore
   reverted if the unregularized fit objective rose, so
   `fit_objective_monotone_nonincreasing` is a fact about the reported trace.
   On the toy no sub-step was ever reverted (`reverted_substeps == 0`).

7. **Gate bias columns are penalized.** Only `b_out` is a plain intercept of its
   design and is fitted unpenalized by centering; `b_B`, `b_C`, `b_x`,
   `conv_bias` enter multiplied by data-dependent factors.

8. **Direction A's endpoint gate is scored against the band target.** Gating on
   the full delta would fail exactly when `rho_far` is large and correctly
   routed. The full-delta number is reported beside it.

9. **Direction B has an unrepresentable FIR prefix.** A zero-padded FIR writes
   nothing for a lag that does not exist yet, but every causal softmax row sums
   to one, so the lag-`l` head at `t < l` unavoidably writes some other key's
   value. Positions `0..L-2` are outside the operator class in exactly the sense
   `Delta_far` is in direction A, and are dropped from the *endpoint solve*
   under the design's own rule (A.3 item 1: "not fittable, and must not be
   fitted"), counted, scored separately, and carried by the routed remainder.
   Validation is never narrowed. **This is not cosmetic**: measured on the same
   toy by leaving the prefix rows in the solve, the collapsed-gate validation
   endpoint error goes from `0.0239` to `0.2488` and the live-gate one from
   `0.0417` to `0.2577` — a 10x and 6x degradation, because the writer ridge
   spends its budget on rows no setting of the weights can serve.

Three further additions the design does not specify:

10. **The RoPE band selector's ridge is chosen from a frozen ladder by
    maximizing the perturbation-normalized margin** `c = margin / (||q0|| ||k0||)`.
    The design says "solve the small deterministic LS problem"; with a geometric
    frequency ladder over a short window an unregularized delta target is
    catastrophically ill-posed — `||k0|| ~ 1e4-1e5` and `c ~ 8e-4`, which makes
    the compiled head's band completely destroyed by any realistic Q/K fit error.
    The score perturbation from a projection error `d` is bounded by
    `scaling*||d||*||k0||` while the margin scales as `scaling*||q0||*||k0||*c`,
    so `c` is the scale-invariant quantity that decides whether the band
    survives. Maximizing it over `alpha in (1e-4 .. 3.0)` raises `c` to
    `0.14-0.20` and `||k0||` to `~40`. The criterion depends only on `head_dim`,
    `rope_theta`, `window` and `lag` — **no evidence row of any partition enters
    it** — so it is not a selection leak.

11. **Two extra Direction-A initializations, scored on selection.** The design
    mandates `k_l(0) = mean_{t,h} P^h_{t,t-l}` broadcast over channels plus PLS
    x-directions. That init makes every channel see the same lag mixture, so
    alternation must break a symmetry before it can express the band's CP
    structure. Added:
    * `lag_blocked` — contiguous channel blocks assigned to single lags, block
      sizes from the measured lag profile by largest remainder;
    * `value_transport` — the failure record's own item 6 ("transport actual
      parent input and output projections and use those values as priors"):
      channel blocks are `(query head, lag)` cells sized by their mean in-band
      probability, each carrying that head's value directions recovered from the
      sealed value rows by one intercept-free ridge (recovery relative RMSE
      `1.6e-6` on the toy, itself a sealed diagnostic).
    Selection picks per case, and it picks differently per case: the clean band
    chose `value_transport`, the non-separable band chose `lag_blocked`, the
    far-mass band chose `mean_lag_profile`.

12. **A lag-separable linear floor diagnostic.** The best *unconstrained*
    lag-separable linear operator on the same rows (a ridge from
    `[n_t, ..., n_{t-L+1}]` to `Delta_band`) is fitted and reported. It separates
    the two losses the design names: anything the conv cannot reach beyond the
    floor is the rank-one-per-channel factorization loss (A.2); anything the
    floor itself cannot reach is the softmax normalization coupling (A.3 item 3),
    which no depthwise FIR of any width or gating can represent. On the toy the
    conv fit is within 0.6% of the floor and its validation error is *below* the
    floor's, so the reported band residual is the operator class's limit, not a
    solver failure.

---

## 3. What the read-only attention kernel could not express

The design says to reuse `compile_hf_causal_gqa_attention_fold` unchanged. It
does express per-lag one-hot probability targets, RoPE preimages, and linearized
value streams. Two things it cannot express, with the exact change each would
need:

1. **No output intercept.** `_ridge` (`hf_causal_gqa_attention_fold.py:349-357`)
   solves without an intercept and the target class has no `o_proj` bias, so the
   conv's additive constant `c_t` has no home. *Exact needed change*: an optional
   `fit_output_intercept: bool` on `compile_hf_causal_gqa_attention_fold` that
   centers the context/endpoint design and returns an output bias, plus a target
   layout that carries `o_proj.bias`. **Not requested** — blank LFM2/Qwen3
   attention has no `o_proj` bias, so the change would be unusable today.
   Honest subset implemented: measure `linearized_constant_relative_energy` and
   let the constant ride the residual stream's DC component through the fitted
   `W_o`, per the affine-lift intercept-transport rule.

2. **No selection partition.** The kernel takes only `fit_token_mask` and
   `validation_token_mask`. *Exact needed change*: an optional
   `selection_token_mask` used solely for policy choice and reported, never for
   the solve. Honest subset implemented: Direction B's only data-dependent policy
   choice (self-gate stream inclusion) is made from fit-row linearization norms;
   every other Direction-B constant is frozen in
   `ShortConvToAttentionPolicy`. The caller may still pass a selection mask; it
   is validated for disjointness and reported (`selection_rows`) but not
   consumed. Direction A owns its own solve and does use the selection partition
   for the full 24-candidate ladder.

---

## 4. Measured toy numbers

Toy geometry: `batch 8, sequence 16, hidden 8, channels 8, L = 3`; partitions
`fit 64 / selection 32 / validation 32` rows, disjoint by construction.

### Direction B (ShortConv -> attention)

Operator-norm rows are DC-dominant (`1.5 * u + 0.15 * noise`), which is the
shape a real RMSNorm output has and the regime the design's third risk names.

| quantity | value |
|---|---|
| collapsed-gate banded identity, `max abs err` | `1.9e-15` (`max abs Delta = 5.53`) |
| collapsed-gate `rho_2` | `4.9e-32` |
| collapsed-gate `M_self` | exactly `0` |
| per-stream rank truncation | `<= 9.6e-16` |
| RoPE series construction error (realized vs predicted scores) | `0.0` |
| normalized margin `c` per lag | `0.181 / 0.137 / 0.166` |
| min validation assigned-lag mass | `0.874` |
| **collapsed** validation endpoint rel. RMSE | **`0.0239`** (BF16 `0.0246`) |
| **live-gate** validation endpoint rel. RMSE | **`0.0417`** (BF16 `0.0415`) |
| baseline zero / no-branch | `1.0` |
| frozen-gate/linear-FIR baseline, collapsed | `2.2e-16` |
| frozen-gate/linear-FIR baseline, live | `8.0e-3` |
| fit rows dropped as unrepresentable prefix | `16` of `64` |

Gate-linearization remainder against gate dispersion (`||eps||` on fit rows):

| gate scale | `max(rel disp B, rel disp C)` | `\|\|eps\|\|_fit` | `rho_2` |
|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 | 4.9e-32 |
| 0.25 | 0.0251 | 0.6310 | 5.6e-05 |
| 0.50 | 0.0461 | 1.5004 | 1.5e-04 |
| 1.00 | 0.0760 | 4.0270 | 3.4e-04 |
| 2.00 | 0.1151 | 12.3964 | 5.8e-04 |

Strictly monotone in both, and zero at collapse. Note `||eps||` is not `O(s^2)`:
`eps_B = (B-Bbar)*(x-xbar)` is linear in the gate scale because `x` does not
depend on it, so the growth is `O(s)` with an `O(s^2)` correction.

Per-stream band selector (head_dim 8, window 15, peak 24 logits):

| lag | selected series alpha | margin (logits) | `\|\|k0\|\|` | leak bound |
|---|---|---|---|---|
| 0 | 1.0 | 5.315 | 41.5 | 0.0737 |
| 1 | 3.0 | 3.572 | 37.0 | 0.4217 |
| 2 | 1.0 | 4.930 | 42.1 | 0.1084 |

The leak *bound* is loose (`W exp(-margin)`); the *measured* assigned-lag mass is
`0.87-0.98` and is what the fail-closed gate uses.

### Direction A (attention -> ShortConv)

Cached attention built from `causal_attention_probabilities` on scores
`-4*(t-s-lag)^2 + coupling*<n_t,n_s>`, so the probabilities come out of the
read-only kernel rather than being hand-written. Rows are isotropic
(`noise = 1.0`) here.

| case | `rho_band` | `rho_far` | band mass | conv fit | conv val | floor fit | floor val | factorization residual | remainder (val) | init chosen |
|---|---|---|---|---|---|---|---|---|---|---|
| separable band, lags (0,1) | 1.00000 | 0.00000 | 1.000 | 0.1148 | **0.0830** | 0.1141 | 0.1095 | 0.0132 | 0.0069 | `value_transport` |
| non-separable band (coupling 0.35) | 1.00000 | 0.00000 | 1.000 | 0.1908 | 0.3004 | 0.1886 | 0.3488 | **0.0364** | 0.0902 | `lag_blocked` |
| far mass, lags (0,9) | 0.83836 | **0.17008** | 0.625 | 0.1833 | 0.1588 | 0.1866 | 0.2346 | 0.0336 | **0.2518** | `mean_lag_profile` |

Per-row band-mass deciles (the design asks for the distribution, not just the
mean — a layer can average a high band mass while a fraction of its rows reach
outside the window entirely):

* separable band, lags (0,1): `1.0` at every decile;
* far mass, lags (0,9): `0.5` up to the 70th percentile, `1.0` from the 80th —
  head 1 is out of band on 80% of rows and in band only on the prefix rows where
  lag 9 does not exist yet.

Supporting numbers: band/far reconstruction vs the sealed delta `1.7e-16`;
value-projection recovery `1.6e-6`; BF16-vs-FP64 endpoint delta `0.007-0.008`;
worst regularized condition `342-399` (cap `1e3`); zero/no-branch baseline `1.0`;
`reverted_substeps == 0`; objective trace `539.16 -> 7.11` in one round.

Reading: the separable band is fitted to the *class limit* (fit within 0.6% of
the unconstrained lag-separable floor, validation below it). The non-separable
band raises the factorization residual by 2.8x, which is the design's A.2 loss
being measured rather than smoothed away. The far case routes 25% of the delta
energy to a carrier.

### Routing

`evaluate_carrier_routing_checkpoint` compares the checkpoint cumulative residual
(`Delta_conv + Delta_carrier` in parent coordinates) with the carrier fitted on
`Delta_j` vs `Delta_j + R`. The carrier is a real attention layer recompiled by
`compile_hf_causal_gqa_attention_fold`:

| converted layer | routing off | routing on | verdict |
|---|---|---|---|
| lags (0,4), coupling 0.35 | 0.29165 | **0.28844** | routing accepted |
| lags (0,9), coupling 0 | **0.27557** | 0.28257 | **fails closed** |

The second row is the design's named risk realized on a real compile: a carrier
fitted from its own state cannot always predict another layer's far mass, and the
honest outcome is a refused conversion, not a silent loss. Both are asserted.

---

## 5. Fail-closed conditions implemented

| design condition | implementation | tested |
|---|---|---|
| A: `rho_far` above bound with no carrier | `maximum_unrouted_far_mass`, checked before any solve | yes |
| A.3: routing worse than not routing | `evaluate_carrier_routing_checkpoint(fail_closed=True)` | yes, end to end |
| B: too few heads for the streams | KV-head and query-head checks | yes |
| B: assigned-lag mass below the policy floor on validation | measured from replayed probabilities | yes |
| 4: validation endpoint at/above the no-branch baseline after BF16 replay | both directions | yes (direction B, isotropic rows) |
| 5: partition overlap / empty / outside valid | `_disjoint_partitions` | yes, both directions |
| 5: non-finite tensors | `_tensor` | yes |
| 5: condition-cap exhaustion | `_condition_capped_solve` raises | not tripped on the toy |
| 5: gate-linearization threshold with no carrier | `maximum_unrouted_quadratic_mass` | yes |
| evidence integrity: probabilities/values do not reconstruct the sealed delta | `band_reconstruction_tolerance` | yes |
| routing without a downstream carrier | `seal_kind_translation_remainder` refuses `None` and refuses `carrier <= source` | yes |

Not implemented (out of stage-1 scope): non-default RoPE refusal via
`_require_rope_transport` — that lives in the promotion module and belongs to the
operator wave; the topology contract's own RoPE validation is what runs today.

---

## 6. Sealed artifacts

Three schema ids, canonical-JSON with a `report_sha256` self-seal over the body
(the `hf_causal_gqa_attention_fold._report` pattern) and content-addressed tensor
fingerprints:

* `neural-forge-causal-short-conv-to-attention-fold-v1`
* `neural-forge-causal-attention-to-short-conv-fold-v1`
* `neural-forge-causal-kind-translation-remainder-v1` (also used for the
  routing-comparison receipt)

The fold reports carry, at minimum: the policy, both topologies, per-stream
receipts (kind, lag, head groups, rank budget, truncation error, band-selector
ladder and margin, leak bound, tensor shas), the linearization diagnostics
(`Var[B]`, `Var[C]`, dispersions, `rho_2`, constant energy, lag-map norms), the
band decomposition (`rho_band`, `rho_far`, band-mass min/mean/max), the full
solve-candidate table with its selection scores, the objective trace, condition
receipts, fit/selection/validation row counts and prefix accounting, endpoint
metrics in FP64 and BF16 against three baselines, the remainder shas and
energies, and `training_steps = optimizer_steps = backward_calls =
parent_forward_calls = 0`.

Determinism is asserted directly: two independent builds of both directions
produce identical `report_sha256`, identical kernel report shas, and
`torch.equal` weights and remainders.

---

## 7. Operator wiring — spec for the next wave (NOT implemented here)

Nothing in `dynamic_fold_planner.py`, `dynamic_causal_operator_promotion.py` or
`dynamic_fold_registry.py` was touched. The wiring the design calls for:

1. **Method ids** beside the existing pair (`dynamic_fold_planner.py:30-31`):
   `causal_attention_to_short_conv_layer`, `causal_short_conv_to_attention_layer`.
   They must be enabled *together with* the existing branch methods, for the same
   reason `CAUSAL_LAYER_BRANCH_METHODS` is a pair: a converted layer whose
   remainder has no carrier action cannot be closed.
2. **Planner contract arm.** `_family_neutral_causal_layer_contract`
   (`:746-807`) gains a KIND-crossing arm when the source branch is ATTENTION and
   the target is SHORT_CONV or vice versa and every other clause holds; the LFM2
   contract already tags SHORT_CONV nodes (`:814-831`). The arm must additionally
   refuse a plan in which a converted attention layer has **no surviving
   downstream attention layer**, because that is fail-closed condition 1 and it
   is cheaper to detect in the planner than after compiling.
3. **Two `FoldOperator` classes** following `_CausalBranchOperator` exactly
   (`dynamic_causal_operator_promotion.py:365-602`): bind, compile, snapshot,
   install, transaction. Target parameter paths are the ShortConv
   `in_proj/conv/out_proj` (direction A) and `q_proj/k_proj/v_proj/out_proj`
   (direction B). Direction B must call `_require_rope_transport` (`:199-217`)
   and the Q/K-norm classifier before compiling; direction A must not.
4. **Remainder lifecycle.** The remainder bundle is a first-class artifact whose
   lineage binds the converted layer's report sha, the evidence cache key, and
   the carrier action id — the module already emits all three fields, but the
   executor must (a) order actions front-to-back, (b) hand each carrier the
   bundles routed to it via `carrier_endpoint_target_with_remainder`, and
   (c) run `evaluate_carrier_routing_checkpoint` at the boundary and roll the
   whole transaction back when routing loses.
5. **Sequential recapture.** Carrier fits must consume child-side recaptured
   inputs, not parent-coordinate inputs (the A0025 lesson,
   `reports/92b2e091-a0025-live-prefix/CAUSAL_GAP_AUDIT.md:26-34`). The module's
   compile entry points already take `child_normalized_sequences` separately from
   `parent_normalized_sequences` for exactly this; stage 1 passes the same rows.

---

## 8. Verification

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" \
    -m pytest tests/test_hf_causal_kind_translation.py -q -p no:randomly
```

`26 passed in 23.96s`.

Regression check on the read-only kernel this wave consumes:

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" \
    -m pytest tests/test_hf_causal_gqa_qk_norm_fold.py -q -p no:randomly
```

`4 passed in 14.54s` — including its byte-identity golden receipts, which is the
evidence that the kernel was consumed and not modified.

Coverage: canonical-reference bit-identity (bias and no-bias), gate-collapse
exactness (square and heterogeneous channels), remainder monotonicity in gate
variance, collapsed and live Direction-B compiles with every diagnostic asserted,
five Direction-B fail-closed paths, row-exact band decomposition, Direction-A fit
against the class floor, the non-separable-band comparison, unrouted far mass,
far-mass routing with a sealed bundle, inconsistent evidence, partition overlap
in both directions, the routing checkpoint rule in both outcomes with a real
compiled carrier, projection-replayed evidence, remainder-routing preconditions,
and byte identity across builds for both directions.

---

## 9. Known limitations

1. **Toy scale only.** Stage 2 (single real LFM2 layer), stage 3 (whole-model
   kind swap) and stage 4 (cross-family) are untouched. Nothing here is evidence
   about a real model.
2. **Direction B's band survives on the residual stream's DC component**, and it
   degrades fast without one. Sweeping the toy's row noise against a fixed DC of
   norm `1.5*sqrt(8)`, collapsed gates, mass floor disabled so the endpoint is
   what is observed:

   | row noise | min validation assigned-lag mass | validation endpoint rel. RMSE |
   |---|---|---|
   | 0.15 | 0.874 | 0.024 |
   | 0.50 | 0.521 | 0.318 |
   | 1.00 | 0.371 | 0.715 |
   | 2.00 | 0.242 | 1.000 |
   | 3.00 | — | **fails closed** at the no-branch baseline (1.357) |

   Real RMSNorm outputs are DC-dominant, but only the per-layer replayed
   probability diagnostic says so for a given layer. Both gates are tested: the
   mass floor at `0.99` and the endpoint baseline with the mass floor disabled.
3. **`rho_2` on the toy is small (`3.4e-4` at gate scale 1)** because the toy's
   gates are deliberately mild. A real LFM2 conv may have far more gate
   dispersion, in which case the routed `eps` dominates and the carrier gate
   decides the conversion. Untested at realistic dispersion.
4. **The BF16 direction-B replay reimplements the kernel's replay loop** in the
   target dtype. It tracks the FP64 replay to `0.008` relative on the toy, but it
   is a second implementation of the same semantics, so a future change to
   `HFCausalCoupledAttentionFold.replay` would not be reflected automatically.
   A `dtype=` parameter on the kernel's own `replay` would remove the duplication.
5. **`D > H` channels are seeded with adjacent-pair combinations**, not
   independent directions — `x` is one linear image of `n`, so more than `H`
   independent channel directions do not exist. Those channels are no longer
   inert (they were, with the plain PLS init, and produced numerically identical
   results for `D = 8, 12, 16`), but nothing here proves they help.
6. **Direction A's per-channel Gauss-Seidel is O(D n H L) per sub-step.** Fine at
   toy scale; at `D = H = 1024` with a real corpus this is the compute risk the
   design's last bullet names, and it belongs on the AI server.
7. **The routing checkpoint helper takes predictions, not a carrier compile.**
   The rule is implemented and tested end to end, but the *loop* that recompiles
   a carrier both ways and rolls back is executor work (section 7 item 4).
8. **The 24-candidate ladder is not resumable.** Cold and warm compiles are
   byte-identical because everything is closed form, but there is no phase cache;
   at real scale the ladder should be keyed and sealed like the v5 lane's.
