# LFM2 CPCF Short-Conv V5 Numerical Policy

Date: 2026-07-18

## Status

The additive v5 numerical primitives are implemented in
`src/neural_forge/hf_lfm2_cpcf_short_conv_v5.py`. They do not modify or wrap
the v4 implementation. No v4 schema, policy, report, fingerprint, candidate,
resume artifact, or selection result changes merely by importing the new
module.

This is implementation evidence, not a successful v5 fold. The primitives
still require an explicitly versioned v5 compiler policy, a new compiler and
resume fingerprint, real layer-0 execution against sealed cached evidence, and
held-out macrocell comparison before they can advance the folding result.

All four primitives are deterministic, fit/selection isolated as appropriate,
and zero-training. They perform no model forward, backward call, optimizer
step, parameter pruning, quantization, or validation-driven retry.

## Endpoint-first sealed selection

`rescore_sealed_v4_short_conv_selection_frontier()` consumes only fully sealed
v4 `target_dtype_evaluation` reports whose split is exactly `selection`. Its
interface has no validation argument. It rejects reports without full context
evidence, actual target-dtype parameter/intermediate replay, identical evidence
contracts, or exactly one transported control whose candidate id ends in
`--anchor-00-prior`.

The policy is frozen:

- endpoint maximum case relative RMSE: at most `1.0`;
- context maximum case relative RMSE: at most `1.0`;
- endpoint minimum case cosine similarity: at least `0.0`;
- context minimum case cosine similarity: at least `0.0`; and
- prior-relative context exchange coefficient: exactly `beta = 0.05`.

For candidate `c` and transported prior control `p`, maximum and mean RRMSE are
checked independently:

```text
max(0, Rh(c) - Rh(p)) <= 0.05 * max(0, Re(p) - Re(c))
```

Minimum and mean cosine similarity are also checked independently:

```text
max(0, Ch(p) - Ch(c)) <= 0.05 * max(0, Ce(c) - Ce(p))
```

The frontier fails closed if no candidate is feasible. Feasible candidates use
this exact lexicographic key:

```text
(
  endpoint maximum_case_relative_rmse,
  -endpoint minimum_case_cosine_similarity,
  endpoint mean_case_relative_rmse,
  -endpoint mean_case_cosine_similarity,
  context maximum_case_context_relative_rmse,
  -context minimum_case_context_cosine_similarity,
  context mean_case_context_relative_rmse,
  -context mean_case_context_cosine_similarity,
  candidate_id,
)
```

There is no minimum-gate-slack objective. A synthetic fixture matching the
authenticated layer-0 frontier facts selects the alpha `1e-4`
`pass-01-writer`, then alpha `1e-8` `pass-01-writer`, then the alpha `1e-4`
half chord. A context-exchange violator and the observed `1.001542` context
RRMSE final-BCX shape are rejected.

## Equilibrated prior-residual ridge

`condition_capped_equilibrated_prior_residual_ridge_frontier()` produces the
fixed `kappa=1e3` and `kappa=1e4` candidates. Its default
`fit_intercept=True` mode preserves the original affine behavior. It:

1. computes residuals relative to transported prior weights and bias;
2. uses equal-case weighting so row duplication cannot change case weight;
3. centers the design and prior residual while solving the affine intercept
   separately;
4. equilibrates each identifiable design column;
5. factors the symmetric equilibrated Gram matrix with
   `torch.linalg.eigh`;
6. chooses the minimum nonnegative ridge needed for the declared condition
   cap;
7. explicitly zeros numerical-nullspace eigen-coordinates, preserving the
   transported prior there; and
8. intersects independent parameter-space and function-space trust bounds.

The parameter bound compares coefficient correction RMS with transported
parameter RMS, in coefficient units. The effect bound compares predicted
effect correction RMS with prior/target effect RMS, in function units. The
two references are never substituted for one another; this directly removes
the v4 audit's parameter-versus-activation unit mismatch.

Each candidate seals case identities, normalized tensor hashes, centering,
equilibration, Gram and right-hand-side hashes, eigenvalues and eigenvectors,
raw/accepted eigen corrections, nullspace proof, condition, prior and solution
hashes, and both trust receipts.

FIR and linearized B/C/x designs already encode every physically installable
bias coordinate in their design matrices. They must call the same solver with
`fit_intercept=False` and `prior_bias=None`. This mode does not center the
design or residual, does not create a separate intercept, and returns
`bias=None` under the sealed `absent_frozen_zero` contract. The report binds
that the bias value is absent, its effect is frozen at zero, all bias hashes
are null, and the uncentered case-balanced system is the one that was solved.
Supplying a separate bias in this mode fails closed.

## FIR-commuting balanced all-channel basis

`build_fir_commuting_balanced_all_channel_basis()` accepts only fit evidence:
the source depthwise FIR plus per-source activation and endpoint descriptors.
It equilibrates descriptor blocks, chooses source seeds with deterministic
farthest-first traversal and lowest-index ties, and assigns every remaining
source to a capacity-balanced target group.

Every source receives a strictly positive coefficient derived from the square
root of its within-group energy share. Disjoint group support makes the encoder
columns orthonormal and its transpose an exact dual. It also makes

```text
decoder @ diag(source_fir[:, lag]) @ encoder
```

diagonal for every lag. That projected off-diagonal fact is necessary but
vacuous for a disjoint-support grouping: it does not prove that one child FIR
coefficient reproduces all source FIR coefficients assigned to the group.

The real representability test is the intertwining residual

```text
||diag(source_fir[:, lag]) @ encoder
  - encoder @ diag(target_fir[:, lag])||_F
/ ||diag(source_fir[:, lag]) @ encoder||_F
```

for every lag. `maximum_per_lag_commutator_relative_frobenius` is an
authenticated policy limit and the builder fails closed when the maximum
observed residual exceeds it. The strict default is `1e-12`; a production
compiler that chooses a looser approximation must state and fingerprint that
tolerance explicitly. The report preserves projected off-diagonal energy as a
diagnostic, explicitly says it is insufficient for a commuting claim, and
binds the per-lag values, maximum value, real gate, and pass/fail result.

Passing the commutator gate proves only representability within the declared
numerical tolerance. It does not by itself prove that the tolerance is
behaviorally acceptable; that still requires real selection and validation
replay.

## Every-accepted-edge interpolation

`interpolate_every_lfm2_cpcf_v5_accepted_edge()` requires each accepted path
state to authenticate its immediate predecessor. It applies the fixed interior
continuation grid to every adjacent accepted edge, interpolates every tensor
role, and seals both per-candidate and complete-coverage receipts.

The invariant is explicit:

```text
covered_accepted_edge_count == accepted_path_state_count - 1
skipped_accepted_edge_count == 0
```

This expands continuation around each accepted writer, FIR, and B/C/x
transition instead of searching only the prior-to-final and
prior-to-first-writer chords.

## Validation

The focused local suite covers:

- exact endpoint-first ranking and the layer-0 alpha `1e-4` writer outcome;
- report tampering, validation leakage, fixed-policy drift, and zero-feasible
  rejection;
- ill-conditioned and null-column ridge systems;
- exact intercept recovery and independent same-unit trust bounds;
- uncentered no-intercept recovery when the physical bias is an encoded design
  coordinate, plus rejection of a hidden separate bias;
- row-duplication and column-scale invariance;
- balanced deterministic all-channel FIR grouping, exact dual, zero pruning,
  projected off-diagonal diagnostics, and real commutator-gate pass/fail; and
- authenticated complete interpolation coverage over multiple accepted edges.

The remaining orchestration work is to bind these primitives into the new v5
compiler and resume identity, replay the complete BF16 frontier on both RTX
3090s, freeze the selection winner before validation, and compare the installed
layer-0 macrocell with the frozen v4 checkpoint. The frozen v4 run must remain
untouched.
