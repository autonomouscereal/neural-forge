# Residual-targeted V basis folding (2026-07-11)

## Why this exists

The strict FP32 v2 causal oracle separated two facts which a raw V-cache
regression conflated on one code prompt, one `L19/G4` site, and four forced
cached actions:

- substituting parent V helped the p35 control;
- the same raw parent-V intervention harmed the p34 candidate;
- restoring the complete parent attention residual write helped p34.

For that measured path, “make the folded V look like parent V” is not a
sufficient target.
Once Q, K, and O have been folded, the useful V is the one whose read through
the *candidate's* attention and current O produces the right residual write.
It may be a mixture which is not equal to either raw parent V prototype.

This is a localization result, not evidence that raw V reconstruction is
always harmful. The source reports are
`runs/residual-attention-20260711/p35_layer19_group4_h4_forced_cache_oracle_fp32_v2.json`
and
`runs/residual-attention-20260711/p34_layer19_group4_h4_forced_cache_oracle_fp32_v2.json`.

`hf_residual_attention_value_basis.py` implements that separate alternative.
It does not replace the raw-V four-control probe.

## The operator

For every logical site `s` which shares one survivor prototype, and every
cached forced-decode action `t`, the fitter captures candidate attention
probabilities `A[s,t]`. For a small explicit basis of existing raw V
prototypes `V[i]`, it computes:

```text
X[s,t,i] = O_candidate[s](A_candidate[s,t](v_scale[s] * V[i](U[s,<=t])))
Y[s,t]   = O_parent[s](parent_selected_group_context[s,t])
```

It then solves the bounded anchored ridge:

```text
min_alpha  ||sum_i alpha[i] X[i] - Y||^2
           + lambda ||alpha - alpha_prior||^2
subject to lower[i] <= alpha[i] <= upper[i]
```

The basis is deliberately tiny (maximum eight by default) and must contain:

- exactly one current survivor;
- at least one removed source;
- optional additional real V prototypes.

The coefficient solve enumerates finite box active sets (`3^basis_size`) and
solves only small linear systems. It calls no optimizer, creates no gradient,
and retains no coefficient at runtime. The final tensor is materialized once:

```text
V_survivor := sum_i alpha[i] * V[i]
```

The existing rollback-safe V installer copies that tensor into the existing
survivor parameter. Parameter count and parameter-object identity do not
change.

## What is globally included

The fit refuses partial evidence. Calibration must cover every and only
logical site assigned to the survivor. At each site it explicitly applies:

- the canonical-to-physical `group_permutations` mapping;
- that site's V role scale;
- that site's O role scale through the live materialized O block;
- the current candidate attention probabilities;
- the parent materialized O block for the same physical group.

The parent O bias is excluded because a layer-level bias cannot be uniquely
attributed to one attention group.

## Exact p35 `L19/G4 -> L21/G6` integration contract

The intended first live use is the known p35 redirect. The source provenance
indices are:

```text
removed source:  L19/G4 = 19 * 8 + 4 = 156
survivor source: L21/G6 = 21 * 8 + 6 = 174
```

The required order is:

1. While the p35 incumbent is installed, capture both basis entries with
   `capture_value_prototype_basis_entry`: source 174 with role `survivor`, and
   source 156 with role `removed-source`.
2. Also capture the incumbent materialized parent O column block for every
   affected executable site.
3. Preview and install the tentative redirect. Do not try to find source 156
   after compaction; its immutable basis payload was intentionally captured
   before removal.
4. Prove the tentative survivor's full assignment set. For this lineage it is
   `L2/G1`, `L19/G4`, and `L21/G6`. Convert each physical executable group to
   its canonical assignment axis using the registry permutation.
5. Collect aligned parent and tentative-child forced cached traces on the
   calibration prompts.
6. For each prompt/site pair, call
   `build_residual_value_basis_site_calibration(...)` with the two captured
   basis entries and that site's parent O block.
7. Call `fit_residual_targeted_value_basis(...)` for the tentative survivor
   slot. Start with explicit conservative coefficient bounds; the report
   records the bounds, active set, condition number, residual error, tensor
   hashes, trace hashes, and all affected sites.
8. Install only with `install_residual_targeted_value_basis(...)`. It verifies
   that the complete candidate registry—including Q/K/O and every V tensor—is
   byte-identical to the state used for calibration, then delegates to the
   exact rollback handle.
9. Recollect all affected child traces after V installation. The fixed-Q/K
   calibration is now stale by design, especially for downstream layers.
10. Run the existing block-sparse O fit against the new actual child `A @ V`
    contexts, globally across the same survivor assignment set.
11. Evaluate held-out forced and greedy cached lanes. Commit neither tensor
    unless the complete transaction passes the calibrated gates and exact
    save/reload audit.

The existing experiment helper `_fit_survivor_blocks` already implements the
needed selected-group, shared-survivor O solve for this exact topology. The
public `refit_output_after_value_installation` is a full-layer alternative;
do not confuse it with the block-sparse p35 contract.

## Fail-closed guarantees and limits

- forced aligned traces only for calibration;
- exact candidate trace V hash against the live materialized V projection;
- exact registry content hash binding the captured Q/K/V/O state;
- at least two explicitly provenance-bound real V prototypes;
- complete survivor-site coverage;
- bounded basis, rows, scalar observations, dimensions, active sets, and
  condition number;
- coefficient prior must lie inside explicit bounds;
- immutable design, target, calibration, basis, fitted-payload, and registry
  hashes;
- exact unaffected-V/non-V identity and rollback through the existing
  installer;
- zero new runtime parameters and zero optimizer/backward calls.

Synthetic tests cover exact coefficient recovery across two sites with
different V/O scales and a non-identity group permutation, missing-site
rejection, stale-Q/K/O registry rejection, in-place installation with no new
parameters, and byte-exact rollback.

## Interpretation

This is a bounded hypothesis test, not proof of compaction. A positive result
would show that the missing p34 V can be represented as a residual-useful
combination of already retained prototype directions. A negative result is
also informative: if even a wider real-prototype basis cannot recover the
parent group write under fixed candidate attention, the next required basis
must include a K-aware or joint Q/K/V residual operator rather than a larger
raw-V transplant.
