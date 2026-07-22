# LFM2 normalized-input decoder-only compiler

Date: 2026-07-22

## Outcome

`hf_lfm2_normalized_input_decoder.py` isolates the one normalized child-input
to parent-input decoder required by nonlinear Q-DEIM. It stops before source
weight signatures, neuron assignment, clustering, medoid selection, W1/W3
materialization, W2 aggregation, or endpoint correction.

The output is an authenticated, detached, contiguous CPU FP64 tensor with
shape `[target_hidden, source_hidden]`. Its receipt is self-sealed: the
receipt contains hashes of every authenticated input, FIT evidence, row
weights, normalized chart, polar diagnostic, structural inverse, rowspace
projector, correction, and final decoder, plus a `receipt_sha256` over the
canonical receipt body. `verify()` independently checks the decoder bytes,
receipt seal, and detached CPU FP64 storage.

## Extracted construction

The implementation preserves the existing normalized real-medoid compiler's
operation order and math exactly through its decoder invariants:

1. Authenticate the source-to-target residual chart and both RMSNorm gamma
   vectors.
2. Build the gamma-aware normalized chart.
3. Use the chart transpose for authenticated orthonormal columns, otherwise
   one thin SVD left inverse.
4. Conjugate that raw left inverse by parent and target gamma.
5. Normalize the structural inverse by its operator norm.
6. Solve one strictly positive global gain from all weighted FIT rows.
7. Compute the FIT input rowspace with one thin SVD.
8. Add the ridge-scaled FIT-rowspace correction, capped at the supported
   structural-prior norm with fixed rho equal to one.
9. Preserve the gain-scaled structural decoder exactly in the FIT nullspace.
10. Verify full decoder rank, polar geometry, rowspace confinement, and the
    bounded null completion.

Those checks are structural construction invariants. Numeric diagnostics do
not choose, accept, or veto semantic candidates. Whole decoded behavior
remains the semantic judgment surface.

## Equivalence and failure proof

Focused validation:

```text
4 passed in 6.71s
```

The test fixture proves:

- bitwise equality with
  `compile_lfm2_normalized_input_real_medoid_structural_mlp_fold`;
- identical final decoder hash and rank;
- identical FIT evidence hash and FIT row-weight hash;
- identical complete `producer_bridge` receipt fields;
- self-sealed receipt verification;
- failure on a mismatched authenticated gamma tensor;
- failure on non-positive FIT case weights;
- failure on zero-rank FIT geometry; and
- failure on a rank-deficient residual chart.

## Zero-cluster operation proof

The focused test replaces all three legacy plan builders with functions that
raise immediately:

- weight-complete MLP signature builder;
- multi-anchor cluster builder; and
- activation cluster builder.

The decoder-only compile still completes, and every spy call count is zero.
The materialized receipt also records:

```text
source_mlp_weight_tensor_reads: 0
weight_complete_signature_builds: 0
activation_cluster_plan_builds: 0
multi_anchor_cluster_plan_builds: 0
medoid_selection_operations: 0
source_neuron_assignment_operations: 0
producer_materializations: 0
consumer_materializations: 0
model_forward_calls: 0
heldout_rows_used_by_solver: 0
backward_calls: 0
optimizer_steps: 0
training_steps: 0
```

## Local fixture timing diagnostic

After one warm-up call, the same tiny deterministic fixture was measured with
20 decoder-only samples and five legacy full-compiler samples:

```text
decoder-only median: 0.0022225 seconds
legacy full median:  0.0134767 seconds
fixture ratio:       6.06x
```

This is a scope sanity check, not a production performance claim and not a
semantic decision signal. The important proof is that cluster and medoid
operations are structurally absent.

## Minimal Q-DEIM integration seam

```python
control = compile_lfm2_normalized_input_decoder(
    residual_chart=inventory.residual_chart,
    residual_chart_sha256=lfm2_authenticated_tensor_sha256(
        inventory.residual_chart
    ),
    parent_ffn_norm_weight=inventory.ffn_norm_weight,
    parent_ffn_norm_weight_sha256=lfm2_authenticated_tensor_sha256(
        inventory.ffn_norm_weight
    ),
    target_ffn_norm_weight=target_ffn_norm_weight,
    target_ffn_norm_weight_sha256=lfm2_authenticated_tensor_sha256(
        target_ffn_norm_weight
    ),
    cases=normalized_cases,
    case_sample_weights=tuple(item.sample_weight for item in fit),
    normalized_decoder_ridge=1e-8,
)

qdeim = compile_lfm2_mlp_nonlinear_qdeim(
    ...,
    normalized_input_decoder=control.normalized_input_decoder,
    expected_normalized_input_decoder_sha256=(
        control.normalized_input_decoder_sha256
    ),
    ...,
)
```

Only the import, control constructor, result type, and control receipt handoff
need to change. No source W1/W3/W2 tensors or blank target templates belong in
the decoder-control call.
