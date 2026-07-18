# LFM2 CPCF 100x Short-Conv Failure Record (2026-07-18)

## Summary

The `v3` run did **not** produce a compacted child. It failed closed while
validating a compiled LFM2 short-convolution layer, before that operator, its
MLP, or the complete model could be accepted. The candidate directory,
promotion receipt, and staging residue do not exist. This is useful negative
evidence, not proof of successful compaction and not proof that 100x
zero-training folding is impossible.

The next run must be a materially different `v4`: selection-isolated,
endpoint-aware, signed, source-prior-preserving short-conv folding with
scale-normalized regularization, moment repair, stronger diagnostics, and
sealed resumable intermediate artifacts. It must not weaken the held-out gate.

Relevant 2025-2026 primary sources:

- [Cut Less, Fold More: Model Compression through Projection Geometry
  (2026)](https://arxiv.org/abs/2602.18116)
- [ICLR 2025 model-folding paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/f102c495b31bcc7b2a11c27de75475fc-Paper-Conference.pdf)
- [ICLR 2025 tensorized state-space paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/4c12e97f2e05304a451e18c9c945036f-Paper-Conference.pdf)
- [FLAT-LLM: Fine-grained Low-rank Activation Space Transformation
  (2026)](https://aclanthology.org/2026.findings-eacl.156/)
- [IMPACT: Importance-Aware Activation Space Reconstruction
  (2026)](https://aclanthology.org/2026.acl-long.183/)
- [The Curious Case of In-Training Compression of State Space Models
  (ICLR 2026)](https://openreview.net/forum?id=LtzmeSMBTW)
- [Rethinking the Role of Tensor Decompositions in Post-Training LLM
  Compression (2026)](https://arxiv.org/abs/2606.03465)

The 2026 evidence sharpens the v4 design. FLAT-LLM reports that activation
space is a better structural-compression coordinate than raw weight space.
IMPACT further reports that activation dimensions must be weighted by their
effect on behavior rather than treated uniformly. The ICLR SSM work uses
balanced truncation and Hankel singular values to preserve input-output
behavior. Conversely, the 2026 tensor-decomposition study finds that
Frobenius-optimal shared subspaces can distort heterogeneous, function-bearing
operator geometry. Therefore v4 must select directions that are both
predictable from the incoming child state and observable at the residual
endpoint. Raw weight SVD, raw activation PCA, or positive channel averaging
alone is not an adequate objective.

## Frozen `v3` Experiment

This experiment was parameter compaction into a newly constructed blank
target. It was not pruning, training, recovery tuning, or quantization.

Parent:

- LiquidAI LFM2.5-230M, local snapshot only
- 229,693,184 unique parameters
- hidden size 1024, FFN size 2560, 14 layers
- 16 query heads, 8 KV heads, convolution cache length 3
- vocabulary size 65,536, BF16
- layer order:
  `conv, conv, full_attention, conv, full_attention, conv, full_attention,`
  `conv, full_attention, conv, full_attention, conv, full_attention, conv`
- snapshot:
  `/home/cereal/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots/37b30cce3446f3f2e26a0d3f8c67c9167f5079d7`

Blank target:

- exactly 2,291,840 unique parameters
- 100.22217x fewer unique parameters than the parent
- hidden size 128, FFN size 175, 14 layers
- 2 query heads, 1 KV head, head dimension 64
- vocabulary size 4,096 through the strict token bridge
- BF16

Local-only inputs:

- bridge:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2`
- parent trace cache:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95`
- corpus:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_bridge_folding_corpora_20260717/v4096_broad256_v1/trajectories.json`
- 256 trajectories; each contains exact prefill plus `decode.000` through
  `decode.007` evidence for all 14 parent layers
- approximately 90 GB of sealed cached evidence; the parent was not repeatedly
  sampled during folding

Independent pins:

| Input | SHA-256 |
|---|---|
| Parent model identity | `96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb` |
| Bridge artifact | `5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384` |
| Bridge profile | `e3fc98c666b70b367962e400435afe7588a9f1a2b13619dffc6f40c6458c5ca1` |
| Target tokenizer semantics | `cffb809c40ddd3382131a8ed8b1b8536457da4e46dfced85dff5669fb6d73342` |
| Target tokenizer package | `5e76c6c0f9180ac69fae4729ecf2239100d7409fd148be14d80574340160675b` |
| Corpus | `c680ff99da0bf9c6db06347873d89f79cdd6a4a70420b138d6c8e1281a2f7dd7` |
| Cache manifest file | `5c4c1db1e94dad400bec8a86d1b74e6ecfcd75135c9c6253f7085e7c4aa56300` |
| Cache manifest contents | `5cef56936c8d407d27ac55aae5d9f3716ba4f42730592d53460277e9ca6c72d1` |
| Cache completion record | `f81a91a012b00fc7e4171c78490d6d53b825165caba52e52522a2a598ae155ab` |

## Exact Outcome

The run executed for about 2 hours 21 minutes, reached approximately 115 GB
RSS and more than 31 CPU-hours, and exited normally through the fail-closed
error path. It did not OOM and the host showed no resource-pressure failure.

Log:

`/home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_folds_20260717/lfm2_230m_to_v4096_100x_cpcf_v3.log`

Log SHA-256:

`dd51573ee53bdc4cad717e52231141d1c6d791a89e9c525ad9bfec784673c0de`

Exact terminal error:

```text
error: Compiled CPCF short-conv failed enforced behavior gates: validation_endpoint_relative_rmse
```

Observed gate state:

- fit endpoint relative RMSE passed its `<= 1` requirement;
- fit endpoint cosine passed its `>= 0` requirement;
- validation endpoint cosine passed its `>= 0` requirement;
- validation endpoint relative RMSE was greater than 1 and failed;
- an all-zero endpoint prediction has relative RMSE 1, but this is not the
  same baseline as retaining the incoming residual and omitting the branch.
  `v3` did not measure that identity/no-branch baseline explicitly.

The failure occurred after replaying current child inputs for the failing
short-conv layer and before installing its compiled operator, fitting its MLP,
or evaluating whole-model behavior. Because layer zero is convolutional, the
failure could be in layer zero or a later convolutional layer. The exception
discarded the layer index, compilation stage, incoming-state-bank identity,
per-case metrics, and solver diagnostics. That loss of failure context is an
observability defect to fix before `v4`.

### What the gate proves

- At least one `v3` compiled short-conv endpoint generalized worse than the
  zero-prediction RMSE reference on its untouched validation cases.
- Passing fit error and non-negative validation cosine was insufficient; the
  fitted magnitude and/or case generalization was unstable.
- The transactional compiler correctly rejected the candidate and did not
  publish a partial child.

### What the gate does not prove

- It does not prove that any complete 2,291,840-parameter child was coherent,
  incoherent, or even executable; no complete child existed.
- It does not test the complete attention/MLP/residual stack, generation, the
  independent 64-case holdout, or manual whole-response intelligence.
- It does not show that token bridging, 100x parameter compaction, CPCF, or
  LFM2 short-conv folding is fundamentally impossible.
- It does not establish that validation RMSE failed because of one specific
  numerical or structural cause; `v3` recorded too little diagnostic context
  to make that attribution.

## Diagnosis

### Numerical and selection weaknesses

1. The ridge value was an absolute `1e-8`, not normalized by design-matrix
   energy or condition. It therefore changes meaning when activation scale
   changes.
2. The intercept was penalized with the other coefficients.
3. Writer fitting used only fit-suffix rows and did not use grouped,
   trajectory-balanced model selection.
4. The corpus already has `selection`, `fit`, and `validation` partitions, but
   the short-conv compiler admitted only `fit` and `validation`. It discarded
   the proper selection partition instead of using it to select solver policy.
5. Operators were solved in higher precision and then materialized in BF16
   without separately gating the BF16 endpoint degradation.
6. Parent input and output projections were not transported as source priors.
   The compiler regressed them from traces while retaining only the parent FIR
   coefficients directly.
7. Channel groups used farthest-seed, one-pass assignment rather than a
   converged deterministic clustering or endpoint-predictive basis.
8. Positive L1-normalized merge weights prohibited signed cancellation.
9. The input projection, context/FIR representation, and writer were optimized
   as separate local problems rather than selected against the final residual
   endpoint.

These facts are consistent with unstable gain, noise fitting, or extrapolation:
the validation direction remained weakly non-negative while its endpoint
magnitude error exceeded the zero-prediction reference.

### Structural weakness

Independent channel averaging does not commute with a channelwise dynamic
operator. In general,

```text
(sum_s a_s B_s) (sum_t a_t X_t) != sum_s a_s (B_s X_s)
```

The left side creates cross terms absent from the parent. A folded FIR and
separately folded input/output maps add further cross terms. Positive
barycenters can also erase source circuits that depend on signed cancellation.
Consequently, matching each local tensor or channel statistic does not
guarantee preservation of the composed residual write. The current happy-path
test uses effectively identical channels and does not stress heterogeneous
channels, signed cancellation, high compression such as 1024-to-128,
ill-conditioned inputs, distribution shift, or BF16 materialization.

There is a more useful operator view. After mapping the parent operator input
into child coordinates, each parent short-conv channel contributes one
lagged cubic (Volterra) rank-one term:

```text
C_s(current) * K_s(lag) * B_s(lagged) * X_s(lagged) * output_direction_s
```

A width-128 child is therefore a coupled rank-128 CP approximation of the
parent's complete input-to-residual operator, not an average of 1,024
independent B, C, X, FIR, and writer tensors. V4 should fit that structured
operator from sampled causal evidence with deterministic blockwise
variable-projection/ALS solves. This avoids materializing the enormous cubic
coefficient tensor while optimizing the composed function that the child must
actually execute.

## Required `v4` Change

`v3` must not be rerun unchanged. It is deterministic against sealed inputs and
already failed the enforced held-out gate; an identical run would spend the
same resources to reproduce the same evidence. The validation threshold must
not be weakened to manufacture a candidate.

`v4` should implement the following bounded change:

1. Persist every short-conv rejection as a sealed, non-promotable diagnostic
   receipt containing layer index, stage, incoming-state-bank hash, fit,
   selection, and validation case identities, per-case metrics, solver policy,
   condition/effective ranks, coefficient norms, RMS ratios, float64-versus-
   BF16 deltas, signed-cancellation measures, and noncommutation error.
2. Measure explicit baselines: zero endpoint, identity/no-branch residual,
   current `v3` barycenter, source-prior transport, and direct endpoint
   predictability.
3. Admit the frozen `selection` partition. Select method and regularization
   only on grouped leave-one-trajectory-out fit/selection evidence. Keep final
   validation completely untouched until the selected operator is frozen.
4. Use scale-normalized, prior-preserving ridge such as
   `lambda = alpha * trace(X^T X) / d`, with an unpenalized intercept,
   case-balanced scoring, bounded coefficient norms, and trust radii around
   transported source operators.
5. Fit the composed residual endpoint. Let `H` be actual 128-dimensional child
   inputs, `Z` the 1,024 exact parent channel contributions, and `Y` the desired
   128-dimensional residual write. First isolate the component of `Z`
   predictable from `H`; then derive signed channel directions that maximize
   covariance with `Y`, using a deterministic PLS/balanced-circuit basis rather
   than positive averaging.
6. Transport actual parent input and output projections through the selected
   channel encoder/dual decoder and use those values as priors. Run only a
   small, deterministic number of variable-projection refinements that
   alternately solve the writer, FIR, and bounded B/C/X factor corrections
   against the composed endpoint. Treat these as a sampled coupled CP/Volterra
   factorization rather than three unrelated activation regressions.
   The affine child-to-parent input lift must transport its intercept too:
   for `h_parent ~= h_child R + r`, a parent role `(W, b)` contributes target
   prior weight `E^T W R^T` and bias `E^T (b + W r)`.
   Do not explicitly expand a per-channel Jacobian over the 128 output
   coordinates. For valid suffix rows `J`, endpoint residual `R`, and writer
   column `w`, the exact equal-case normal-equation terms are
   `G = (||w||^2 / O) (J^T J / n)` and
   `q = J^T (R w) / (n O)`. Accumulate these sufficient statistics by case.
   This is algebraically identical to the expanded solve while avoiding
   multi-gigabyte per-channel tensors.
7. Select activation RMS/mean repair on the selection partition, following
   folding evidence that uncorrected activation-statistic collapse damages
   merged networks. Recheck after materializing every candidate parameter and
   replaying B/C/X, FIR, and writer intermediates in BF16; casting only the
   final FP64 prediction is not a target-dtype evaluation.
8. Seal and key reusable boundary bundles, incoming state banks, parent
   contexts, Gram/covariance/SVD summaries, and selected solver artifacts by
   parent, cache, corpus, bridge, blank target, layer, incoming state, solver,
   and policy identities. A resumed compile must be byte-identical to a cold
   compile.
9. Add adversarial tests for heterogeneous cross terms, signed cancellation,
   collinear distribution shift, scale invariance, unpenalized bias, validation
   isolation, float64/BF16 divergence, failure receipt/resume behavior,
   cold/resumed byte identity, and a synthetic 1024-to-128 stress case.

## Implemented `v4` Production Candidate

The bounded `v4` change is now implemented in the canonical `neural-forge`
repository. This is implementation and preflight evidence, not an accepted
compaction result. No real `v4` child has passed the standalone behavior gate
yet.

The implementation adds:

- composed residual-endpoint short-conv fitting with source-projection priors,
  signed channel directions, context-aware selection, deterministic
  variable-projection updates for writer/FIR/B-C-X factors, and target-BF16
  replay before promotion;
- strict selection/fit/validation isolation and fail-closed diagnostics that
  retain the layer, state-bank, solver, per-case, and rollback identities;
- an exact `2,291,840`-parameter target contract rather than an
  `at_or_below` budget;
- authenticated cold-versus-resumed continuation that reinstalls an exact
  accepted prefix into a fresh blank of the same topology and continues only
  at the next layer;
- synchronous checkpoints after the bridged seed and every accepted layer,
  containing only the child prefix and child state bank, never copied parent
  evidence;
- immutable content-addressed boundary/checkpoint/failure artifacts and an
  invocation-bound, self-sealed, atomically replaced latest-resume pointer;
- cancellation-safe rollback, transactional standalone publication, and one
  canonical receipt inside the atomically published output directory.

The full-vocabulary embedding cannot fit in this target: a tied
`65,536 x 128` table alone is `8,388,608` parameters. The strict 4,096-row
token bridge is therefore a necessary part of the exact 2.29M experiment, not
quantization or an attempt to count fewer bytes as fewer parameters.

Local validation after the final resume-pointer audit:

- `122/122` model-fold, resume-store, resume-bridge, runner, rollback,
  cancellation, tamper, and atomic-pointer tests passed;
- the broader CPCF regression reported `349 passed, 2 skipped`;
- the repository-wide run reported `4,114 passed, 6 skipped, 9 failed`.

Eight repository-wide failures are the intentional single-pass replay
allowlist rejecting local Transformers `5.13.0`; the certified allowlist and
the production server use Transformers `5.12.1`. The production fold consumes
sealed cached evidence and does not call that single-pass capture path. The
ninth failure is an unchanged legacy GPT-2-to-Llama synthetic
blank-bootstrap fixture that has been stale since an earlier feature-fold
algorithm change: the candidate improves its RMSE metrics but loses one of 64
top-1 anchors, so the strict gate correctly rejects publication. These
failures must not be presented as green, but neither failure is inside the
changed CPCF `v4` production path.

The next valid evidence is the real offline run against the already sealed
LFM2 parent, 4,096-token bridge, 256-trajectory corpus, and approximately
90 GB parent cache. It must use one BF16-capable RTX 3090, publish no candidate
on any failed layer gate, and retain its latest accepted layer checkpoint for
an exact resume.

## Acceptance Rules

A future run is not a folding success merely because it writes tensors or
passes a local reconstruction metric. Promotion requires all of the following:

- the blank target reloads with exactly 2,291,840 unique parameters and the
  authenticated 4,096-token bridge;
- no pruning, training, gradient update, recovery tuning, quantization, remote
  model loading, or unsealed parent resampling occurred;
- every required target role is deterministically derived and covered, with
  no random required operator left in the child;
- selection and validation identities are disjoint and validation never
  influences method, rank, regularization, repair, or retry choice;
- every short-conv, attention, MLP, residual-boundary, and whole-model gate
  passes after BF16 materialization;
- publication is transactional and the receipt authenticates the exact blank
  target, parent, bridge, corpus, cache, compiler policy, candidate state, and
  post-publication cleanup state;
- the published child reloads standalone and passes the separately sealed
  64-case holdout;
- manual whole-response scoring finds coherent, relevant answers rather than
  relying on exact-token matching, and the frozen aggregate capability rubric
  reaches approximately 90% or more of the parent score.

Until all of those conditions hold, the correct result is "no accepted
compaction yet."
