# Residual-attention sequence-operator experiment

This is an ephemeral, synthetic, zero-training experiment against the
production full-attention frontier in
`hf_lfm2_cpcf_layer_compiler.py`. It did not modify tracked source or tests.

## Exact setup

- Production controls: `direct_monolithic` and
  `residual_attention_split`; both share the exact fit-compiled Q/K producer.
- Alternative: keep that exact Q/K, form each causal head's attended-input
  operator `z_h = P_h X_K`, ridge-fit the complete residual delta as an
  effective multi-head operator, extract a shared value subspace by rank-4
  SVD, and monotonically alternate closed-form V/O solves. A producer-value
  subspace anchor preserves context information.
- Source: hidden 32, 8 query heads, 4 KV heads, head dimension 4.
- Child: hidden 8, 2 query heads, 1 KV head, head dimension 4.
- Reduction: 4x hidden, 4x query heads, 4x KV heads; 3,072 to 192 dense
  attention parameters (16x, or 93.75% fewer parameters).
- Sequence: 6-token causal prefill plus 6 ordered decode steps.
- Three independent seeds: 1101, 2202, 3303.
- Per seed: 28 fit, 14 selection, and 28 held-out sequences.
- Total held-out causal query rows: 1,008.
- Alternative frontier: ridge scale in `{1e-10, 1e-8, 1e-6, 1e-4}`,
  horizon power in `{0, 0.5, 1}`, and producer-value anchor fraction in
  `{0, 0.1, 0.25, 0.5}`: 48 candidates. Selection never reads heldout.
- Selection score:
  `endpoint RRMSE + 0.20 * mean closed-loop endpoint RRMSE
  + 0.10 * final closed-loop endpoint RRMSE
  + 0.20 * context readout RRMSE`.
- Training, backward, and optimizer steps: zero.

## Aggregate held-out results

| Candidate | Endpoint RRMSE | Endpoint cosine | Attention JS | Context readout RRMSE | Mean closed-loop drift | Final drift |
|---|---:|---:|---:|---:|---:|---:|
| Direct monolithic | 0.134925 | 0.990770 | 0.00315774 | 0.122588 | 0.210287 | 0.250470 |
| Residual-attention split | 0.130058 | 0.991408 | 0.00315774 | 0.137132 | 0.221057 | 0.276004 |
| Current selection winner | 0.130058 | 0.991408 | 0.00315774 | 0.137132 | 0.221057 | 0.276004 |
| Sequence effective operator | **0.121525** | **0.992507** | **0.00315774** | **0.136013** | **0.198951** | **0.241157** |

Relative to the current selection winner, the alternative improved held-out
endpoint RRMSE by 6.56%, residual-delta RRMSE by 6.63%, mean closed-loop drift
by 10.00%, final drift by 12.63%, and context-readout RRMSE by 0.82%.
Attention-map divergence was exactly unchanged because Q/K was byte-identical.

The alternative beat both production frontier arms on held-out endpoint RRMSE
and mean closed-loop drift on every seed. Selected settings were:

| Seed | Ridge scale | Horizon power | Value anchor | Accepted ALS solves |
|---|---:|---:|---:|---:|
| 1101 | 1e-4 | 0.0 | 0.25 | 12 |
| 2202 | 1e-4 | 0.0 | 0.25 | 7 |
| 3303 | 1e-10 | 0.0 | 0.50 | 12 |

The horizon-weighted variants did not win. The gain came from fitting the
complete fixed-attention sequence operator and factorizing it into shared V/O,
not from simply upweighting later decode rows.

## Integration decision

The result supports adding
`fixed_qk_sequence_effective_vo_als` as a third bounded full-attention
macrocell candidate. It should not replace the direct control: direct retained
the best raw compressed-context readout in this fixture, while the new arm
retained the current selected split's context quality and won the actual
residual endpoint and closed-loop metrics.

This synthetic result is evidence for a production candidate, not proof of
real-model quality. The next integration test should give the new arm its own
fit-only MLP, run the existing selection macrocell, and require it to beat both
controls on real cached LFM2 prefill/decode evidence before promotion.

## Reproducibility

- Script SHA-256:
  `90fc2b34b0f3ecc180d37c36d1d4cd8a818975ee8630b139051dad246e99ee6f`
- Result JSON SHA-256:
  `0394a4aadd6c091f7fc0d00e3fbf998d0b7caff32f65f752660a4a247d7bba27`
- Two consecutive executions produced byte-identical result JSON.
