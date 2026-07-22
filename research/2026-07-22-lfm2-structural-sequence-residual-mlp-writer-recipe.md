# LFM2 structural-sequence residual-MLP-writer recipe receipt

Date: 2026-07-22

Status: implemented and locally unit-tested; bounded folding experiment **not
run**. This document records a candidate lane, not an intelligence result.

## Authenticated recipe

Recipe ID: `structural_sequence_residual_mlp_writer`

Recipe SHA-256:
`11298ed377b2ad775b5a1a29cae3dc38292adf995af88dc197d601d4c2f7b784`

| Component | Authenticated lane |
|---|---|
| Q/K | `maximum_trust_radius_full_alpha_source_derived_residual` |
| V/O | `joint_then_fixed_structural_priority` |
| short convolution | `full_source_structural_priority` |
| MLP | `uncapped_w2_residual_writer_priority` |

This is one complete non-Cartesian body recipe. It keeps the full-source
structural sequence-producing path through attention and short convolution,
keeps full-source W1/W3 producers, and applies the analytic residual-writer
choice only at W2.

## Why this bounded lane exists

A0004 manual review found no coherent candidate in `fit_guided`,
`full_strength_structural`, or `residual_writer`. The observed decoded failures
changed with body composition, while terminal scalar variation changed only the
surface attractor. The existing `source_producer_residual_writer` experiment
combines residual-writer V/O with structural short convolution and residual W2.
This fifth lane is the remaining one-variable isolation: structural V/O instead
of residual-writer V/O, while retaining residual W2.

The exact composition was absent from the authenticated recipe inventory and
reviewed experiment history. It does not repeat scale sweeps, standalone
projection/OT/PCA variants, whole-FFN sharing, endpoint-only repair, or the
previous residual-writer sequence composition.

## Decision authority

Candidate metrics are advisory diagnostics only. They cannot accept, reject,
promote, delete, or semantically rank this recipe. Structural validity and an
effectful non-incumbent proposal are the only compiler eligibility checks.
Decoded whole-response human review remains the authority for coherence,
intelligence retention, and whether the lane should continue.

No training, optimizer, backward pass, pruning, quantization, sparsification,
remote execution, model inference, or behavioral evaluation was performed for
this implementation receipt. A sealed-foundation replay and qualitative review
are still required before any result can be claimed.

## Compatibility and invocation

Writers now emit the five-recipe inventory. Readers continue accepting the
sealed historical three- and four-recipe inventories when their materialized
recipe belongs to that historical inventory. The existing postfold CLI and
one-command fold-to-review driver obtain choices from the authenticated
inventory, so the new recipe is selected with:

```text
--body-recipe structural_sequence_residual_mlp_writer
```

