# LFM2 Source-Producer / Residual-Writer Recipe

Date: 2026-07-22

This record defines the fourth authenticated semantic-body recipe,
`source_producer_residual_writer`. It is an implementation receipt and a
bounded next experiment, not a coherent-child claim. No model run or checkpoint
materialization was performed while adding it.

## Why this lane exists

The existing component experiments separated two useful behaviors:

- source-derived producer structure preserves broad parent coverage; and
- analytic residual writers repair the child-side readout after compression.

The previous complete recipes coupled those choices differently. The new lane
tests their untried composition as one complete causal body while reusing the
same sealed foundation, cached evidence, anchor handling, and terminal review
contract.

## Authenticated component composition

| Component | Selected existing lane |
|---|---|
| Attention Q/K | maximum-trust-radius, full-alpha, source-derived residual |
| Attention V/O | fixed-source V followed by analytic O residual writer |
| Short convolution | full-source structural producer |
| MLP | full-source W1/W3 producers followed by analytic W2 residual writer |

The recipe hash under
`neural-forge-lfm2-semantic-body-recipe-v1` is:

```text
5767c45aeb2b022c50fd084d4c7e17d9451b33b6a188c4fba4e9c705095cc4da
```

This is not a new fitting algorithm and does not add an optimizer. It composes
already implemented source-derived and closed-form analytic folding
primitives. It performs no training, backward pass, pruning, quantization,
sparsification, parameter sharing, runtime adapter, or teacher recapture.

## Dynamic selection and compatibility

The recipe is selectable anywhere the semantic postfold body recipe is
selected, including the public runner. Component dispatch is authenticated by
the selected component-lane identity rather than by special-casing a complete
recipe ID. This lets later complete recipes reuse a proven component lane
without duplicating dispatch policy.

New writers seal the current ordered four-recipe inventory. Readers accept
either the original ordered three-recipe inventory or the current ordered
four-recipe inventory. Arbitrary subsets, reordered inventories, inconsistent
counts, and a legacy inventory claiming the fourth recipe are rejected. This
keeps the already produced a0004 artifacts readable without changing or
re-sealing them. Nested component-equivalence receipts likewise accept only
the exact legacy or current same-lane inventory, and require the materialized
recipe to be present; this avoids invalidating an older residual-writer receipt
merely because the new recipe reuses its V/O policy.

Manual qualitative-review preparation now accepts any nonempty, consistently
sized set of recipe dossiers and authenticated file pins. Existing three-lane
review inputs remain valid; a later fourth-lane dossier can be merged for the
same blind whole-response review.

## Decision authority

Numerical values may describe and order candidates, but they do not establish
intelligence and are not hard quality gates. The terminal frontier remains
review-pending until decoded, whole-response behavior is manually judged for
coherence, relevance, reasoning, instruction following, and failure modes.
The stated 90% intelligence-retention objective is a target for that holistic
assessment, not an automatic threshold that can publish or discard a child.

## Verification boundary

Unit tests authenticate the complete recipe identity and the intended
component equivalences:

- attention state equals the existing residual-writer attention lane;
- short-convolution state equals the full-source structural lane;
- MLP state equals the residual-writer lane;
- the complete recipe hash is distinct;
- legacy-three and current-four receipt inventories remain readable; and
- the public CLI and qualitative-review preparation accept the new lane.

Those tests prove deterministic composition and receipt integrity only. A
future run must start from the same sealed blank-target foundation, produce a
separate dossier, and be judged alongside the existing lanes by manual decoded
intelligence review before any behavioral claim is made.
