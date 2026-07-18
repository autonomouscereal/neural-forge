# LFM2 CPCF bounded evidence residency boundary

## Status

`dynamic_evidence_residency.py` now supplies a generic, authenticated
current-plus-next layer window. It is intentionally not wired into the CPCF
compiler yet. Wiring the window directly to the current phase objects would
silently change transaction fingerprints, rollback proofs, and resume
authentication.

The window requires an explicit maximum logical-resident-byte budget. Every
layer reference and provider authentication receipt binds the same positive
logical byte count. A current/successor pair that exceeds the budget fails
before the successor is prefetched or promoted. This controls authenticated
payload bytes; Python allocator overhead and operating-system page cache still
require separate process-RSS measurement.

The existing fold remains the reference implementation until every parity
gate below passes. The live fold deployment must not import or use the new
window.

## Current ownership that prevents safe eviction

The present ownership chain is:

1. `Lfm2CpcfInstallabilityAdapter.iter_compiler_inputs()` loads one cache
   batch and creates a full `Lfm2CpcfInstallablePhaseEvidence`.
2. Each phase evidence owns a tuple containing evidence and tensors for every
   source layer plus its final-normalization tensors.
3. `build_lfm2_cpcf_target_phases()` embeds that full object in
   `Lfm2CpcfTargetPhase.parent_evidence`.
4. Every long-lived `Lfm2CpcfPhaseStateBank` retains those same target phase
   objects.
5. `fit_lfm2_cpcf_boundary_bundle()` materializes a second all-layer boundary
   inventory.
6. `compile_and_install_lfm2_cpcf_layer()` reaches the current evidence
   directly through `phase.parent_evidence.layers[layer_index]`.

The following authentication paths also traverse that full object:

- `_model_fold_inputs_sha256()` hashes the complete phase dataclass tree,
  including tensor identities and bytes.
- `_authenticate_model_fold_resume_state()` compares the complete incoming
  phase tree against the newly loaded phases.
- `_live_layer_evidence_sha256()` and `_live_input_bank_sha256()` use direct
  parent-evidence ownership in rollback proofs.
- failure snapshots retain phase references and state-bank inventories derived
  from the same objects.
- `_fit_and_install_final_norm()` reads
  `phase.parent_evidence.final_norm_tensors`.

Consequently, clearing a layer mapping, inserting a lazy proxy, or rebuilding
phases around only the current layer is not a memory optimization under the
current schema. It changes authenticated compiler inputs.

## Required provider split

A safe CPCF integration needs an architecture-specific provider behind the
generic window. It must expose two immutable products.

### Tensor-free phase catalog

The first canonical cache pass must emit lightweight phase references in the
same batch/phase order as today. Each reference must bind:

- cache key, batch index, input SHA-256, and phase name;
- the existing full phase-evidence SHA-256;
- ordered existing layer-evidence SHA-256 values and operator kinds;
- exact logical tensor bytes for every layer payload;
- the final-normalization evidence/inventory SHA-256;
- target tokenizer alignment and existing target phase SHA-256;
- selection, fit, or validation split and row lineage.

This pass may discard tensor payloads after their existing functional checks
and hashes complete. It must not rerun the parent.

### Selective authenticated payload loader

The provider must load only tensor names belonging to one requested layer
across all phases. It must:

- reopen only committed local cache artifacts;
- revalidate completion/record bindings, safetensors metadata and key
  inventory, selected tensor descriptors, and the existing per-layer evidence
  SHA-256;
- return phases in the catalog's exact canonical order;
- report the cache key and provider implementation fingerprint;
- recompute and authenticate the same sealed logical byte count;
- provide deterministic release of all references and mutable staging;
- perform zero parent forwards, zero training, and no arithmetic conversion.

The cache currently authenticates whole batch files and returns all tensor
names. Selective loading therefore requires a dedicated cache API. Reusing
`load_batch()` and discarding unwanted tensors does not provide the intended
RAM bound.

## Compiler API changes

These changes must land together behind a new compiler implementation
fingerprint:

1. Split `Lfm2CpcfInstallablePhaseEvidence` into a tensor-free phase reference
   and independently loaded layer/final-norm payloads.
2. Make `Lfm2CpcfTargetPhase` bind the tensor-free parent reference. Preserve
   its existing logical `phase_sha256` body fields and values.
3. Make state banks own only lightweight target phases and actual-child
   residual states.
4. Change `compile_and_install_lfm2_cpcf_layer()` to receive one explicit
   authenticated layer payload. Remove all direct
   `phase.parent_evidence.layers[index]` access.
5. Change layer rollback proofs to bind the explicit payload's existing layer
   evidence hashes plus the residency authentication receipt.
6. Stream boundary-chart fitting one layer at a time. Final-normalization rows
   may be held as a separate bounded payload for the tied interface and final
   norm. Preserve the current concatenation order, dtype, solver call order,
   and chart report bodies exactly.
7. Replace deep full-phase object traversal in the model-fold transaction with
   an authenticated catalog binding. This necessarily changes the transaction
   input hash representation even if every logical phase/layer SHA remains
   identical, so it must not masquerade as the current compiler fingerprint.
8. Update failure diagnostics and resume authentication to compare catalog
   and phase-reference seals rather than resident tensor object trees.
9. Instantiate `DynamicEvidenceResidencyWindow` at the verified resume layer.
   Promotion order must be `N, N+1, ...`; no prior completed layer may load.

Boundary charts for all layers are currently stored in one resume boundary
bundle. The lowest-risk implementation retains that resume shape: fit charts
through a bounded sequential evidence pass, release each layer, then perform
the compilation pass through the current-plus-next window. This reads cached
layer evidence twice but remains zero-parent-rerun and bounded in private
process memory. A single-pass chart-and-compile design would require a larger
resume schema redesign.

## Required parity and failure tests

Integration is not safe until all of these pass:

1. On a fixed toy cache, compare old and new logical target phase SHA-256
   values, split phase/row hashes, boundary bundle SHA-256, every layer result
   SHA-256, final state-bank SHA-256, final norm report, installed child state
   fingerprint, and whole model-fold result.
2. Prove the new provider returns the same layer evidence objects and tensor
   bytes as the current full `iter_compiler_inputs()` path for every layer and
   phase.
3. Instrument payload ownership and prove peak layer occupancy is at most two
   during compilation and one layer during the chart pass. Measure process RSS
   on the production cache as a separate acceptance check.
4. Resume at every possible layer boundary and prove only current and next
   layer payloads load, while completed receipt chains remain accepted.
5. Corrupt each binding independently: cache key, completion entry, record
   seal, safetensors metadata, tensor key inventory, selected tensor
   descriptor, phase reference, layer reference, provider fingerprint, and
   canonical order. Every case must fail before target mutation.
6. Mutate resident evidence during a layer transaction and prove promotion
   reauthentication fails, the target rolls back, and current/next payloads
   release.
7. Force provider load, authentication, prefetch, and release failures at each
   layer. Prove no skip, duplicate promotion, leaked payload, parent rerun, or
   resume publication with an unauthenticated boundary.
8. Run the existing CPCF boundary, layer compiler, model compiler, resume,
   failure diagnostic, production catalog, and reviewed-gate suites unchanged.

Until these tests pass, the standalone residency abstraction is infrastructure
only and must not be cited as a completed reduction in production fold RSS.
