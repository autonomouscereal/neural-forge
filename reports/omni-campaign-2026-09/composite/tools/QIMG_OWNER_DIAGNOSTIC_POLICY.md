# Current QIMG owner policy: complete artifacts, then evaluate capability

The current owner policy is
`chimera_omni.owner_finite_diagnostics_only.2026-10-06.v1`. Finite source-replay
error, held-out error/cosine, stored-update magnitude, compression and conditioning
targets are diagnostics. Exceeding them cannot fail or discard a finite functional
artifact. Capability evaluation follows complete folding and assembly.

Current receipts use `neural_forge.qimg_dual_stream_development_artifact.v2`.
They retain actual `numerical_policy_passed` and
`persisted_numeric_gates_passed` diagnostic booleans, which may be false.
`structural_artifact_valid` describes valid tensors and serialization;
`artifact_admitted` describes genuine capture admission. `accepted`,
`acceptance_enabled`, native block/media/child/capability acceptance remain false.
An emitted development artifact is not evidence of retained intelligence.

Historical v1 policies and rejection receipts are unchanged. The legacy joint
reader still interprets v1 under its recorded numerical pass/update/count rules;
it does not reinterpret a past rejected receipt as a new valid artifact. The
prior October 6 failed fold emitted no tensors; rerunning is needed to create
an artifact under this policy.

## Capture and fitting

`convert_fresh` retains the complete current versioned fresh-capture contract,
including genuine inputs, source/recipe/policy byte pins, chronology and image
splits. Changed writer source needs its own pinned recipe revision. This does
not add those fresh prerequisites to the separate original `convert` path:
verified historical captures can create explicitly unevaluated development
artifacts without being described as fresh data.

The unchanged strong fresh v1 contract declares the historical selector recipe.
The corrected fresh writer explicitly refuses that recipe mismatch until a
reviewed new contract version describes the owner-policy rank fallback. This is
remaining contract engineering, not a new owner approval requirement and not a
blocker for the original historical-capture development conversion. No genuine
fresh production run is claimed supported by this patch alone.

Both streams replay before FIT-only selection and parent-anchored affine ridge.
Normal successful selection is unchanged. The new explicit recipe records a
source-order fallback only for the selector's three known finite rank-shortfall
errors; it never reduces keep or reads held-out rows. Genuine nonempty FIT and
holdout evidence, source geometry, contained indices and memory bounds remain
required. The 2*keep row target and bias-inclusive fourfold floor are measured.
The original Gram route retains its pivots and explicitly fills a shortfall
with unchanged source channels in source order.

The shared `law_abiding_solve` math is unchanged: it filters update>1 ladder
rungs and may retain the prior. This is recorded as a fitting constraint, not
claimed to be an unconstrained optimum or used as an artifact acceptance veto.
The independently measured stored BF16 update may exceed one.

Actual nonfinite inputs, fitted/stored coefficients, or forward outputs remain
unusable. Undefined derived scores or condition metrics are JSON null with
`undefined_diagnostics` classification; diagnostic completeness is false and
unavailable quantities never receive an invented zero or pass.

## Publication and reading

The original `convert` route writes new-only safetensors and a receipt under the
new owner schema, reads back their exact tensor identities and never claims fresh
or quality acceptance. `fold_loaded` can return ephemeral tensors to that writer;
the separate development API/CLI always discards tensors and persists JSON only.

The fresh route preserves selected FP8 first-layer bytes, BF16 fitted output
weights/biases and int64 indices. Reload checks exact shapes, dtypes, tensor/file
hashes, frozen/verbatim bytes, selection and actual parameter counts. Persisted
tensors are rescored without another selector, refit or retry. Source, recipe
and capture admission are rechecked around serialization.

Fresh payloads and staged JSON are fsynced, then an exclusive atomic hard link
publishes canonical `fold_receipt.json`. Provisional/pending files grant no
publication authority. Failures preserve partial bytes and a structural rejection
receipt; no overwrite or cleanup is performed. Power-loss persistence of the new
directory entry is not promised; a missing marker grants no admission.

`load_development_component` reads a genuine canonical nonfixture fresh artifact
and revalidates its pinned capture admission and persisted tensors. It allows
false finite numerical diagnostics. `load_accepted_component` remains only as a
compatibility alias; it grants no quality or capability acceptance. Fixture,
renamed, pending, provisional and symlink receipts remain excluded from genuine
admission. Resource allocation belongs to the coordinator.

Synthetic CPU tests establish implementation contracts, not real QIMG fidelity,
model behavior or intelligence. No model weights, GPU jobs or training are part
of this isolated patch's author validation.
