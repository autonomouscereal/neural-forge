# Historical fresh QIMG paired-MLP writer notes

The numerical acceptance rules below describe the implementation before the
October 6 owner correction. They are historical reference, not the current
workflow. Read [QIMG_OWNER_DIAGNOSTIC_POLICY.md](QIMG_OWNER_DIAGNOSTIC_POLICY.md)
for the current diagnostic-only numerical policy, new receipt schema and
structurally valid unevaluated artifact reader. Historical policy/receipt bytes
remain unchanged; finite out-of-target artifacts now publish without quality or
capability acceptance.

`c16_qimg_fresh_fold.convert_fresh` is the separate capture-admitted writer.
Its result can accept only a numerical paired token-local MLP component. Native
block, media installation, complete child, and retained capability acceptance
remain false. The development CLI and `fold_loaded` API still return metadata
only with acceptance and artifact writing permanently disabled.

## Frozen inputs and execution

Before generation, the recipe must pin the actual writer, development kernel,
FIT selector, all existing capture dependencies, and the exact fold-policy file.
The recipe's stream gates must equal that policy. A changed source or policy
requires a new recipe revision and freeze; historical capture evidence cannot
be promoted by changing a readiness flag. Existing capture v1 comprehensive
family/frozen/gold/media QA is that proposal's scope, not an extra requirement
in the original numerical fold policy. A differently scoped capture contract
needs its own explicit review/version rather than weakening v1.

Call `convert_fresh` with the stage inputs accepted by
`c16_qimg_fresh_capture.admit`, the pinned run receipt and policy, a new output
directory, and an optional new sibling rejection-receipt path. The CLI uses the
same keyword names as flags, with underscores changed to hyphens. `--device`
is `cpu` by default or `cuda:0`. No caller-selected captures or fixture option
exists. Capture refs derive only from the admitted native audited run. Resource
authorization and actual GPU ownership remain external coordinator obligations;
the writer allocates no jobs, stops no other work and authenticates no authority.

The writer preserves the reviewed kernel: both complete BF16 source streams
replay before FIT-only output-norm weighted verbatim channel selection, then
parent-anchored affine ridge fitting. Bias counts toward the fourfold floor.
Stored BF16 affine update must be at most one; pooled and every held forward
must meet the frozen error/cosine gates. Finite guards apply before acceptance.

## Publication and consumption

The two safetensors payloads include retained FP8 source bytes, BF16 fitted
output weights and biases, and separate int64 selection indices. A provisional
receipt remains nonaccepted. Reload validates every recorded hash, tensor key,
shape, dtype, frozen byte and verbatim source row using the existing structural
loader; persisted tensors are scored without a new selector, refit or retry.
Source, recipe, policy and capture admission are rechecked around serialization.

Payloads and a complete staged receipt are fsynced before an exclusive atomic
hard link publishes `fold_receipt.json`. Pending/provisional files grant no
acceptance even if a pending body contains completed acceptance diagnostics.
An interrupted or failed publication preserves partial files plus an honest
rejection receipt reporting staged and canonical-marker presence separately.
The last link gives atomic visibility with no overwrite. A sudden crash may
lose the new canonical directory entry; its post-crash persistence is not
promised. Missing canonical publication means no component acceptance.

Use `load_accepted_component` to consume an accepted component. Supply the exact
canonical receipt/hash, the same pinned run and policy, and stage inputs. This
API rejects renamed, pending, provisional, symlink and fixture receipts before
authority reads; requires the narrow acceptance scope and false larger claims;
re-admits the real capture; verifies frozen/verbatim structural identities; and
rescans persisted held scores and file hashes without fitting. The old
`joint.load_candidate` is a structural loader and does not grant publication or
component acceptance. No API here installs a block, media module or model child.

Synthetic CPU tests use the private fixture path, actual tiny FP8/BF16 weights
and safetensors I/O. Every fixture artifact remains `accepted=false`; invalid
header-only metadata is used solely to exercise consumer refusal. Those tests
are engineering evidence, not real capture, model-quality or authority proof.
